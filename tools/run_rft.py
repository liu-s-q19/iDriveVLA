import os
import sys
import yaml
import torch
import argparse
import functools
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _inject_navsim_v2() -> None:
    """Prefer NavSim v2 path to avoid importing the in-repo v1.1 code."""
    candidates = []
    env_root = os.environ.get("NAVSIM_DEVKIT_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/data/liushiqi/navsim"))

    for root in candidates:
        if (root / "navsim" / "__init__.py").exists():
            sys.path.insert(0, str(root))
            return


_inject_navsim_v2()

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from models.autovla import GRPOAutoVLA
from models.utils.trainer_progress import build_tqdm_progress_bar
from dataset_utils.rft_dataset import RFTDataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime
import time
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")


torch.set_float32_matmul_precision('high')


def _to_tag(value):
    """Convert config values into filename-safe run_id tags."""
    if value is None:
        return "NA"
    return str(value).replace("/", "-").replace(" ", "")


def _build_run_id(config):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    sample_conf = config.get("training", {}).get("sample", {})
    temp_tag = _to_tag(sample_conf.get("temperature", "NA"))
    top_p_tag = _to_tag(sample_conf.get("top_p", "NA"))
    top_k_tag = _to_tag(sample_conf.get("top_k", "NA"))
    max_steps = int(config.get("training", {}).get("max_steps", -1))
    max_steps_tag = str(max_steps) if max_steps >= 0 else "NA"
    return f"grpo_{ts}_t{temp_tag}_p{top_p_tag}_k{top_k_tag}_ms{max_steps_tag}"


def _resolve_run_id(config):
    """Resolve a single run_id consistently across all local distributed workers."""
    env_run_id = os.environ.get("RFT_RUN_ID")
    if env_run_id:
        return env_run_id

    is_primary = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
    master_port = os.environ.get("MASTER_PORT", "29500")
    run_id_sync_file = Path("/tmp") / f"autovla_rft_run_id_{os.getuid()}_{master_port}.txt"

    if is_primary:
        run_id = _build_run_id(config)
        run_id_sync_file.write_text(run_id, encoding="utf-8")
        return run_id

    deadline = time.time() + 60.0
    while time.time() < deadline:
        if run_id_sync_file.exists():
            run_id = run_id_sync_file.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id
        time.sleep(0.1)

    raise RuntimeError(
        f"Timed out waiting for run_id sync file: {run_id_sync_file}. "
        "Set RFT_RUN_ID explicitly to avoid this issue."
    )


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class GroupSampler(DistributedSampler):
    """
    A sampler for distributed training that returns the same indices on every device.

    This sampler differs from the default DistributedSampler in that every process
    gets the complete set of indices (optionally shuffled deterministically) rather than a subset of them. 

    If the distributed process group is not initialized, the sampler falls back to a
    single-process mode (num_replicas=1, rank=0) to avoid errors.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if not dist.is_initialized():
            num_replicas = 1
            rank = 0
        else:
            num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
            rank = rank if rank is not None else dist.get_rank()

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.num_samples = len(dataset)
        self.total_size = len(dataset)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices)

    def __len__(self):
        return self.total_size


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Dataset and dataloader
    train_dataset = RFTDataset(config['data']['train'], config['model'])
    val_dataset = RFTDataset(config['data']['val'], config['model'])

    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        sampler=GroupSampler(train_dataset, shuffle=True),
        collate_fn=train_dataset.collate_fn,
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        num_workers=config['inference']['num_workers'],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )    

    # Model
    model = GRPOAutoVLA(config)
    # model.load_state_dict(
    #     torch.load(config['sft_model_path'])['state_dict'], 
    #     strict=False
    # )

    # TODO: remove this hard coding
    print(f"Loading and remapping checkpoint from: {config['model']['sft_model_path']}")
    full_checkpoint = torch.load(config['model']['sft_model_path'], map_location="cpu")
    sd = full_checkpoint['state_dict']
    
    # Load the state dict
    msg = model.load_state_dict(sd, strict=False)

    # Create a LoRA configuration. Adjust the parameters (r, lora_alpha, lora_dropout) as needed.
    if config['model']['lora'].get("use", False):
        print("Using LoRA mode for GRPO training.")
        lora_conf = config['model']['lora']
        lora_config = LoraConfig(
            task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
            target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            r=lora_conf.get("r", 8),
            lora_alpha=lora_conf.get("alpha", 8),
            lora_dropout=lora_conf.get("dropout", 0.1),
            bias=lora_conf.get("bias", "none")
        )
        model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)
        print("LoRA-enabled model trainable parameters:",
              sum(p.numel() for p in model.autovla.vlm.parameters() if p.requires_grad))
    model = model.to(torch.bfloat16)

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    run_id = _resolve_run_id(config)
    run_root = PROJECT_ROOT / "runs" / "grpo" / run_id
    ckpt_dir = run_root / "ckpt"
    csv_dir = run_root / "csv"
    tb_dir = PROJECT_ROOT / "tensorboard" / "grpo" / run_id
    manifest_path = run_root / "run_manifest.yaml"
    for path in (ckpt_dir, csv_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(
        save_dir=str(csv_dir.parent),
        name=csv_dir.name,
        version=""
    )
    tb_logger = TensorBoardLogger(
        save_dir=str(tb_dir.parent),
        name=tb_dir.name,
        version=""
    )

    is_primary_process = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
    if is_primary_process:
        manifest = {
            "run_id": run_id,
            "start_time_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "command": f"{sys.executable} {' '.join(sys.argv)}",
            "config_path": f"./config/{args.config}.yaml",
            "paths": {
                "run_root": str(run_root),
                "ckpt_dir": str(ckpt_dir),
                "csv_dir": str(csv_logger.log_dir),
                "tb_dir": str(tb_logger.log_dir),
            },
            "hyperparameters": {
                "learning_rate": config["training"].get("learning_rate"),
                "max_steps": int(config["training"].get("max_steps", -1)),
                "kl_beta": config.get("rl", {}).get("kl_beta"),
                "sample": {
                    "temperature": config.get("training", {}).get("sample", {}).get("temperature"),
                    "top_p": config.get("training", {}).get("sample", {}).get("top_p"),
                    "top_k": config.get("training", {}).get("sample", {}).get("top_k"),
                },
            },
            "save_policy": {
                "every_n_train_steps": 500,
                "save_top_k": -1,
                "save_on_train_epoch_end": False,
                "filename": "rft-step{step}-reward{avg_train_reward:.4f}",
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=False)
        print(f"RUN_ID={run_id}")
        print(f"CKPT_DIR={ckpt_dir}")
        print(f"CSV_DIR={csv_logger.log_dir}")
        print(f"TB_DIR={tb_logger.log_dir}")
        print(f"MANIFEST={manifest_path}")

    distributed_strategy = str(config['training'].get('distributed_strategy', 'ddp')).lower()
    if distributed_strategy == "fsdp":
        trainer_strategy = FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            # Mixed precision training
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            # sharding strategy
            sharding_strategy='FULL_SHARD',
            # prefetching backward computation
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            # save state dict type
            state_dict_type='full', # can be full or sharded
        )
    elif distributed_strategy == "ddp":
        ddp_find_unused = bool(config['training'].get('ddp_find_unused_parameters', True))
        trainer_strategy = "ddp_find_unused_parameters_true" if ddp_find_unused else "ddp_find_unused_parameters_false"
    else:
        raise ValueError(
            f"Unsupported training.distributed_strategy: {distributed_strategy}. "
            f"Expected one of ['ddp', 'fsdp']."
        )
    
    callbacks = [
        ModelCheckpoint(
            monitor="avg_train_reward",
            mode="max",
            save_top_k=-1,
            dirpath=str(ckpt_dir),
            filename="rft-step{step}-reward{avg_train_reward:.4f}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            every_n_train_steps=500,
            save_on_train_epoch_end=False
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    progress_bar_callback = build_tqdm_progress_bar(config.get("training", {}))
    if progress_bar_callback is not None:
        callbacks.append(progress_bar_callback)

    trainer = Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        max_steps=int(config['training'].get('max_steps', -1)),
        accelerator="gpu",
        devices=config['training']['devices'], 
        num_sanity_val_steps=0,
        strategy=trainer_strategy,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        enable_model_summary=True,
        log_every_n_steps=int(config['training'].get('log_every_n_steps', 1)),
        enable_progress_bar=bool(config['training'].get('enable_progress_bar', True)),
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        limit_val_batches=0
    )

    trainer.fit(model, train_dataloaders=train_data)
