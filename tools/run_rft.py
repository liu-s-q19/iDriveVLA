import os
import sys
import csv
import yaml
import torch
import argparse
import functools
import re
from pathlib import Path

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
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from lightning_fabric.utilities.types import _Stateful
from lightning_fabric.loggers.csv_logs import _ExperimentWriter
from lightning_fabric.loggers.logger import rank_zero_experiment

from models.autovla import GRPOAutoVLA
from models.utils.model_backends import detect_model_family
from models.utils.model_backends import resolve_transformer_layer_classes
from models.utils.model_backends import maybe_wrap_with_lora
from models.utils.trainer_progress import build_tqdm_progress_bar
from dataset_utils.rft_dataset import RFTDataset
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
        self._cursor = 0
        self._indices = None
        self._indices_epoch = None
        self._resume_pending = False

    def _build_indices(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return torch.randperm(len(self.dataset), generator=generator).tolist()
        return list(range(len(self.dataset)))

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if self._resume_pending and self._indices is not None and self._cursor < len(self._indices):
            return
        if epoch != self.epoch:
            self.epoch = epoch
            self._cursor = 0
            self._indices = None
            self._indices_epoch = None

    def __iter__(self):
        if self._indices is None or self._indices_epoch != self.epoch:
            self._indices = self._build_indices()
            self._indices_epoch = self.epoch

        start = max(0, min(int(self._cursor), len(self._indices)))
        for i in range(start, len(self._indices)):
            self._cursor = i + 1
            yield self._indices[i]

        self._cursor = 0
        self._resume_pending = False

    def __len__(self):
        return self.total_size

    def state_dict(self):
        if self._indices is None or self._indices_epoch != self.epoch:
            self._indices = self._build_indices()
            self._indices_epoch = self.epoch
        return {
            "epoch": int(self.epoch),
            "cursor": int(self._cursor),
            "num_indices": int(len(self._indices)),
        }

    def load_state_dict(self, state_dict):
        self.epoch = int(state_dict.get("epoch", self.epoch))
        self._indices = self._build_indices()
        self._indices_epoch = self.epoch
        cursor = int(state_dict.get("cursor", 0))
        self._cursor = max(0, min(cursor, len(self._indices)))
        self._resume_pending = self._cursor < len(self._indices)


class StatefulDataLoader(DataLoader):
    """Expose sampler state so Lightning can restore mid-epoch progress."""

    def state_dict(self):
        sampler = getattr(self, "sampler", None)
        if isinstance(sampler, _Stateful):
            return {"sampler": sampler.state_dict()}
        batch_sampler = getattr(self, "batch_sampler", None)
        if isinstance(batch_sampler, _Stateful):
            return {"batch_sampler": batch_sampler.state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        sampler = getattr(self, "sampler", None)
        if "sampler" in state_dict and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state_dict["sampler"])
            return
        batch_sampler = getattr(self, "batch_sampler", None)
        if "batch_sampler" in state_dict and hasattr(batch_sampler, "load_state_dict"):
            batch_sampler.load_state_dict(state_dict["batch_sampler"])


class ResumeStepOffsetCallback(Callback):
    """Keep logger step continuity when resuming from weights-only checkpoints."""

    def __init__(self, step_offset: int):
        super().__init__()
        self.step_offset = int(step_offset)
        self._applied = False

    def on_fit_start(self, trainer, pl_module):
        if self._applied:
            return
        epoch_loop = trainer.fit_loop.epoch_loop
        opt_step_total = epoch_loop.automatic_optimization.optim_progress.optimizer.step.total
        opt_step_total.ready = self.step_offset
        opt_step_total.completed = self.step_offset
        epoch_loop._batches_that_stepped = self.step_offset
        self._applied = True
        if trainer.is_global_zero:
            print(f"[ResumeStepOffset] applied step offset={self.step_offset}")


class AppendableExperimentWriter(_ExperimentWriter):
    """CSV writer that preserves existing metrics when resuming the same run."""

    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        self.hparams_file_path = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)

    def _check_log_dir_exists(self) -> None:
        if self._fs.isfile(self.metrics_file_path):
            with self._fs.open(self.metrics_file_path, "r", newline="") as file:
                reader = csv.DictReader(file)
                self.metrics_keys = list(reader.fieldnames or [])

    def log_hparams(self, params) -> None:
        if self._fs.isfile(self.hparams_file_path):
            return
        with self._fs.open(self.hparams_file_path, "w") as file:
            yaml.safe_dump(dict(params), file, sort_keys=False, allow_unicode=False)


class AppendableCSVLogger(CSVLogger):
    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = AppendableExperimentWriter(log_dir=self.log_dir)
        return self._experiment


class ResumeStepSyncCallback(Callback):
    """Keep logger-facing step counters aligned with restored global_step."""

    def on_train_start(self, trainer, pl_module):
        global_step = int(getattr(trainer, "global_step", 0))
        if global_step <= 0:
            return
        target_step = global_step
        epoch_loop = trainer.fit_loop.epoch_loop
        current_batches = int(getattr(epoch_loop, "_batches_that_stepped", 0))
        if current_batches < target_step:
            epoch_loop._batches_that_stepped = target_step
        opt_step_total = epoch_loop.automatic_optimization.optim_progress.optimizer.step.total
        if int(getattr(opt_step_total, "ready", 0)) < target_step:
            opt_step_total.ready = target_step
        if int(getattr(opt_step_total, "completed", 0)) < target_step:
            opt_step_total.completed = target_step
        if trainer.is_global_zero:
            print(f"[ResumeStepSync] synced logger step to target_step={target_step} (global_step={global_step})")


class DataLoaderStateCheckpointCallback(Callback):
    """Persist train dataloader state explicitly for strict mid-epoch resume."""

    def __init__(self):
        super().__init__()
        self._train_dataloader_state = None

    @staticmethod
    def _extract_train_dataloader(trainer):
        dataloader = getattr(trainer, "train_dataloader", None)
        if callable(dataloader):
            dataloader = dataloader()
        if isinstance(dataloader, (list, tuple)):
            return dataloader[0] if dataloader else None
        iterables = getattr(dataloader, "iterables", None)
        if isinstance(iterables, (list, tuple)):
            return iterables[0] if iterables else None
        if isinstance(iterables, dict):
            return next(iter(iterables.values())) if iterables else None
        return dataloader

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        train_dataloader = self._extract_train_dataloader(trainer)
        if hasattr(train_dataloader, "state_dict"):
            checkpoint["autovla_train_dataloader_state"] = train_dataloader.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self._train_dataloader_state = checkpoint.get("autovla_train_dataloader_state")

    def on_fit_start(self, trainer, pl_module):
        if not self._train_dataloader_state:
            return
        train_dataloader = self._extract_train_dataloader(trainer)
        if hasattr(train_dataloader, "load_state_dict"):
            train_dataloader.load_state_dict(self._train_dataloader_state)
            if trainer.is_global_zero:
                print("[DataLoaderStateCheckpoint] restored train dataloader state")


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
    max_steps = config["training"].get("max_steps", -1)
    if max_steps is None:
        max_steps = -1
    else:
        max_steps = int(max_steps)
        if max_steps <= 0:
            max_steps = -1
    log_every_n_steps = max(1, int(config["training"].get("log_every_n_steps", 1)))
    progress_bar_refresh_rate = max(1, int(config["training"].get("progress_bar_refresh_rate", 10)))
    checkpoint_every_n_train_steps = max(1, int(config["training"].get("checkpoint_every_n_train_steps", 500)))
    checkpoint_save_on_train_epoch_end = bool(config["training"].get("checkpoint_save_on_train_epoch_end", False))
    checkpoint_save_last = bool(config["training"].get("checkpoint_save_last", False))
    save_weights_only = bool(config["training"].get("save_weights_only", False))
    resume_ckpt_path = config["training"].get("resume_ckpt_path")
    if resume_ckpt_path is not None:
        resume_ckpt_path = str(resume_ckpt_path).strip()
        if not resume_ckpt_path:
            resume_ckpt_path = None
        elif not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"resume_ckpt_path not found: {resume_ckpt_path}")
    resume_weights_path = config["training"].get("resume_weights_path")
    if resume_weights_path is not None:
        resume_weights_path = str(resume_weights_path).strip()
        if not resume_weights_path:
            resume_weights_path = None
        elif not os.path.isfile(resume_weights_path):
            raise FileNotFoundError(f"resume_weights_path not found: {resume_weights_path}")
    resume_global_step = config["training"].get("resume_global_step")
    if resume_global_step is not None:
        resume_global_step = int(resume_global_step)
    elif resume_ckpt_path is not None or resume_weights_path is not None:
        step_source_path = resume_ckpt_path if resume_ckpt_path is not None else resume_weights_path
        step_match = re.search(r"step(\d+)", os.path.basename(step_source_path))
        if step_match:
            resume_global_step = int(step_match.group(1))

    # Dataset and dataloader
    train_dataset = RFTDataset(config['data']['train'], config['model'])
    val_dataset = RFTDataset(config['data']['val'], config['model'])

    train_num_workers = int(config["training"]["num_workers"])
    train_loader_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": train_num_workers,
        "sampler": GroupSampler(train_dataset, shuffle=True),
        "collate_fn": train_dataset.collate_fn,
        "pin_memory": bool(config["training"].get("pin_memory", False)),
    }
    if train_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = bool(config["training"].get("persistent_workers", False))
        if "prefetch_factor" in config["training"]:
            train_loader_kwargs["prefetch_factor"] = int(config["training"]["prefetch_factor"])

    train_data = StatefulDataLoader(train_dataset, **train_loader_kwargs)

    val_num_workers = int(config["inference"]["num_workers"])
    val_loader_kwargs = {
        "batch_size": config["inference"]["batch_size"],
        "num_workers": val_num_workers,
        "shuffle": False,
        "collate_fn": val_dataset.collate_fn,
        "pin_memory": bool(config["inference"].get("pin_memory", False)),
    }
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = bool(config["inference"].get("persistent_workers", False))
        if "prefetch_factor" in config["inference"]:
            val_loader_kwargs["prefetch_factor"] = int(config["inference"]["prefetch_factor"])

    val_data = StatefulDataLoader(val_dataset, **val_loader_kwargs)

    # Model
    model = GRPOAutoVLA(config)
    # model.load_state_dict(
    #     torch.load(config['sft_model_path'])['state_dict'], 
    #     strict=False
    # )

    # TODO: remove this hard coding
    init_weights_path = resume_weights_path if resume_weights_path is not None else config['model']['sft_model_path']
    print(f"Loading and remapping checkpoint from: {init_weights_path}")
    full_checkpoint = torch.load(init_weights_path, map_location="cpu")
    sd = full_checkpoint['state_dict'] if isinstance(full_checkpoint, dict) and 'state_dict' in full_checkpoint else full_checkpoint
    
    # Load the state dict
    msg = model.load_state_dict(sd, strict=False)

    if config['model'].get('lora', {}).get("use", False):
        print("Using LoRA mode for GRPO training.")
        model.autovla.vlm, _ = maybe_wrap_with_lora(model.autovla.vlm, config['model']['lora'])
        print(
            "LoRA-enabled model trainable parameters:",
            sum(p.numel() for p in model.autovla.vlm.parameters() if p.requires_grad),
        )
    model = model.to(torch.bfloat16)

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(
            resolve_transformer_layer_classes(
                detect_model_family(config['model']['pretrained_model_path'])
            )
        ),
    )

    run_id = _resolve_run_id(config)
    run_root = PROJECT_ROOT / "runs" / "grpo" / run_id
    ckpt_dir = run_root / "ckpt"
    csv_dir = run_root / "csv"
    tb_dir = PROJECT_ROOT / "tensorboard" / "grpo" / run_id
    manifest_path = run_root / "run_manifest.yaml"
    for path in (ckpt_dir, csv_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=True)

    tb_root = str(config["training"].get("tensorboard_root", str(PROJECT_ROOT / "tensorboard" / "grpo")))
    tb_name = str(config["training"].get("tensorboard_name", ""))
    tb_version = config["training"].get("tensorboard_version", run_id)
    if tb_version is None or str(tb_version).strip() == "":
        tb_version = run_id
    tb_version = str(tb_version)
    tb_dir = Path(tb_root)
    if tb_name:
        tb_dir = tb_dir / tb_name
    tb_dir = tb_dir / tb_version

    csv_logger = AppendableCSVLogger(
        save_dir=str(csv_dir.parent),
        name=csv_dir.name,
        version=""
    )
    tb_logger = TensorBoardLogger(
        save_dir=tb_root,
        name=tb_name,
        version=tb_version,
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
                "every_n_train_steps": checkpoint_every_n_train_steps,
                "save_top_k": -1,
                "save_on_train_epoch_end": checkpoint_save_on_train_epoch_end,
                "save_last": checkpoint_save_last,
                "monitor": "train_reward",
                "filename": "rft-step{step}-reward{train_reward:.4f}",
                "save_weights_only": save_weights_only,
            },
            "resume": {
                "resume_ckpt_path": resume_ckpt_path,
                "resume_weights_path": resume_weights_path,
                "resume_global_step": resume_global_step,
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
        DataLoaderStateCheckpointCallback(),
        ResumeStepSyncCallback(),
        ModelCheckpoint(
            monitor="train_reward",
            mode="max",
            save_top_k=-1,
            dirpath=str(ckpt_dir),
            filename="rft-step{step}-reward{train_reward:.4f}",
            auto_insert_metric_name=False,
            save_weights_only=save_weights_only,
            every_n_train_steps=checkpoint_every_n_train_steps,
            save_on_train_epoch_end=checkpoint_save_on_train_epoch_end,
            save_last=checkpoint_save_last,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    fit_ckpt_path = resume_ckpt_path
    if resume_weights_path is not None and resume_ckpt_path is not None:
        print("resume_weights_path is set; ignoring resume_ckpt_path to avoid conflicting restore modes.")
        fit_ckpt_path = None
    if resume_global_step is not None and fit_ckpt_path is None:
        callbacks.append(ResumeStepOffsetCallback(resume_global_step))
    progress_bar_callback = build_tqdm_progress_bar(config.get("training", {}))
    if progress_bar_callback is not None:
        callbacks.append(progress_bar_callback)

    trainer = Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        max_steps=max_steps,
        accelerator="gpu",
        devices=config['training']['devices'], 
        num_sanity_val_steps=0,
        strategy=trainer_strategy,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        enable_model_summary=True,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=bool(config['training'].get('enable_progress_bar', True)),
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        limit_val_batches=0
    )

    trainer.fit(model, train_dataloaders=train_data, ckpt_path=fit_ckpt_path)
