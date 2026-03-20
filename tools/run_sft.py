import sys
import os
from pathlib import Path

# Add project root to path for imports
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

import yaml
import torch
import argparse
import functools
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader

from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA
from models.utils.model_backends import detect_model_family
from models.utils.model_backends import load_processor_for_model
from models.utils.model_backends import maybe_wrap_with_lora
from models.utils.model_backends import resolve_transformer_layer_classes
from models.utils.model_backends import set_gradient_checkpointing_for_model
from models.utils.trainer_progress import build_tqdm_progress_bar
import datetime

torch.set_float32_matmul_precision('high')


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def resolve_trainer_devices(configured_devices, launched_by_torchrun):
    if launched_by_torchrun:
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        if local_world_size:
            return int(local_world_size)
        if isinstance(configured_devices, int):
            return configured_devices
        if isinstance(configured_devices, list):
            return len(configured_devices)
        if isinstance(configured_devices, str):
            if configured_devices.lower() == "auto":
                gpu_count = torch.cuda.device_count()
                return gpu_count if gpu_count > 0 else 1
            return int(configured_devices)
        return 1

    if isinstance(configured_devices, str):
        if configured_devices.lower() in ("auto", "-1"):
            return "auto"
        return int(configured_devices)
    return configured_devices


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")
    training_cfg = config.get("training", {})

    # Multi-node / torchrun compatibility:
    # - Keep Trainer devices aligned with LOCAL_WORLD_SIZE.
    # - num_nodes can be overridden by PL_NUM_NODES env in launch scripts.
    launched_by_torchrun = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ
    trainer_num_nodes = int(os.environ.get("PL_NUM_NODES", training_cfg.get("num_nodes", 1)))
    trainer_devices = resolve_trainer_devices(
        training_cfg.get("devices", "auto"),
        launched_by_torchrun,
    )

    # Model, dataset, and dataloader
    processor = load_processor_for_model(config['model']['pretrained_model_path'])
    
    # Get using_cot setting from config (default to True if not specified)
    using_cot = config['model']['use_cot']
    

    train_dataset = SFTDataset(config['data']['train'], config['model'], processor, using_cot=using_cot)
        
    # Randomly sample from training set if train_sample_size is specified
    train_sample_size = config['training']['train_sample_size']
    if train_sample_size is not None and len(train_dataset) > train_sample_size:
        indices = torch.randperm(len(train_dataset))[:train_sample_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    else:
        print("no sampling")
        
    val_dataset = SFTDataset(config['data']['val'], config['model'], processor, using_cot=using_cot)

    model = SFTAutoVLA(config)
    model.autovla.vlm, sft_lora_enabled = maybe_wrap_with_lora(
        model.autovla.vlm,
        config.get("model", {}).get("lora"),
    )
    if sft_lora_enabled:
        trainable_params = sum(p.numel() for p in model.autovla.vlm.parameters() if p.requires_grad)
        print(f"Using LoRA mode for SFT training. trainable_params={trainable_params}")

    # Gradient checkpointing can reduce memory but significantly slows throughput.
    # Keep it configurable so large multi-node jobs can prioritize speed.
    use_gradient_checkpointing = bool(training_cfg.get("gradient_checkpointing", True))
    set_gradient_checkpointing_for_model(
        model.autovla.vlm,
        enabled=use_gradient_checkpointing,
    )

    if hasattr(model.autovla.vlm, "config") and hasattr(model.autovla.vlm.config, "use_cache"):
        model.autovla.vlm.config.use_cache = not use_gradient_checkpointing

    # checkpoint_path = Path(".ckpt")
    # state_dict = torch.load(checkpoint_path)['state_dict']
    # model.load_state_dict(state_dict)
    
    # Create data collator with config parameters
    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id']
    )
    
    train_num_workers = int(training_cfg.get('num_workers', 0))
    train_pin_memory = bool(training_cfg.get('pin_memory', True))
    train_persistent_workers = bool(training_cfg.get('persistent_workers', train_num_workers > 0))
    train_prefetch_factor = training_cfg.get('prefetch_factor', 2)
    train_loader_kwargs = {}
    if train_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = train_persistent_workers
        if train_prefetch_factor is not None:
            train_loader_kwargs["prefetch_factor"] = int(train_prefetch_factor)

    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        num_workers=train_num_workers,
        pin_memory=train_pin_memory,
        shuffle=True,
        **train_loader_kwargs,
    )

    val_num_workers = int(config['inference'].get('num_workers', 0))
    val_pin_memory = bool(config['inference'].get('pin_memory', True))
    val_persistent_workers = bool(config['inference'].get('persistent_workers', val_num_workers > 0))
    val_prefetch_factor = config['inference'].get('prefetch_factor', 2)
    val_loader_kwargs = {}
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = val_persistent_workers
        if val_prefetch_factor is not None:
            val_loader_kwargs["prefetch_factor"] = int(val_prefetch_factor)

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        num_workers=val_num_workers,
        pin_memory=val_pin_memory,
        shuffle=False,
        **val_loader_kwargs,
    )    

    # Distributed strategy
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(
            resolve_transformer_layer_classes(
                detect_model_family(config['model']['pretrained_model_path'])
            )
        ),
    )
    strategy_name = str(training_cfg.get("distributed_strategy", "fsdp")).lower()
    if strategy_name == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=bool(training_cfg.get("ddp_find_unused_parameters", False))
        )
    elif strategy_name == "fsdp":
        fsdp_cfg = training_cfg.get("fsdp", {})
        backward_prefetch_name = str(fsdp_cfg.get("backward_prefetch", "BACKWARD_PRE")).upper()
        if not hasattr(BackwardPrefetch, backward_prefetch_name):
            raise ValueError(f"Unknown FSDP backward_prefetch={backward_prefetch_name}")
        strategy = FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            sharding_strategy=fsdp_cfg.get('sharding_strategy', 'FULL_SHARD'),
            backward_prefetch=getattr(BackwardPrefetch, backward_prefetch_name),
            state_dict_type=fsdp_cfg.get("state_dict_type", "full"),
            limit_all_gathers=bool(fsdp_cfg.get("limit_all_gathers", True)),
        )
    elif strategy_name in ("auto", "none"):
        strategy = "auto"
    else:
        raise ValueError(f"Unsupported distributed_strategy={strategy_name} (expected ddp/fsdp/auto)")

    current_date = os.environ.get("SFT_RUN_TS", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_dir = f"runs/sft/{current_date}"
    
    logging_cfg = training_cfg.get("logging", {})
    logger_type = str(logging_cfg.get("type", "tensorboard")).lower()
    if logger_type == "tensorboard":
        trainer_logger = TensorBoardLogger(save_dir=save_dir, name="lightning_logs")
    elif logger_type == "csv":
        trainer_logger = CSVLogger(save_dir=save_dir, name="lightning_logs")
    elif logger_type == "both":
        trainer_logger = [
            TensorBoardLogger(save_dir=save_dir, name="tb_logs"),
            CSVLogger(save_dir=save_dir, name="csv_logs"),
        ]
    else:
        raise ValueError(
            f"Unsupported training.logging.type={logger_type} (expected tensorboard/csv/both)"
        )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            dirpath=f"{save_dir}",
            filename="epoch={epoch}-loss={val_loss:.4f}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            every_n_epochs=1,
        ),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]
    progress_bar_callback = build_tqdm_progress_bar(training_cfg)
    if progress_bar_callback is not None:
        callbacks.append(progress_bar_callback)

    trainer = pl.Trainer(
        num_nodes=trainer_num_nodes,
        max_epochs=config['training']['epochs'],
        accelerator="gpu",
        devices=trainer_devices,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy=strategy,
        callbacks=callbacks,
        gradient_clip_algorithm = 'value',
        gradient_clip_val = 1.0,

        logger=trainer_logger,
        enable_model_summary=True,
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 50)),
        enable_progress_bar=bool(training_cfg.get("enable_progress_bar", True)),

        # limit_val_batches=0.001
    )
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
