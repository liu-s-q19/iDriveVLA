import os
import sys
import yaml
import torch
import argparse
import functools
import re
from peft import get_peft_model, LoraConfig, TaskType

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
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

from models.autovla import GRPOAutoVLA
from dataset_utils.rft_dataset import RFTDataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")


torch.set_float32_matmul_precision('high')

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

    def _build_indices(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return torch.randperm(len(self.dataset), generator=generator).tolist()
        return list(range(len(self.dataset)))

    def set_epoch(self, epoch):
        # Lightning calls set_epoch at every epoch start. Keep cursor when epoch is unchanged
        # so mid-epoch checkpoint resume can continue from the exact sampler position.
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self._cursor = 0
            self._indices = None
            self._indices_epoch = None

    def __iter__(self):
        if self._indices is None or self._indices_epoch != self.epoch:
            self._indices = self._build_indices()
            self._indices_epoch = self.epoch

        start = int(self._cursor)
        if start < 0:
            start = 0
        if start > len(self._indices):
            start = len(self._indices)

        for i in range(start, len(self._indices)):
            self._cursor = i + 1
            yield self._indices[i]

        # End-of-epoch: reset for the next epoch unless a checkpoint restores this field.
        self._cursor = 0

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
        epoch = int(state_dict.get("epoch", self.epoch))
        self.epoch = epoch
        self._indices = self._build_indices()
        self._indices_epoch = self.epoch
        cursor = int(state_dict.get("cursor", 0))
        if cursor < 0:
            cursor = 0
        if cursor > len(self._indices):
            cursor = len(self._indices)
        self._cursor = cursor


class StatefulDataLoader(DataLoader):
    """DataLoader wrapper exposing sampler state for Lightning mid-epoch checkpoint resume."""

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
    """Apply a global-step offset for weight-only resume so logger steps continue."""
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
        # TensorBoard logger uses this internal counter as default step.
        epoch_loop._batches_that_stepped = self.step_offset
        self._applied = True
        if trainer.is_global_zero:
            print(f"[ResumeStepOffset] applied step offset={self.step_offset}")


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
    max_steps = config['training'].get('max_steps', -1)
    if max_steps is None:
        max_steps = -1
    else:
        max_steps = int(max_steps)
        if max_steps <= 0:
            max_steps = -1

    log_every_n_steps = int(config['training'].get('log_every_n_steps', 1))
    if log_every_n_steps < 1:
        log_every_n_steps = 1
    progress_bar_refresh_rate = int(config['training'].get('progress_bar_refresh_rate', 10))
    if progress_bar_refresh_rate < 1:
        progress_bar_refresh_rate = 1
    resume_ckpt_path = config['training'].get('resume_ckpt_path', None)
    if resume_ckpt_path is not None:
        resume_ckpt_path = str(resume_ckpt_path).strip()
        if resume_ckpt_path == "":
            resume_ckpt_path = None
        elif not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"resume_ckpt_path not found: {resume_ckpt_path}")
    resume_weights_path = config['training'].get('resume_weights_path', None)
    if resume_weights_path is not None:
        resume_weights_path = str(resume_weights_path).strip()
        if resume_weights_path == "":
            resume_weights_path = None
        elif not os.path.isfile(resume_weights_path):
            raise FileNotFoundError(f"resume_weights_path not found: {resume_weights_path}")
    resume_global_step = config['training'].get('resume_global_step', None)
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

    train_data = StatefulDataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        sampler=GroupSampler(train_dataset, shuffle=True),
        collate_fn=train_dataset.collate_fn,
    )

    val_data = StatefulDataLoader(
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
    init_weights_path = resume_weights_path if resume_weights_path is not None else config['model']['sft_model_path']
    print(f"Loading and remapping checkpoint from: {init_weights_path}")
    full_checkpoint = torch.load(init_weights_path, map_location="cpu")
    sd = full_checkpoint['state_dict'] if isinstance(full_checkpoint, dict) and 'state_dict' in full_checkpoint else full_checkpoint
    
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

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"runs/grpo/{current_date}"
    tb_root = str(config['training'].get('tensorboard_root', 'runs/tensorboard/grpo'))
    tb_name = config['training'].get('tensorboard_name', '')
    if tb_name is None:
        tb_name = ''
    tb_name = str(tb_name)
    tb_version = config['training'].get('tensorboard_version', current_date)
    if tb_version is None or str(tb_version).strip() == "":
        tb_version = current_date
    tb_version = str(tb_version)
    
    checkpoint_every_n_train_steps = int(config['training'].get('checkpoint_every_n_train_steps', 500))
    if checkpoint_every_n_train_steps < 1:
        checkpoint_every_n_train_steps = 1
    save_weights_only = bool(config['training'].get('save_weights_only', False))

    trainer_callbacks = [
            ModelCheckpoint(
                monitor="avg_train_reward",
                mode="max",
                save_top_k=-1,
                dirpath=f"{save_dir}",
                filename="rft-step{step}-reward{avg_train_reward:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=save_weights_only,
                every_n_train_steps=checkpoint_every_n_train_steps,
                save_on_train_epoch_end=False
            ),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=progress_bar_refresh_rate),
        ]
    fit_ckpt_path = resume_ckpt_path
    if resume_weights_path is not None and resume_ckpt_path is not None:
        print("resume_weights_path is set, ignore resume_ckpt_path to avoid incompatible full-state restore.")
        fit_ckpt_path = None
    use_weight_only_resume = (resume_global_step is not None and fit_ckpt_path is None)
    if use_weight_only_resume:
        trainer_callbacks.append(ResumeStepOffsetCallback(resume_global_step))

    trainer = Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        max_steps=max_steps,
        accelerator="gpu",
        devices=config['training']['devices'], 
        num_sanity_val_steps=0,
        strategy=FSDPStrategy(
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
        ),
        callbacks=trainer_callbacks,
        logger=[
            CSVLogger(save_dir="runs/"),
            TensorBoardLogger(save_dir=tb_root, name=tb_name, version=tb_version),
        ],
        enable_model_summary=True,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        limit_val_batches=0
    )

    trainer.fit(model, train_dataloaders=train_data, ckpt_path=fit_ckpt_path)
