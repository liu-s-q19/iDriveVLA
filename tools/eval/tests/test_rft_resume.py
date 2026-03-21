import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tools.run_rft import (
    AppendableCSVLogger,
    DataLoaderStateCheckpointCallback,
    GroupSampler,
    ResumeStepSyncCallback,
    StatefulDataLoader,
)


class _TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return idx


class _LongDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return idx


class _TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.seen_batches = []

    def training_step(self, batch, batch_idx):
        batch = batch.float().view(-1, 1)
        self.seen_batches.extend(int(x) for x in batch.view(-1).tolist())
        self.log("toy_metric", batch.mean(), on_step=True, on_epoch=False)
        loss = self.layer(batch).sum() * 0.0
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class TestGroupSamplerResume(unittest.TestCase):
    @staticmethod
    def _collect_tensorboard_steps(log_dir: Path, tag: str):
        steps = []
        for event_file in sorted(log_dir.glob("events.out.tfevents.*")):
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            steps.extend(item.step for item in accumulator.Scalars(tag))
        return steps

    def test_sampler_state_dict_restores_mid_epoch_cursor(self):
        sampler = GroupSampler(_TinyDataset(), shuffle=False)

        iterator = iter(sampler)
        first_two = [next(iterator), next(iterator)]
        state = sampler.state_dict()

        resumed = GroupSampler(_TinyDataset(), shuffle=False)
        resumed.load_state_dict(state)

        self.assertEqual(first_two, [0, 1])
        self.assertEqual(list(iter(resumed)), [2, 3, 4])

    def test_sampler_state_resets_when_epoch_changes(self):
        sampler = GroupSampler(_TinyDataset(), shuffle=False)

        iterator = iter(sampler)
        _ = next(iterator)
        sampler.set_epoch(1)

        self.assertEqual(list(iter(sampler)), [0, 1, 2, 3, 4])

    def test_lightning_resume_restores_sampler_cursor(self):
        with TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            dataloader = StatefulDataLoader(
                _TinyDataset(),
                batch_size=1,
                sampler=GroupSampler(_TinyDataset(), shuffle=False),
                num_workers=0,
            )

            first_model = _TinyModule()
            first_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=2,
                default_root_dir=tmpdir,
                logger=False,
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[
                    DataLoaderStateCheckpointCallback(),
                    ModelCheckpoint(
                        dirpath=str(ckpt_dir),
                        filename="resume-step{step}",
                        every_n_train_steps=2,
                        save_top_k=-1,
                        save_on_train_epoch_end=False,
                    ),
                ],
            )
            first_trainer.fit(first_model, train_dataloaders=dataloader)
            ckpt_path = next(ckpt_dir.glob("*.ckpt"))

            resumed_model = _TinyModule()
            resumed_dataloader = StatefulDataLoader(
                _TinyDataset(),
                batch_size=1,
                sampler=GroupSampler(_TinyDataset(), shuffle=False),
                num_workers=0,
            )
            resumed_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=4,
                default_root_dir=tmpdir,
                logger=False,
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[DataLoaderStateCheckpointCallback()],
            )
            resumed_trainer.fit(
                resumed_model,
                train_dataloaders=resumed_dataloader,
                ckpt_path=str(ckpt_path),
            )

            self.assertEqual(resumed_trainer.global_step, 4)
            self.assertEqual(resumed_model.seen_batches[:2], [2, 3])

    def test_lightning_resume_restores_loader_position_at_step_ten(self):
        with TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            dataloader = StatefulDataLoader(
                _LongDataset(),
                batch_size=1,
                sampler=GroupSampler(_LongDataset(), shuffle=False),
                num_workers=0,
            )

            first_model = _TinyModule()
            first_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=10,
                default_root_dir=tmpdir,
                logger=False,
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[
                    DataLoaderStateCheckpointCallback(),
                    ModelCheckpoint(
                        dirpath=str(ckpt_dir),
                        filename="resume-step{step}",
                        every_n_train_steps=10,
                        save_top_k=-1,
                        save_on_train_epoch_end=False,
                    ),
                ],
            )
            first_trainer.fit(first_model, train_dataloaders=dataloader)
            ckpt_path = next(ckpt_dir.glob("*.ckpt"))

            resumed_model = _TinyModule()
            resumed_dataloader = StatefulDataLoader(
                _LongDataset(),
                batch_size=1,
                sampler=GroupSampler(_LongDataset(), shuffle=False),
                num_workers=0,
            )
            resumed_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=12,
                default_root_dir=tmpdir,
                logger=False,
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[DataLoaderStateCheckpointCallback()],
            )
            resumed_trainer.fit(
                resumed_model,
                train_dataloaders=resumed_dataloader,
                ckpt_path=str(ckpt_path),
            )

            self.assertEqual(resumed_trainer.global_step, 12)
            self.assertEqual(resumed_model.seen_batches[:2], [10, 11])

    def test_resume_preserves_csv_and_tensorboard_step_history(self):
        with TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpt"
            csv_dir = Path(tmpdir) / "csv"
            tb_dir = Path(tmpdir) / "tb"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            dataloader = StatefulDataLoader(
                _TinyDataset(),
                batch_size=1,
                sampler=GroupSampler(_TinyDataset(), shuffle=False),
                num_workers=0,
            )

            logger_callbacks = [
                DataLoaderStateCheckpointCallback(),
                ResumeStepSyncCallback(),
            ]
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="resume-step{step}",
                every_n_train_steps=2,
                save_top_k=-1,
                save_on_train_epoch_end=False,
            )

            first_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=2,
                default_root_dir=tmpdir,
                logger=[
                    AppendableCSVLogger(save_dir=str(csv_dir.parent), name=csv_dir.name, version=""),
                    TensorBoardLogger(save_dir=str(tb_dir.parent), name=tb_dir.name, version=""),
                ],
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
                callbacks=logger_callbacks + [checkpoint_callback],
            )
            first_trainer.fit(_TinyModule(), train_dataloaders=dataloader)
            ckpt_path = next(ckpt_dir.glob("*.ckpt"))

            resumed_dataloader = StatefulDataLoader(
                _TinyDataset(),
                batch_size=1,
                sampler=GroupSampler(_TinyDataset(), shuffle=False),
                num_workers=0,
            )
            resumed_trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=4,
                default_root_dir=tmpdir,
                logger=[
                    AppendableCSVLogger(save_dir=str(csv_dir.parent), name=csv_dir.name, version=""),
                    TensorBoardLogger(save_dir=str(tb_dir.parent), name=tb_dir.name, version=""),
                ],
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
                callbacks=[DataLoaderStateCheckpointCallback(), ResumeStepSyncCallback()],
            )
            resumed_trainer.fit(
                _TinyModule(),
                train_dataloaders=resumed_dataloader,
                ckpt_path=str(ckpt_path),
            )

            metrics_path = csv_dir / "metrics.csv"
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                csv_steps = [int(row["step"]) for row in reader if row.get("step")]
            tb_steps = self._collect_tensorboard_steps(tb_dir, "toy_metric")

            self.assertEqual(csv_steps, [0, 1, 2, 3])
            self.assertEqual(tb_steps, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
