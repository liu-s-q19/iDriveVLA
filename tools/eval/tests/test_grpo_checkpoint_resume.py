import types
import unittest

import torch

from models.autovla import GRPOAutoVLA


class TestGRPOCheckpointResume(unittest.TestCase):
    def test_on_save_checkpoint_keeps_reference_model_state(self):
        module = object.__new__(GRPOAutoVLA)
        checkpoint = {
            "state_dict": {
                "reference_model.layer.weight": torch.tensor([1.0]),
                "autovla.layer.weight": torch.tensor([2.0]),
            }
        }

        module.on_save_checkpoint(checkpoint)

        self.assertIn("reference_model.layer.weight", checkpoint["state_dict"])

    def test_on_load_checkpoint_backfills_missing_reference_model_state(self):
        module = object.__new__(GRPOAutoVLA)
        module.reference_model = types.SimpleNamespace(
            state_dict=lambda: {
                "layer.weight": torch.tensor([3.0]),
                "layer.bias": torch.tensor([4.0]),
            }
        )
        checkpoint = {"state_dict": {"autovla.layer.weight": torch.tensor([2.0])}}

        module.on_load_checkpoint(checkpoint)

        self.assertIn("reference_model.layer.weight", checkpoint["state_dict"])
        self.assertIn("reference_model.layer.bias", checkpoint["state_dict"])


if __name__ == "__main__":
    unittest.main()
