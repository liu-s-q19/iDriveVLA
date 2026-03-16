import unittest

import torch

from models.utils.grpo_metrics import masked_token_mean


class TestGRPOMetrics(unittest.TestCase):
    def test_masked_token_mean_ignores_masked_positions(self):
        values = torch.tensor([[1.0, 2.0, 100.0], [3.0, 9.0, 12.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        result = masked_token_mean(values, mask)

        self.assertTrue(torch.isclose(result, torch.tensor(2.25)))


if __name__ == "__main__":
    unittest.main()
