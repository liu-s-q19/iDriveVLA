import types
import unittest

import torch

from models.autovla import GRPOAutoVLA


class TestGRPOInvalidRewardPenalty(unittest.TestCase):
    def _build_module(self, cfg):
        module = object.__new__(GRPOAutoVLA)
        module.cfg = cfg
        module.use_cot = False
        module.log = types.MethodType(lambda self, *args, **kwargs: None, module)
        param = torch.nn.Parameter(torch.zeros(1))
        module.parameters = types.MethodType(lambda self: iter([param]), module)
        return module

    def test_invalid_reward_uses_configured_small_penalty(self):
        module = self._build_module({"rl": {"reward": {"invalid_penalty": -0.1}}})

        reward = module.reward_function({"reward_input_valid": 0.0})

        self.assertAlmostEqual(float(reward.item()), -0.1, places=6)

    def test_invalid_reward_defaults_to_zero_when_penalty_missing(self):
        module = self._build_module({"rl": {"reward": {}}})

        reward = module.reward_function({"reward_input_valid": 0.0})

        self.assertAlmostEqual(float(reward.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
