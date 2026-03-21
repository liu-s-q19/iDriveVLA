import unittest


class TestRunSFTStrategyConfig(unittest.TestCase):
    def test_internvl_ddp_does_not_force_static_graph_by_default(self):
        from tools.run_sft import build_training_strategy

        strategy = build_training_strategy(
            training_cfg={
                "distributed_strategy": "ddp",
                "ddp_find_unused_parameters": True,
            },
            model_family="internvl_chat",
        )

        self.assertEqual(strategy._ddp_kwargs.get("find_unused_parameters"), True)
        self.assertNotIn("static_graph", strategy._ddp_kwargs)

    def test_internvl_explicit_static_graph_disables_find_unused_parameters(self):
        from tools.run_sft import build_training_strategy

        strategy = build_training_strategy(
            training_cfg={
                "distributed_strategy": "ddp",
                "ddp_find_unused_parameters": True,
                "ddp_static_graph": True,
            },
            model_family="internvl_chat",
        )

        self.assertEqual(strategy._ddp_kwargs.get("static_graph"), True)
        self.assertEqual(strategy._ddp_kwargs.get("find_unused_parameters"), False)

    def test_qwen_ddp_does_not_force_static_graph(self):
        from tools.run_sft import build_training_strategy

        strategy = build_training_strategy(
            training_cfg={
                "distributed_strategy": "ddp",
                "ddp_find_unused_parameters": False,
            },
            model_family="qwen2_5_vl",
        )

        self.assertEqual(strategy._ddp_kwargs.get("find_unused_parameters"), False)
        self.assertNotIn("static_graph", strategy._ddp_kwargs)

    def test_config_can_disable_static_graph_override(self):
        from tools.run_sft import build_training_strategy

        strategy = build_training_strategy(
            training_cfg={
                "distributed_strategy": "ddp",
                "ddp_find_unused_parameters": True,
                "ddp_static_graph": False,
            },
            model_family="internvl_chat",
        )

        self.assertEqual(strategy._ddp_kwargs.get("find_unused_parameters"), True)
        self.assertEqual(strategy._ddp_kwargs.get("static_graph"), False)

if __name__ == "__main__":
    unittest.main()
