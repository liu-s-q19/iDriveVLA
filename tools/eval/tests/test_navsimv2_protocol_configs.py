import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: str):
    with open(REPO_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestNavSimV2ProtocolConfigs(unittest.TestCase):
    def test_task_tracks_recogdrive_sft_comparison_plan(self):
        task_path = REPO_ROOT / "task" / "navsimv2_rft_action_block_task.md"
        text = task_path.read_text(encoding="utf-8")

        self.assertIn("ReCogDrive-VLM-2B", text)
        self.assertIn("Qwen2.5-VL-3B", text)
        self.assertIn("ReCogDrive-VLM-8B", text)
        self.assertIn("不进入当前主线", text)

    def test_primary_launchers_default_to_canonical_qwen_entries(self):
        run_sft = (REPO_ROOT / "scripts" / "run_sft.sh").read_text(encoding="utf-8")
        run_rft = (REPO_ROOT / "scripts" / "run_rft.sh").read_text(encoding="utf-8")
        navtest = (REPO_ROOT / "scripts" / "eval" / "run_navtest_epdms_standard_8gpu.sh").read_text(encoding="utf-8")
        navhard = (REPO_ROOT / "scripts" / "eval" / "run_navhard_two_stage_autovla_current_8gpu.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("training/qwen2.5-vl-3B-navsimv2-mix-sft-local8gpu", run_sft)
        self.assertIn("training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4-default-format", run_rft)
        self.assertIn("config/eval/navsimv2_epdms_standard_autovla_qwen_sft8_retry3_epoch4.yaml", navtest)
        self.assertIn("config/eval/navhard_two_stage_autovla_qwen_sft8_retry3_epoch4.yaml", navhard)

    def test_dedicated_qwen_and_recogdrive_launchers_exist(self):
        expected_scripts = [
            "scripts/run_sft_qwen_local8gpu.sh",
            "scripts/run_sft_recogdrive_vlm2b_local8gpu.sh",
            "scripts/eval/run_navtest_epdms_standard_qwen_8gpu.sh",
            "scripts/eval/run_navtest_epdms_standard_recogdrive_vlm2b_8gpu.sh",
        ]

        for rel_path in expected_scripts:
            script_path = REPO_ROOT / rel_path
            self.assertTrue(script_path.exists(), f"{rel_path} should exist")

    def test_primary_navsimv2_standard_eval_entry_uses_canonical_8_pose(self):
        cfg = _load_yaml("config/eval/navsimv2_epdms_standard.yaml")
        self.assertEqual(cfg["model"]["trajectory_sampling"]["num_poses"], 8)
        self.assertEqual(cfg["model"]["trajectory_sampling"]["interval_length"], 0.5)

    def test_primary_navsimv2_sft_config_uses_canonical_8_pose(self):
        cfg = _load_yaml("config/training/qwen2.5-vl-3B-navsimv2-mix-sft.yaml")
        self.assertEqual(cfg["model"]["trajectory"]["num_poses"], 8)
        self.assertEqual(cfg["model"]["trajectory"]["interval_length"], 0.5)
        self.assertEqual(cfg["model"]["trajectory"]["time_horizon"], 4.0)

    def test_rft_default_and_answer_configs_share_canonical_trajectory(self):
        default_cfg = _load_yaml("config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4-default-format.yaml")
        answer_cfg = _load_yaml("config/training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4-answer-format.yaml")

        self.assertEqual(default_cfg["model"]["trajectory"]["num_poses"], 8)
        self.assertEqual(answer_cfg["model"]["trajectory"]["num_poses"], 8)
        self.assertFalse(default_cfg["model"]["action_answer_protocol"]["enabled"])
        self.assertTrue(answer_cfg["model"]["action_answer_protocol"]["enabled"])

    def test_rft_eval_templates_explicitly_enable_lora(self):
        navtest_cfg = _load_yaml("config/eval/navsimv2_epdms_standard_autovla_rft_answer_format.yaml")
        navhard_cfg = _load_yaml("config/eval/navhard_two_stage_autovla_rft_answer_format.yaml")

        self.assertTrue(navtest_cfg["model"]["lora_conf"]["use_lora"])
        self.assertTrue(navhard_cfg["model"]["lora_conf"]["use_lora"])

    def test_navhard_step6000_eval_uses_canonical_8_pose(self):
        cfg = _load_yaml("config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml")

        self.assertEqual(cfg["model"]["trajectory_sampling"]["num_poses"], 8)
        self.assertEqual(cfg["model"]["trajectory_sampling"]["interval_length"], 0.5)

    def test_upstream_autovla_agent_uses_canonical_four_second_sampling(self):
        cfg = _load_yaml("navsim/navsim/planning/script/config/common/agent/autovla_agent.yaml")
        self.assertEqual(cfg["trajectory_sampling"]["time_horizon"], 4)
        self.assertEqual(cfg["trajectory_sampling"]["interval_length"], 0.5)


if __name__ == "__main__":
    unittest.main()
