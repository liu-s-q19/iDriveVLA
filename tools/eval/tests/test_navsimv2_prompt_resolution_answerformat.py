import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SFT_DATASET_PATH = REPO_ROOT / "dataset_utils" / "sft_dataset.py"
AUTOVLA_PATH = REPO_ROOT / "models" / "autovla.py"


class TestNavSimV2PromptAndResolution(unittest.TestCase):
    def test_sft_dataset_does_not_hardcode_video_pixels(self):
        source = SFT_DATASET_PATH.read_text(encoding="utf-8")
        self.assertNotIn("28 * 28 * 128", source)
        self.assertIn("self.video_min_pixels", source)
        self.assertIn("self.video_max_pixels", source)

    def test_prompt_semantics_use_trajectory_in_sft_and_inference(self):
        sft_source = SFT_DATASET_PATH.read_text(encoding="utf-8")
        autovla_source = AUTOVLA_PATH.read_text(encoding="utf-8")

        self.assertNotIn("The final output action is:", sft_source)
        self.assertIn("The final output trajectory is:", sft_source)

        self.assertNotIn("predict the most appropriate driving action", sft_source)
        self.assertNotIn("predict the optimal driving action", sft_source)
        self.assertIn("predict the future trajectory", sft_source)

        self.assertNotIn("predict the most appropriate driving action", autovla_source)
        self.assertNotIn("predict the optimal driving action", autovla_source)
        self.assertIn("predict the future trajectory", autovla_source)


if __name__ == "__main__":
    unittest.main()
