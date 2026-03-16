import unittest

from models.utils.trainer_progress import build_tqdm_progress_bar


class TestTrainerProgress(unittest.TestCase):
    def test_defaults_to_refresh_every_10_steps(self):
        callback = build_tqdm_progress_bar({})
        self.assertIsNotNone(callback)
        self.assertEqual(callback.refresh_rate, 10)

    def test_allows_override_from_config(self):
        callback = build_tqdm_progress_bar({"progress_bar_refresh_rate": 25})
        self.assertEqual(callback.refresh_rate, 25)

    def test_returns_none_when_progress_bar_disabled(self):
        callback = build_tqdm_progress_bar({"enable_progress_bar": False})
        self.assertIsNone(callback)


if __name__ == "__main__":
    unittest.main()
