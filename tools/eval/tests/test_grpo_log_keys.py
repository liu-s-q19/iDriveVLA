import unittest

from models.utils.grpo_log_keys import progress_bar_metric_names


class TestGRPOLogKeys(unittest.TestCase):
    def test_progress_bar_metrics_include_group_std_and_action_len(self):
        keys = progress_bar_metric_names()
        self.assertIn('group_reward_std', keys)
        self.assertIn('sample_action_tokens_len', keys)
        self.assertIn('group_adv_fallback', keys)


if __name__ == '__main__':
    unittest.main()
