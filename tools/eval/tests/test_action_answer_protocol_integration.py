import unittest

import torch

from models.utils.action_answer_protocol import ActionAnswerParseResult, summarize_group_outcomes
from tools.analysis.check_sft_answer_protocol import protocol_result_to_row, summarize_protocol_results


class TestActionAnswerProtocolIntegration(unittest.TestCase):
    def test_protocol_result_to_row_maps_parse_metadata(self):
        row = protocol_result_to_row(
            ActionAnswerParseResult(
                is_valid=True,
                invalid_reason="",
                answer_block_count=1,
                answer_block_at_tail=True,
                action_token_ids=[101, 102],
                action_token_count=2,
            )
        )

        self.assertEqual(
            row,
            {
                "has_answer_block": 1,
                "multiple_answer_blocks": 0,
                "answer_block_at_tail": 1,
                "answer_action_tokens_len": 2,
                "protocol_valid": 1,
                "invalid_reason": "",
            },
        )

    def test_summarize_protocol_results_aggregates_rates_and_reasons(self):
        rows = [
            {
                "has_answer_block": 1,
                "multiple_answer_blocks": 0,
                "answer_block_at_tail": 1,
                "answer_action_tokens_len": 8,
                "protocol_valid": 1,
                "invalid_reason": "",
            },
            {
                "has_answer_block": 1,
                "multiple_answer_blocks": 1,
                "answer_block_at_tail": 0,
                "answer_action_tokens_len": 4,
                "protocol_valid": 0,
                "invalid_reason": "multiple_answer_blocks",
            },
            {
                "has_answer_block": 0,
                "multiple_answer_blocks": 0,
                "answer_block_at_tail": 0,
                "answer_action_tokens_len": 0,
                "protocol_valid": 0,
                "invalid_reason": "missing_answer_block",
            },
        ]

        summary = summarize_protocol_results(rows)

        self.assertEqual(summary["num_samples"], 3)
        self.assertAlmostEqual(summary["protocol_valid_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["has_answer_block_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["multiple_answer_blocks_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["answer_block_at_tail_rate"], 1.0 / 3.0)
        self.assertEqual(summary["invalid_reason_counts"]["multiple_answer_blocks"], 1)
        self.assertEqual(summary["invalid_reason_counts"]["missing_answer_block"], 1)

    def test_summarize_group_outcomes_detects_all_invalid_and_zero_std(self):
        summary = summarize_group_outcomes(
            grouped_rewards=torch.tensor([0.0, 0.0, 0.0]),
            grouped_valid_mask=torch.tensor([0.0, 0.0, 0.0]),
            group_std_eps=1e-6,
        )

        self.assertEqual(summary["group_valid_count"], 0.0)
        self.assertEqual(summary["group_all_invalid"], 1.0)
        self.assertEqual(summary["group_all_same_reward"], 1.0)
        self.assertTrue(summary["use_zero_advantage"])


if __name__ == "__main__":
    unittest.main()
