import unittest

import torch

from models.utils.action_answer_protocol import extract_action_tokens_from_first_answer_block
from models.utils.action_answer_protocol import parse_action_answer_completion


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 999
        self.pad_token_id = 0
        self.all_special_tokens = ["<|im_end|>", "<|endoftext|>", "<|custom_special|>"]
        self.id_to_text = {
            0: "",
            1: "<answer>",
            2: "</answer>",
            3: "The final output action is: ",
            4: " ",
            5: "\n",
            6: "tail text",
            7: "The final output trajectory is: ",
            8: "<|im_end|>",
            9: "<|endoftext|>",
            10: "<|custom_special|>",
            101: "<action_0>",
            102: "<action_1>",
            103: "<action_2>",
            104: "<action_3>",
            105: "<action_4>",
            106: "<action_5>",
            107: "<action_6>",
            108: "<action_7>",
            109: "<action_8>",
        }

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.id_to_text[int(token_id)] for token_id in token_ids)


class TestActionAnswerProtocol(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.expected_action_len = 8
        self.action_start_id = 100

    def test_parse_single_tail_answer_block_extracts_exact_action_tokens(self):
        completion_ids = torch.tensor([1, 3, 101, 102, 103, 104, 105, 106, 107, 108, 2, 5, 999, 0])

        result = parse_action_answer_completion(
            completion_ids,
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.invalid_reason, "")
        self.assertEqual(result.answer_block_count, 1)
        self.assertTrue(result.answer_block_at_tail)
        self.assertEqual(result.action_token_ids, [101, 102, 103, 104, 105, 106, 107, 108])
        self.assertEqual(result.action_token_count, 8)

    def test_parse_does_not_depend_on_action_keyword(self):
        completion_ids = torch.tensor([1, 7, 101, 102, 103, 104, 105, 106, 107, 108, 2])

        result = parse_action_answer_completion(
            completion_ids,
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.action_token_ids, [101, 102, 103, 104, 105, 106, 107, 108])

    def test_parse_rejects_missing_answer_block(self):
        result = parse_action_answer_completion(
            torch.tensor([3, 101, 102, 103]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "missing_answer_block")
        self.assertEqual(result.answer_block_count, 0)

    def test_parse_rejects_multiple_answer_blocks(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 101, 2, 1, 102, 2]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "multiple_answer_blocks")
        self.assertEqual(result.answer_block_count, 2)

    def test_parse_rejects_tail_text_after_answer_block(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 3, 101, 102, 103, 104, 105, 106, 107, 108, 2, 6]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "answer_block_not_at_tail")
        self.assertFalse(result.answer_block_at_tail)

    def test_parse_allows_special_end_tokens_after_answer_block(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 3, 101, 102, 103, 104, 105, 106, 107, 108, 2, 8, 5, 9]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.invalid_reason, "")
        self.assertTrue(result.answer_block_at_tail)
        self.assertEqual(result.action_token_count, 8)

    def test_parse_allows_any_tokenizer_special_tokens_after_answer_block(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 7, 101, 102, 103, 104, 105, 106, 107, 108, 2, 10, 5, 8, 9]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.invalid_reason, "")
        self.assertTrue(result.answer_block_at_tail)
        self.assertEqual(result.action_token_count, 8)

    def test_parse_rejects_special_tokens_followed_by_plain_tail_text(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 7, 101, 102, 103, 104, 105, 106, 107, 108, 2, 10, 5, 6]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "answer_block_not_at_tail")
        self.assertFalse(result.answer_block_at_tail)

    def test_parse_rejects_short_action_block(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 3, 101, 102, 103, 2]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "action_count_mismatch")
        self.assertEqual(result.action_token_count, 3)

    def test_parse_rejects_long_action_block(self):
        result = parse_action_answer_completion(
            torch.tensor([1, 3, 101, 102, 103, 104, 105, 106, 107, 108, 109, 2]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
            expected_action_len=self.expected_action_len,
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "action_count_mismatch")
        self.assertEqual(result.action_token_count, 9)

    def test_extract_first_answer_block_tokens_ignores_tail_text(self):
        action_ids = extract_action_tokens_from_first_answer_block(
            torch.tensor([1, 3, 101, 102, 103, 104, 105, 106, 107, 108, 2, 6, 101, 102]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
        )

        self.assertEqual(action_ids, [101, 102, 103, 104, 105, 106, 107, 108])

    def test_extract_first_answer_block_tokens_returns_partial_block_without_tail_tokens(self):
        action_ids = extract_action_tokens_from_first_answer_block(
            torch.tensor([1, 3, 101, 102, 103, 2, 6, 107, 108]),
            tokenizer=self.tokenizer,
            action_start_id=self.action_start_id,
        )

        self.assertEqual(action_ids, [101, 102, 103])


if __name__ == "__main__":
    unittest.main()
