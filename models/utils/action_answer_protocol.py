from dataclasses import dataclass
import re
from typing import Dict, List

import torch


@dataclass
class ActionAnswerParseResult:
    is_valid: bool
    invalid_reason: str
    answer_block_count: int
    answer_block_at_tail: bool
    action_token_ids: List[int]
    action_token_count: int


_ACTION_TOKEN_RE = re.compile(r"<action_(\d+)>")


def _normalize_completion_ids(completion_ids: torch.Tensor, tokenizer) -> List[int]:
    token_ids = [int(token_id) for token_id in completion_ids.tolist()]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    while token_ids and token_ids[-1] in {eos_token_id, pad_token_id}:
        token_ids.pop()
    return token_ids


def extract_action_tokens_from_first_answer_block(
    completion_ids: torch.Tensor,
    tokenizer,
    action_start_id: int,
    answer_open: str = "<answer>",
    answer_close: str = "</answer>",
) -> List[int]:
    token_ids = _normalize_completion_ids(completion_ids, tokenizer)
    if not token_ids:
        return []

    token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]
    completion_text = "".join(token_texts)
    start_char = completion_text.find(answer_open)
    if start_char < 0:
        return []
    end_char = completion_text.find(answer_close, start_char + len(answer_open))
    if end_char < 0:
        return []

    answer_token_ids: List[int] = []
    answer_content_start = start_char + len(answer_open)
    answer_content_end = end_char
    cursor = 0
    for token_id, token_text in zip(token_ids, token_texts):
        next_cursor = cursor + len(token_text)
        overlaps_answer_content = (cursor < answer_content_end) and (next_cursor > answer_content_start)
        if overlaps_answer_content and token_id >= action_start_id and _ACTION_TOKEN_RE.fullmatch(token_text.strip()):
            answer_token_ids.append(token_id)
        cursor = next_cursor

    return answer_token_ids


def parse_action_answer_completion(
    completion_ids: torch.Tensor,
    tokenizer,
    action_start_id: int,
    expected_action_len: int,
    answer_open: str = "<answer>",
    answer_close: str = "</answer>",
) -> ActionAnswerParseResult:
    token_ids = _normalize_completion_ids(completion_ids, tokenizer)
    if not token_ids:
        return ActionAnswerParseResult(
            is_valid=False,
            invalid_reason="missing_answer_block",
            answer_block_count=0,
            answer_block_at_tail=False,
            action_token_ids=[],
            action_token_count=0,
        )

    token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]
    completion_text = "".join(token_texts)

    answer_block_count = completion_text.count(answer_open)
    if answer_block_count == 0:
        return ActionAnswerParseResult(
            is_valid=False,
            invalid_reason="missing_answer_block",
            answer_block_count=0,
            answer_block_at_tail=False,
            action_token_ids=[],
            action_token_count=0,
        )
    if answer_block_count > 1:
        return ActionAnswerParseResult(
            is_valid=False,
            invalid_reason="multiple_answer_blocks",
            answer_block_count=answer_block_count,
            answer_block_at_tail=False,
            action_token_ids=[],
            action_token_count=0,
        )

    start_char = completion_text.find(answer_open)
    end_char = completion_text.find(answer_close, start_char + len(answer_open))
    if end_char < 0:
        return ActionAnswerParseResult(
            is_valid=False,
            invalid_reason="missing_answer_block",
            answer_block_count=1,
            answer_block_at_tail=False,
            action_token_ids=[],
            action_token_count=0,
        )

    close_end_char = end_char + len(answer_close)
    answer_block_at_tail = completion_text[close_end_char:].strip() == ""
    if not answer_block_at_tail:
        return ActionAnswerParseResult(
            is_valid=False,
            invalid_reason="answer_block_not_at_tail",
            answer_block_count=1,
            answer_block_at_tail=False,
            action_token_ids=[],
            action_token_count=0,
        )

    answer_token_ids: List[int] = []
    cursor = 0
    inside_answer = False
    for token_id, token_text in zip(token_ids, token_texts):
        next_cursor = cursor + len(token_text)
        if not inside_answer and start_char < next_cursor:
            inside_answer = True

        if inside_answer and token_id >= action_start_id and _ACTION_TOKEN_RE.fullmatch(token_text.strip()):
            answer_token_ids.append(token_id)

        if inside_answer and close_end_char <= next_cursor:
            inside_answer = False
        cursor = next_cursor

    action_token_count = len(answer_token_ids)
    if action_token_count != expected_action_len:
        return ActionAnswerParseResult(
            is_valid=False,
            invalid_reason="action_count_mismatch",
            answer_block_count=1,
            answer_block_at_tail=True,
            action_token_ids=answer_token_ids,
            action_token_count=action_token_count,
        )

    return ActionAnswerParseResult(
        is_valid=True,
        invalid_reason="",
        answer_block_count=1,
        answer_block_at_tail=True,
        action_token_ids=answer_token_ids,
        action_token_count=action_token_count,
    )


def summarize_group_outcomes(
    grouped_rewards: torch.Tensor,
    grouped_valid_mask: torch.Tensor,
    group_std_eps: float,
) -> Dict[str, float]:
    rewards = grouped_rewards.detach().flatten().float()
    valid_mask = grouped_valid_mask.detach().flatten().float()

    group_valid_count = float(valid_mask.sum().item())
    group_all_invalid = 1.0 if group_valid_count <= 0.0 else 0.0
    group_all_same_reward = 1.0 if rewards.numel() == 0 or torch.allclose(rewards, rewards[:1]) else 0.0
    group_std = rewards.std(unbiased=False) if rewards.numel() > 0 else torch.tensor(0.0)
    use_zero_advantage = bool(group_std.item() < float(group_std_eps))

    return {
        "group_valid_count": group_valid_count,
        "group_all_invalid": group_all_invalid,
        "group_all_same_reward": group_all_same_reward,
        "use_zero_advantage": use_zero_advantage,
    }
