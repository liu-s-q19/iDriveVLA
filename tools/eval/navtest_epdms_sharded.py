from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


def safe_token_list(tokens: Optional[Sequence[str]]) -> List[str]:
    return [] if tokens is None else [str(token) for token in tokens]


def intersect_tokens(lhs: Optional[Sequence[str]], rhs: Optional[Sequence[str]]) -> List[str]:
    rhs_set = set(safe_token_list(rhs))
    seen = set()
    ordered: List[str] = []
    for token in safe_token_list(lhs):
        if token in rhs_set and token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def shard_tokens(tokens: Sequence[str], num_shards: int) -> List[List[str]]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    for idx, token in enumerate(safe_token_list(tokens)):
        shards[idx % num_shards].append(token)
    return shards


def summarize_merged_results(shard_dirs: Sequence[Path]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    successful = 0
    failed = 0
    invalid_sum = 0
    weighted_score_sum = 0.0
    weighted_score_count = 0

    for shard_dir in shard_dirs:
        summary_path = Path(shard_dir) / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            shard_successful = int(summary.get("successful", 0))
            successful += shard_successful
            failed += int(summary.get("failed", 0))
            invalid_sum += int(summary.get("invalid_sum", 0))
            weighted_score_sum += float(summary.get("score_mean", 0.0)) * shard_successful
            weighted_score_count += shard_successful

        for csv_path in sorted(Path(shard_dir).glob("*.csv")):
            rows.append(pd.read_csv(csv_path))

    merged_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    score_mean = weighted_score_sum / weighted_score_count if weighted_score_count else 0.0
    merged_summary = {
        "successful": successful,
        "failed": failed,
        "invalid_sum": invalid_sum,
        "score_mean": score_mean,
    }
    return merged_df, merged_summary
