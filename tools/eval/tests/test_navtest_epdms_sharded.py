from pathlib import Path

import pandas as pd

from tools.eval import navtest_epdms_sharded as mod


def test_intersect_tokens_preserves_scene_order_and_deduplicates():
    scene_tokens = ["tok_c", "tok_a", "tok_b", "tok_a", "tok_d"]
    cache_tokens = ["tok_b", "tok_d", "tok_c"]

    assert mod.intersect_tokens(scene_tokens, cache_tokens) == ["tok_c", "tok_b", "tok_d"]


def test_shard_tokens_round_robin_covers_all_tokens():
    shards = mod.shard_tokens(["tok0", "tok1", "tok2", "tok3", "tok4"], num_shards=3)

    assert shards == [
        ["tok0", "tok3"],
        ["tok1", "tok4"],
        ["tok2"],
    ]


def test_summarize_merged_results_aggregates_csv_and_summary(tmp_path):
    shard0 = tmp_path / "shard_00"
    shard1 = tmp_path / "shard_01"
    shard0.mkdir()
    shard1.mkdir()

    (shard0 / "summary.json").write_text(
        '{"successful": 2, "failed": 0, "invalid_sum": 1, "score_mean": 0.8}',
        encoding="utf-8",
    )
    (shard1 / "summary.json").write_text(
        '{"successful": 1, "failed": 1, "invalid_sum": 0, "score_mean": 0.4}',
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"token": "tok_a", "score": 1.0, "valid": True},
            {"token": "tok_b", "score": 0.6, "valid": True},
        ]
    ).to_csv(shard0 / "rows.csv", index=False)
    pd.DataFrame(
        [
            {"token": "tok_c", "score": 0.4, "valid": True},
            {"token": "tok_d", "score": 0.0, "valid": False},
        ]
    ).to_csv(shard1 / "rows.csv", index=False)

    merged_df, summary = mod.summarize_merged_results([shard0, shard1])

    assert list(merged_df["token"]) == ["tok_a", "tok_b", "tok_c", "tok_d"]
    assert summary["successful"] == 3
    assert summary["failed"] == 1
    assert summary["invalid_sum"] == 1
    assert summary["score_mean"] == 0.6666666666666666
