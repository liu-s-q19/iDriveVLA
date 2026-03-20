from pathlib import Path

import pandas as pd
import pytest

from tools.eval import navhard_two_stage_sharded as mod


def test_intersect_tokens_preserves_lhs_order():
    lhs = ["tok_c", "tok_a", "tok_b", "tok_d"]
    rhs = ["tok_b", "tok_d", "tok_c"]

    assert mod.intersect_tokens(lhs, rhs) == ["tok_c", "tok_b", "tok_d"]


def test_build_eval_jobs_and_round_robin_shards_cover_all_jobs():
    jobs = mod.build_eval_jobs(
        stage1_tokens=["orig_a", "orig_b"],
        stage2_tokens=["syn_a", "syn_b", "syn_c"],
    )

    assert [job["job_id"] for job in jobs] == [
        "stage_one:orig_a",
        "stage_one:orig_b",
        "stage_two:syn_a",
        "stage_two:syn_b",
        "stage_two:syn_c",
    ]

    shards = mod.shard_eval_jobs(jobs, num_shards=3)
    shard_job_ids = [[job["job_id"] for job in shard] for shard in shards]

    assert shard_job_ids == [
        ["stage_one:orig_a", "stage_two:syn_b"],
        ["stage_one:orig_b", "stage_two:syn_c"],
        ["stage_two:syn_a"],
    ]

    flat_job_ids = [job_id for shard in shard_job_ids for job_id in shard]
    assert sorted(flat_job_ids) == sorted(job["job_id"] for job in jobs)


def test_select_stage_two_tokens_preserves_scene_filter_order():
    tokens = mod.select_stage_two_tokens(
        traffic_mode="reactive",
        available_synthetic_tokens=["syn_b", "syn_c", "syn_a"],
        reactive_initial_tokens=["syn_a", "syn_c", "syn_x", "syn_b"],
        non_reactive_initial_tokens=["other"],
        metric_cache_tokens=["syn_c", "syn_b", "syn_a"],
    )

    assert tokens == ["syn_a", "syn_c", "syn_b"]


def test_stable_token_seed_is_deterministic_and_sensitive_to_inputs():
    seed_a_1 = mod.stable_token_seed("tok_a", seed_base=7)
    seed_a_2 = mod.stable_token_seed("tok_a", seed_base=7)
    seed_b = mod.stable_token_seed("tok_b", seed_base=7)
    seed_a_other_base = mod.stable_token_seed("tok_a", seed_base=8)

    assert seed_a_1 == seed_a_2
    assert seed_a_1 != seed_b
    assert seed_a_1 != seed_a_other_base
    assert 0 <= seed_a_1 < 2**31 - 1


def test_validate_job_coverage_rejects_missing_jobs():
    jobs = mod.build_eval_jobs(stage1_tokens=["orig_a"], stage2_tokens=["syn_a"])
    partial_df = pd.DataFrame(
        [
            {"token": "orig_a", "stage_name": "stage_one", "valid": True},
        ]
    )

    with pytest.raises(RuntimeError, match="Missing shard results"):
        mod.validate_job_coverage(expected_jobs=jobs, combined_rows=partial_df)


def test_finalize_merged_results_without_mappings_writes_expected_summary(tmp_path):
    combined_rows = pd.DataFrame(
        [
            {"token": "orig_a", "valid": True, "frame_type": "ORIGINAL", "score": 0.8, "metric_a": 0.8, "stage_name": "stage_one"},
            {"token": "orig_b", "valid": True, "frame_type": "ORIGINAL", "score": 0.6, "metric_a": 0.6, "stage_name": "stage_one"},
            {"token": "syn_a", "valid": True, "frame_type": "SYNTHETIC", "score": 0.4, "metric_a": 0.4, "stage_name": "stage_two"},
            {"token": "syn_b", "valid": False, "frame_type": "SYNTHETIC", "score": 0.2, "metric_a": 0.2, "stage_name": "stage_two"},
        ]
    )

    final_df, summary = mod.finalize_merged_results(
        combined_rows=combined_rows,
        all_mappings={},
        proposal_sampling="unused",
        scene_frame_type_original="ORIGINAL",
        scene_frame_type_synthetic="SYNTHETIC",
        pdm_result_field_names=["metric_a"],
        output_dir=tmp_path,
        write_artifacts=False,
    )

    assert list(final_df["token"][-3:]) == [
        "extended_pdm_score_stage_one",
        "extended_pdm_score_stage_two",
        "extended_pdm_score_combined",
    ]
    assert summary["num_successful_scenarios"] == 3
    assert summary["num_failed_scenarios"] == 1
    assert summary["final_extended_pdm_score"] == pytest.approx((0.8 + 0.6 + 0.4) / 3.0)

    combined_row = final_df[final_df["token"] == "extended_pdm_score_combined"].iloc[0]
    assert combined_row["score"] == pytest.approx((0.8 + 0.6 + 0.4) / 3.0)
    assert pd.isna(combined_row["metric_a_stage_one"])
    assert pd.isna(combined_row["metric_a_stage_two"])


def test_finalize_merged_results_aggregates_protocol_diagnostics(tmp_path):
    combined_rows = pd.DataFrame(
        [
            {
                "token": "orig_a",
                "valid": True,
                "frame_type": "ORIGINAL",
                "score": 0.8,
                "metric_a": 0.8,
                "stage_name": "stage_one",
                "protocol_valid": 1,
                "protocol_reason": "",
                "action_tokens_count": 8,
                "raw_pose_count": 8,
                "was_padded": 0,
                "was_truncated": 0,
                "used_zero_fallback": 0,
            },
            {
                "token": "orig_b",
                "valid": True,
                "frame_type": "ORIGINAL",
                "score": 0.6,
                "metric_a": 0.6,
                "stage_name": "stage_one",
                "protocol_valid": 0,
                "protocol_reason": "action_count_mismatch",
                "action_tokens_count": 4,
                "raw_pose_count": 4,
                "was_padded": 1,
                "was_truncated": 0,
                "used_zero_fallback": 0,
            },
            {
                "token": "syn_a",
                "valid": True,
                "frame_type": "SYNTHETIC",
                "score": 0.4,
                "metric_a": 0.4,
                "stage_name": "stage_two",
                "protocol_valid": 0,
                "protocol_reason": "missing_answer_block",
                "action_tokens_count": 0,
                "raw_pose_count": 0,
                "was_padded": 1,
                "was_truncated": 0,
                "used_zero_fallback": 1,
            },
        ]
    )

    _, summary = mod.finalize_merged_results(
        combined_rows=combined_rows,
        all_mappings={},
        proposal_sampling="unused",
        scene_frame_type_original="ORIGINAL",
        scene_frame_type_synthetic="SYNTHETIC",
        pdm_result_field_names=["metric_a"],
        output_dir=tmp_path,
        write_artifacts=False,
    )

    assert summary["num_protocol_invalid"] == 2
    assert summary["num_padded"] == 2
    assert summary["num_truncated"] == 0
    assert summary["num_zero_fallback"] == 1
    assert summary["action_tokens_count_mean"] == pytest.approx(4.0)
    assert summary["action_tokens_count_min"] == 0
    assert summary["action_tokens_count_max"] == 8
