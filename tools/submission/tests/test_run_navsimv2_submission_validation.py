from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tools.submission import run_navsimv2_submission_validation as mod


class _FakeTrajectory:
    def __init__(self, num_poses: int, interval_length: float = 0.5):
        self.poses = np.zeros((num_poses, 3), dtype=np.float32)
        self.trajectory_sampling = TrajectorySampling(num_poses=num_poses, interval_length=interval_length)


def test_load_prediction_dicts_requires_single_seed_lists():
    payload = {
        "first_stage_predictions": [{"tok_a": "traj_a"}],
        "second_stage_predictions": [{"tok_b": "traj_b"}],
    }

    first_stage, second_stage = mod._load_prediction_dicts(payload)

    assert first_stage == {"tok_a": "traj_a"}
    assert second_stage == {"tok_b": "traj_b"}


def test_validate_submission_schema_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="Missing required submission fields"):
        mod._validate_submission_schema({"team_name": "team"})


def test_validate_private_submission_checks_filename_and_token_coverage():
    payload = {
        "team_name": "team",
        "authors": "authors",
        "email": "email@example.com",
        "institution": "org",
        "country / region": "Country",
        "first_stage_predictions": [{"tok_a": _FakeTrajectory(8)}],
        "second_stage_predictions": [{"tok_b": _FakeTrajectory(8)}],
    }

    summary = mod._validate_private_submission(
        submission_data=payload,
        submission_path=Path("/tmp/submission.pkl"),
        expected_first_stage_tokens=["tok_a"],
        expected_second_stage_tokens=["tok_b"],
        expected_num_poses=8,
        expected_interval_length=0.5,
        split_name="private_test_hard_two_stage",
    )

    assert summary["valid"] is True
    assert summary["missing_first_stage_tokens"] == []
    assert summary["missing_second_stage_tokens"] == []


def test_validate_private_submission_rejects_duplicate_token_across_stages():
    payload = {
        "team_name": "team",
        "authors": "authors",
        "email": "email@example.com",
        "institution": "org",
        "country / region": "Country",
        "first_stage_predictions": [{"tok_a": _FakeTrajectory(8)}],
        "second_stage_predictions": [{"tok_a": _FakeTrajectory(8)}],
    }

    summary = mod._validate_private_submission(
        submission_data=payload,
        submission_path=Path("/tmp/submission.pkl"),
        expected_first_stage_tokens=["tok_a"],
        expected_second_stage_tokens=["tok_b"],
        expected_num_poses=8,
        expected_interval_length=0.5,
        split_name="private_test_hard_two_stage",
    )

    assert summary["valid"] is False
    assert summary["duplicate_tokens_across_stages"] == ["tok_a"]
    assert summary["missing_second_stage_tokens"] == ["tok_b"]


def test_validate_private_submission_rejects_sampling_mismatch():
    payload = {
        "team_name": "team",
        "authors": "authors",
        "email": "email@example.com",
        "institution": "org",
        "country / region": "Country",
        "first_stage_predictions": [{}],
        "second_stage_predictions": [{"tok_b": _FakeTrajectory(10)}],
    }

    summary = mod._validate_private_submission(
        submission_data=payload,
        submission_path=Path("/tmp/submission.pkl"),
        expected_first_stage_tokens=[],
        expected_second_stage_tokens=["tok_b"],
        expected_num_poses=8,
        expected_interval_length=0.5,
        split_name="private_test_hard_two_stage",
    )

    assert summary["valid"] is False
    assert summary["sampling_mismatch_second_stage_tokens"] == ["tok_b"]


def test_build_warmup_summary_uses_extended_pdm_score_row():
    df = pd.DataFrame(
        [
            {"token": "tok_a", "valid": True, "score": 0.5},
            {"token": "extended_pdm_score_stage_one", "valid": True, "score": 0.6},
            {"token": "extended_pdm_score_stage_two", "valid": True, "score": 0.7},
            {"token": "extended_pdm_score_combined", "valid": True, "score": 0.75},
        ]
    )

    summary = mod._build_warmup_summary(
        score_df=df,
        csv_path=Path("/tmp/result.csv"),
        split_name="warmup_two_stage",
    )

    assert summary["valid"] is True
    assert summary["successful"] == 1
    assert summary["failed"] == 0
    assert summary["final_extended_pdm_score"] == 0.75


def test_build_warmup_validation_command_prefixes_upstream_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing")
    cfg = mod.OmegaConf.create(
        {
            "upstream": {"navsim_root": "/data/liushiqi/navsim"},
            "data": {
                "train_test_split": "warmup_two_stage",
                "metric_cache_path": "/tmp/cache",
                "navsim_log_path": "/tmp/logs",
                "original_sensor_path": "/tmp/orig",
                "synthetic_sensor_path": "/tmp/synth",
                "synthetic_scenes_path": "/tmp/scenes",
            },
        }
    )

    cmd, env = mod._build_warmup_validation_command(
        cfg=cfg,
        submission_path=Path("/tmp/submission.pkl"),
        output_dir=Path("/tmp/out"),
    )

    assert cmd[1].endswith("run_pdm_score_from_submission.py")
    assert env["PYTHONPATH"].startswith("/data/liushiqi/navsim:")
    assert "metric_cache_path=/tmp/cache" in cmd


def test_collect_expected_tokens_from_mapping_keeps_order():
    first_stage, second_stage = mod._collect_tokens_from_mapping(
        [
            ("orig_a", "prev_a", [("syn_1", "syn_2")]),
            ("orig_b", "prev_a", [("syn_2", "syn_3")]),
        ]
    )

    assert first_stage == ["orig_a", "prev_a", "orig_b"]
    assert second_stage == ["syn_1", "syn_2", "syn_3"]
