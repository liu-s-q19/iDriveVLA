from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.submission import run_navsimv2_autovla_submission as mod


class _FakeTrajectory:
    def __init__(self, poses, trajectory_sampling):
        self.poses = poses
        self.trajectory_sampling = trajectory_sampling


def test_scene_loader_module_name_matches_split():
    assert mod._scene_loader_module_name("warmup_two_stage") == "navsim.common.dataloader"
    assert mod._scene_loader_module_name("navhard_two_stage") == "navsim.common.dataloader"
    assert mod._scene_loader_module_name("private_test_hard_two_stage") == "navsim.common.dataloader_private"


def test_build_submission_payload_uses_official_schema():
    payload = mod._build_submission_payload(
        metadata={
            "team_name": "team",
            "authors": "authors",
            "email": "email@example.com",
            "institution": "org",
            "country / region": "Country",
        },
        first_stage_predictions={"tok_a": "traj_a"},
        second_stage_predictions={"tok_b": "traj_b"},
    )

    assert payload == {
        "team_name": "team",
        "authors": "authors",
        "email": "email@example.com",
        "institution": "org",
        "country / region": "Country",
        "first_stage_predictions": [{"tok_a": "traj_a"}],
        "second_stage_predictions": [{"tok_b": "traj_b"}],
    }


def test_build_run_summary_counts_stage_tokens():
    summary = mod._build_run_summary(
        split_name="warmup_two_stage",
        loader_module_name="navsim.common.dataloader",
        output_dir=Path("/tmp/out"),
        first_stage_predictions={"tok_a": "traj_a", "tok_b": "traj_b"},
        second_stage_predictions={"tok_c": "traj_c"},
        dry_run=False,
    )

    assert summary["split_name"] == "warmup_two_stage"
    assert summary["loader_module_name"] == "navsim.common.dataloader"
    assert summary["first_stage_tokens"] == 2
    assert summary["second_stage_tokens"] == 1
    assert summary["submission_file"] == "/tmp/out/submission.pkl"


def test_collect_tokens_from_mapping_deduplicates_stage_tokens():
    first_stage, second_stage = mod._collect_tokens_from_mapping(
        [
            ("orig_a", "prev_a", [("syn_1", "syn_2"), ("syn_3", "syn_4")]),
            ("orig_b", "prev_a", [("syn_2", "syn_5")]),
        ]
    )

    assert first_stage == ["orig_a", "prev_a", "orig_b"]
    assert second_stage == ["syn_1", "syn_2", "syn_3", "syn_4", "syn_5"]


def test_collect_split_tokens_skips_private_stage_one_tokens():
    upstream_cfg = SimpleNamespace(
        train_test_split=SimpleNamespace(
            data_split="private_test_hard",
            scene_filter={
                "tokens": ["orig_a", "orig_b"],
                "reactive_synthetic_initial_tokens": ["syn_a", "syn_b"],
            },
            get=lambda key, default=None: None,
        )
    )

    first_stage, second_stage = mod._collect_split_tokens(upstream_cfg)

    assert first_stage == []
    assert second_stage == ["syn_a", "syn_b"]


def test_normalize_tokens_sorts_for_deterministic_sharding():
    assert mod._normalize_tokens(["tok_b", "tok_a", "tok_c"]) == ["tok_a", "tok_b", "tok_c"]


def test_predict_for_tokens_raises_when_any_token_fails():
    class _Predictor:
        def predict(self, payload):
            if payload["token"] == "tok_b":
                raise RuntimeError("boom")
            return payload["predicted_poses"], None

    class _SceneLoader:
        def get_scene_from_token(self, token):
            return object()

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    with pytest.raises(RuntimeError, match="Failed to generate trajectories for tokens: tok_b"):
        mod._predict_for_tokens(
            tokens=["tok_a", "tok_b"],
            scene_loader=_SceneLoader(),
            predictor=_Predictor(),
            sensor_root=Path("/tmp/sensors"),
            dataset_name="navsim",
            split_name="warmup_two_stage",
            trajectory_num_poses=2,
            trajectory_interval=0.5,
            trajectory_cls=_FakeTrajectory,
            payload_builder=payload_builder,
        )


def test_predict_for_tokens_accepts_predictor_returning_sampling_and_cot():
    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], "ignored_sampling", "cot"

    class _SceneLoader:
        def get_scene_from_token(self, token):
            return object()

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    preds = mod._predict_for_tokens(
        tokens=["tok_a"],
        scene_loader=_SceneLoader(),
        predictor=_Predictor(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        split_name="warmup_two_stage",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_FakeTrajectory,
        payload_builder=payload_builder,
    )

    assert list(preds) == ["tok_a"]
    assert preds["tok_a"].poses == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_predict_for_tokens_uses_agent_input_loader_for_private_split():
    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], "ignored_sampling", "cot"

    class _SceneLoader:
        def get_scene_from_token(self, token):
            raise AssertionError("private split should not call get_scene_from_token")

        def get_agent_input_from_token(self, token):
            return {"token": token}

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        assert scene.get_agent_input() == {"token": token}
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    preds = mod._predict_for_tokens(
        tokens=["tok_private"],
        scene_loader=_SceneLoader(),
        predictor=_Predictor(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        split_name="private_test_hard_two_stage",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_FakeTrajectory,
        payload_builder=payload_builder,
    )

    assert list(preds) == ["tok_private"]


def test_write_submission_artifacts_persists_summary_and_pickle(tmp_path):
    summary = mod._write_submission_artifacts(
        output_dir=tmp_path,
        resolved_config_text="config: value\n",
        submission_payload=mod._build_submission_payload(
            metadata={
                "team_name": "team",
                "authors": "authors",
                "email": "email@example.com",
                "institution": "org",
                "country / region": "Country",
            },
            first_stage_predictions={"tok_a": "traj_a"},
            second_stage_predictions={"tok_b": "traj_b"},
        ),
        summary=mod._build_run_summary(
            split_name="warmup_two_stage",
            loader_module_name="navsim.common.dataloader",
            output_dir=tmp_path,
            first_stage_predictions={"tok_a": "traj_a"},
            second_stage_predictions={"tok_b": "traj_b"},
            dry_run=False,
        ),
    )

    assert (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8") == "config: value\n"
    assert (tmp_path / "submission.pkl").exists()
    written_summary = pd.read_json(tmp_path / "summary.json", typ="series")
    assert written_summary["submission_file"] == str(tmp_path / "submission.pkl")
    assert summary["summary_file"] == str(tmp_path / "summary.json")


def test_slice_tokens_for_shard_distributes_without_overlap():
    tokens = [f"tok_{i}" for i in range(10)]

    shards = [mod._slice_tokens_for_shard(tokens, shard_index=i, num_shards=4) for i in range(4)]

    assert shards == [
        ["tok_0", "tok_4", "tok_8"],
        ["tok_1", "tok_5", "tok_9"],
        ["tok_2", "tok_6"],
        ["tok_3", "tok_7"],
    ]
