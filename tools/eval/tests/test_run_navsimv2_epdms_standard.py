from pathlib import Path
from types import SimpleNamespace
import os
import logging

import pandas as pd

from tools.eval import run_navsimv2_epdms_standard as mod


class _FakeTrajectory:
    def __init__(self, poses, trajectory_sampling):
        self.poses = poses
        self.trajectory_sampling = trajectory_sampling


class _FakePredictor:
    def predict(self, payload):
        assert payload["token"] in {"tok_a", "tok_b"}
        return payload["predicted_poses"], "cot"


class _FakeSceneLoader:
    def get_scene_from_token(self, token):
        return SimpleNamespace(token=token)


class _FakeMetricCacheLoader:
    def get_from_token(self, token):
        return SimpleNamespace(token=token)


def test_evaluate_autovla_one_stage_tokens_appends_average_row(tmp_path):
    tokens = ["tok_a", "tok_b"]
    payloads = {
        "tok_a": {
            "token": "tok_a",
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
        "tok_b": {
            "token": "tok_b",
            "predicted_poses": [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        },
    }

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        assert sensor_root == Path("/tmp/sensors")
        assert dataset_name == "navsim"
        assert trajectory_num_poses == 2
        return payloads[token]

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        base = 1.0 if metric_cache.token == "tok_a" else 0.5
        return pd.DataFrame([{"score": base, "custom_metric": base * 2.0}])

    df = mod._evaluate_autovla_one_stage_tokens(
        tokens=tokens,
        scene_loader=_FakeSceneLoader(),
        metric_cache_loader=_FakeMetricCacheLoader(),
        predictor=_FakePredictor(),
        pdm_score_fn=pdm_score_fn,
        simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
        scorer=object(),
        traffic_agents_policy=object(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_FakeTrajectory,
        payload_builder=payload_builder,
    )

    assert list(df["token"]) == ["tok_a", "tok_b", "average"]
    assert list(df["valid"]) == [True, True, True]
    assert df.loc[df["token"] == "average", "score"].item() == 0.75
    assert df.loc[df["token"] == "average", "custom_metric"].item() == 1.5


def test_run_evaluation_dispatches_by_mode():
    calls = []

    def fake_upstream(**kwargs):
        calls.append(("upstream", kwargs["dry_run"]))
        return 11

    def fake_autovla(**kwargs):
        calls.append(("autovla", kwargs["dry_run"]))
        return 22

    cfg = {
        "evaluation": {
            "mode": "autovla_one_stage",
        }
    }

    hydra_cfg = SimpleNamespace()

    code = mod._run_evaluation(
        cfg=cfg,
        hydra_cfg=hydra_cfg,
        navsim_root=Path("/tmp/upstream"),
        overrides=["metric_cache_path=/tmp/cache"],
        dry_run=True,
        run_upstream_fn=fake_upstream,
        run_autovla_fn=fake_autovla,
    )

    assert code == 22
    assert calls == [("autovla", True)]


def test_run_autovla_one_stage_from_components_writes_outputs(tmp_path):
    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], None

    class _Trajectory:
        def __init__(self, poses, trajectory_sampling):
            self.poses = poses
            self.trajectory_sampling = trajectory_sampling

    scene_loader = _FakeSceneLoader()
    metric_cache_loader = _FakeMetricCacheLoader()

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] if token == "tok_a" else [[0.0, 1.0, 0.0]],
        }

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        if metric_cache.token == "tok_a":
            return pd.DataFrame([{"score": 1.0, "invalid": 0}])
        return pd.DataFrame([{"score": 0.5, "invalid": 0}])

    run_dir = tmp_path / "run"
    code = mod._run_autovla_one_stage_from_components(
        tokens=["tok_a", "tok_b"],
        output_dir=run_dir,
        scene_loader=scene_loader,
        metric_cache_loader=metric_cache_loader,
        predictor=_Predictor(),
        pdm_score_fn=pdm_score_fn,
        simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
        scorer=object(),
        traffic_agents_policy=object(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_Trajectory,
        payload_builder=payload_builder,
    )

    assert code == 0

    csv_files = sorted(run_dir.glob("*.csv"))
    assert len(csv_files) == 1
    df = pd.read_csv(csv_files[0])
    assert list(df["token"]) == ["tok_a", "tok_b", "average"]
    assert df.loc[df["token"] == "average", "score"].item() == 0.75

    summary_path = run_dir / "summary.json"
    assert summary_path.exists()
    summary = pd.read_json(summary_path, typ="series")
    assert int(summary["successful"]) == 2
    assert int(summary["failed"]) == 0
    assert int(summary["invalid_sum"]) == 0


def test_evaluate_autovla_one_stage_tokens_accepts_tuple_pdm_score():
    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], None

    class _Trajectory:
        def __init__(self, poses, trajectory_sampling):
            self.poses = poses
            self.trajectory_sampling = trajectory_sampling

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        return pd.DataFrame([{"score": 0.25, "invalid": 0}]), ["ego_states"]

    df = mod._evaluate_autovla_one_stage_tokens(
        tokens=["tok_a"],
        scene_loader=_FakeSceneLoader(),
        metric_cache_loader=_FakeMetricCacheLoader(),
        predictor=_Predictor(),
        pdm_score_fn=pdm_score_fn,
        simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
        scorer=object(),
        traffic_agents_policy=object(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_Trajectory,
        payload_builder=payload_builder,
    )

    assert df.loc[df["token"] == "tok_a", "score"].item() == 0.25


def test_apply_process_env_sets_configured_variables(monkeypatch):
    monkeypatch.delenv("NUPLAN_MAPS_ROOT", raising=False)
    monkeypatch.delenv("OPENSCENE_DATA_ROOT", raising=False)

    mod._apply_process_env(
        {
            "run": {
                "env_vars": {
                    "NUPLAN_MAPS_ROOT": "/data/dataset/navsim/maps",
                    "OPENSCENE_DATA_ROOT": "/data/dataset/navsim",
                }
            }
        }
    )

    assert os.environ["NUPLAN_MAPS_ROOT"] == "/data/dataset/navsim/maps"
    assert os.environ["OPENSCENE_DATA_ROOT"] == "/data/dataset/navsim"


def test_evaluate_autovla_one_stage_tokens_logs_periodic_progress(caplog):
    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], None

    class _Trajectory:
        def __init__(self, poses, trajectory_sampling):
            self.poses = poses
            self.trajectory_sampling = trajectory_sampling

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        return pd.DataFrame([{"score": 1.0, "invalid": 0}])

    with caplog.at_level(logging.INFO, logger=mod.LOGGER.name):
        mod._evaluate_autovla_one_stage_tokens(
            tokens=["tok_a", "tok_b", "tok_c", "tok_d", "tok_e"],
            scene_loader=_FakeSceneLoader(),
            metric_cache_loader=_FakeMetricCacheLoader(),
            predictor=_Predictor(),
            pdm_score_fn=pdm_score_fn,
            simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
            scorer=object(),
            traffic_agents_policy=object(),
            sensor_root=Path("/tmp/sensors"),
            dataset_name="navsim",
            trajectory_num_poses=2,
            trajectory_interval=0.5,
            trajectory_cls=_Trajectory,
            payload_builder=payload_builder,
            progress_every=2,
            shard_name="shard_03",
        )

    progress_logs = [record.message for record in caplog.records if "Progress shard_03" in record.message]
    assert len(progress_logs) == 3
    assert "2/5" in progress_logs[0]
    assert "4/5" in progress_logs[1]
    assert "5/5" in progress_logs[2]
    assert "last_token=tok_e" in progress_logs[2]


def test_evaluate_autovla_one_stage_tokens_disables_progress_when_interval_zero(caplog):
    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], None

    class _Trajectory:
        def __init__(self, poses, trajectory_sampling):
            self.poses = poses
            self.trajectory_sampling = trajectory_sampling

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        return pd.DataFrame([{"score": 1.0, "invalid": 0}])

    with caplog.at_level(logging.INFO, logger=mod.LOGGER.name):
        mod._evaluate_autovla_one_stage_tokens(
            tokens=["tok_a", "tok_b"],
            scene_loader=_FakeSceneLoader(),
            metric_cache_loader=_FakeMetricCacheLoader(),
            predictor=_Predictor(),
            pdm_score_fn=pdm_score_fn,
            simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
            scorer=object(),
            traffic_agents_policy=object(),
            sensor_root=Path("/tmp/sensors"),
            dataset_name="navsim",
            trajectory_num_poses=2,
            trajectory_interval=0.5,
            trajectory_cls=_Trajectory,
            payload_builder=payload_builder,
            progress_every=0,
            shard_name="shard_00",
        )

    assert not [record.message for record in caplog.records if "Progress shard_00" in record.message]


def test_run_autovla_one_stage_from_components_uses_pdm_score_when_score_missing(tmp_path):
    class _Predictor:
        def predict(self, payload):
            return payload["predicted_poses"], None

    class _Trajectory:
        def __init__(self, poses, trajectory_sampling):
            self.poses = poses
            self.trajectory_sampling = trajectory_sampling

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        value = 1.0 if metric_cache.token == "tok_a" else 0.5
        return pd.DataFrame([{"pdm_score": value, "invalid": 0}]), ["ego_states"]

    run_dir = tmp_path / "run_pdm_only"
    mod._run_autovla_one_stage_from_components(
        tokens=["tok_a", "tok_b"],
        output_dir=run_dir,
        scene_loader=_FakeSceneLoader(),
        metric_cache_loader=_FakeMetricCacheLoader(),
        predictor=_Predictor(),
        pdm_score_fn=pdm_score_fn,
        simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
        scorer=object(),
        traffic_agents_policy=object(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_Trajectory,
        payload_builder=payload_builder,
    )

    summary = pd.read_json(run_dir / "summary.json", typ="series")
    assert float(summary["score_mean"]) == 0.75
