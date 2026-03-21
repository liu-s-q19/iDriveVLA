from pathlib import Path
from types import SimpleNamespace
import os
import logging
import sys
import yaml

import pandas as pd
import pytest
from omegaconf import OmegaConf

from tools.eval import run_navsimv2_epdms_standard as mod
from tools.eval import run_navhard_two_stage_autovla as navhard_mod


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


def test_ensure_repo_root_import_path_prepends_repo_root(monkeypatch):
    original = ["/tmp/other", str(Path(mod.__file__).resolve().parents[2]), "/tmp/third"]
    monkeypatch.setattr(sys, "path", list(original))

    mod._ensure_repo_root_import_path()

    repo_root = str(Path(mod.__file__).resolve().parents[2])
    assert sys.path[0] == repo_root
    assert sys.path.count(repo_root) == 1


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


def test_run_autovla_one_stage_from_components_handles_empty_tokens(tmp_path):
    class _Predictor:
        def predict(self, payload):
            raise AssertionError("predict should not be called for empty token lists")

    run_dir = tmp_path / "empty-run"
    code = mod._run_autovla_one_stage_from_components(
        tokens=[],
        output_dir=run_dir,
        scene_loader=_FakeSceneLoader(),
        metric_cache_loader=_FakeMetricCacheLoader(),
        predictor=_Predictor(),
        pdm_score_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pdm_score_fn should not run")),
        simulator=SimpleNamespace(proposal_sampling="proposal_sampling"),
        scorer=object(),
        traffic_agents_policy=object(),
        sensor_root=Path("/tmp/sensors"),
        dataset_name="navsim",
        trajectory_num_poses=2,
        trajectory_interval=0.5,
        trajectory_cls=_FakeTrajectory,
        payload_builder=lambda **kwargs: (_ for _ in ()).throw(AssertionError("payload_builder should not run")),
    )

    assert code == 0
    assert sorted(run_dir.glob("*.csv")) == []

    summary = pd.read_json(run_dir / "summary.json", typ="series")
    assert int(summary["successful"]) == 0
    assert int(summary["failed"]) == 0
    assert int(summary["invalid_sum"]) == 0
    assert float(summary["score_mean"]) == 0.0


def test_append_average_row_handles_empty_dataframe():
    df = mod._append_average_row(pd.DataFrame())

    assert df.empty
    assert list(df.columns) == []


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


def test_evaluate_autovla_one_stage_tokens_includes_protocol_metadata():
    predictor = _FakePredictor()
    predictor.last_protocol_result = {
        "protocol_valid": 0,
        "invalid_reason": "action_count_mismatch",
        "answer_action_tokens_len": 4,
    }

    def payload_builder(token, scene, sensor_root, dataset_name, trajectory_num_poses):
        return {
            "token": token,
            "predicted_poses": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }

    def pdm_score_fn(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
        return pd.DataFrame([{"score": 0.25, "invalid": 0}])

    df = mod._evaluate_autovla_one_stage_tokens(
        tokens=["tok_a"],
        scene_loader=_FakeSceneLoader(),
        metric_cache_loader=_FakeMetricCacheLoader(),
        predictor=predictor,
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

    row = df.iloc[0].to_dict()
    assert row["protocol_valid"] == 0
    assert row["invalid_reason"] == "action_count_mismatch"
    assert row["answer_action_tokens_len"] == 4


def test_navhard_prediction_diagnostics_capture_protocol_and_padding_flags():
    diagnostics = navhard_mod._build_prediction_diagnostics(
        protocol_result={
            "protocol_valid": 0,
            "invalid_reason": "action_count_mismatch",
            "answer_action_tokens_len": 4,
        },
        raw_pose_count=4,
        target_num_poses=8,
    )

    assert diagnostics == {
        "protocol_valid": 0,
        "protocol_reason": "action_count_mismatch",
        "action_tokens_count": 4,
        "protocol_action_tokens_count": 4,
        "generated_action_tokens_count": 0,
        "raw_pose_count": 4,
        "was_padded": 1,
        "was_truncated": 0,
        "used_zero_fallback": 0,
    }


def test_navhard_prediction_diagnostics_marks_zero_fallback():
    diagnostics = navhard_mod._build_prediction_diagnostics(
        protocol_result={
            "protocol_valid": 0,
            "invalid_reason": "missing_answer_block",
            "answer_action_tokens_len": 0,
        },
        raw_pose_count=0,
        target_num_poses=8,
    )

    assert diagnostics["protocol_valid"] == 0
    assert diagnostics["protocol_reason"] == "missing_answer_block"
    assert diagnostics["action_tokens_count"] == 0
    assert diagnostics["protocol_action_tokens_count"] == 0
    assert diagnostics["generated_action_tokens_count"] == 0
    assert diagnostics["raw_pose_count"] == 0
    assert diagnostics["was_padded"] == 1
    assert diagnostics["was_truncated"] == 0
    assert diagnostics["used_zero_fallback"] == 1


def test_navhard_prediction_diagnostics_falls_back_to_raw_action_tokens_when_protocol_disabled():
    diagnostics = navhard_mod._build_prediction_diagnostics(
        protocol_result={
            "protocol_valid": 0,
            "invalid_reason": "not_run",
            "answer_action_tokens_len": 0,
        },
        raw_pose_count=6,
        target_num_poses=8,
        raw_action_tokens_count=6,
    )

    assert diagnostics == {
        "protocol_valid": 0,
        "protocol_reason": "not_run",
        "action_tokens_count": 6,
        "protocol_action_tokens_count": 0,
        "generated_action_tokens_count": 6,
        "raw_pose_count": 6,
        "was_padded": 1,
        "was_truncated": 0,
        "used_zero_fallback": 0,
    }


def test_autovla_predictor_auto_enables_lora_for_rft_checkpoint(tmp_path, monkeypatch):
    config_path = tmp_path / "autovla.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"dummy": True}}), encoding="utf-8")

    class _FakeModel:
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.vlm = object()
            self.load_calls = []

        def state_dict(self):
            return {
                "vlm.base_model.model.layers.0.self_attn.q_proj.base_layer.weight": 1,
                "vlm.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": 1,
                "vlm.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": 1,
            }

        def load_state_dict(self, state_dict, strict=False):
            self.load_calls.append((state_dict, strict))
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def eval(self):
            return self

    fake_model = _FakeModel()
    get_peft_calls = []

    def fake_autovla(*args, **kwargs):
        return fake_model

    def fake_get_peft_model(vlm, lora_config):
        get_peft_calls.append(lora_config)
        return "wrapped_vlm"

    monkeypatch.setattr(navhard_mod, "AutoVLA", fake_autovla)
    monkeypatch.setattr(navhard_mod, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(
        navhard_mod.torch,
        "load",
        lambda *args, **kwargs: {
            "state_dict": {
                "autovla.vlm.base_model.model.layers.0.self_attn.q_proj.base_layer.weight": 1,
                "autovla.vlm.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": 2,
                "autovla.vlm.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": 3,
            }
        },
    )

    model_cfg = OmegaConf.create(
        {
            "config_path": str(config_path),
            "checkpoint_path": "/tmp/fake-rft.ckpt",
            "sensor_data_path": "/",
            "dataset_name": "navsim",
            "device": "cpu",
            "prediction_seed_mode": "none",
            "prediction_seed_base": 0,
            "lora_conf": {
                "use_lora": None,
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "r": 8,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "bias": "none",
            },
        }
    )
    trajectory_cfg = OmegaConf.create({"num_poses": 10, "interval_length": 0.5})

    navhard_mod.AutoVLAPredictor(model_cfg, trajectory_cfg)

    assert len(get_peft_calls) == 1
    assert fake_model.vlm == "wrapped_vlm"


def test_autovla_predictor_fails_on_checkpoint_mismatches(tmp_path, monkeypatch):
    config_path = tmp_path / "autovla.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"dummy": True}}), encoding="utf-8")

    class _FakeModel:
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.vlm = object()

        def state_dict(self):
            return {
                "vlm.model.layers.0.self_attn.q_proj.weight": 1,
                "vlm.model.layers.0.self_attn.k_proj.weight": 1,
            }

        def load_state_dict(self, state_dict, strict=False):
            return SimpleNamespace(
                missing_keys=["vlm.model.layers.0.self_attn.k_proj.weight"],
                unexpected_keys=["vlm.language_model.model.layers.0.self_attn.q_proj.weight"],
            )

        def eval(self):
            return self

    monkeypatch.setattr(navhard_mod, "AutoVLA", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(
        navhard_mod.torch,
        "load",
        lambda *args, **kwargs: {
            "state_dict": {
                "autovla.vlm.language_model.model.layers.0.self_attn.q_proj.weight": 1,
            }
        },
    )

    model_cfg = OmegaConf.create(
        {
            "config_path": str(config_path),
            "checkpoint_path": "/tmp/fake-mismatch.ckpt",
            "sensor_data_path": "/",
            "dataset_name": "navsim",
            "device": "cpu",
            "prediction_seed_mode": "none",
            "prediction_seed_base": 0,
            "lora_conf": {
                "use_lora": False,
            },
        }
    )
    trajectory_cfg = OmegaConf.create({"num_poses": 10, "interval_length": 0.5})

    with pytest.raises(RuntimeError, match="Checkpoint incompatible"):
        navhard_mod.AutoVLAPredictor(model_cfg, trajectory_cfg)


def test_autovla_predictor_fails_when_no_exact_keys_match(tmp_path, monkeypatch):
    config_path = tmp_path / "autovla.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"dummy": True}}), encoding="utf-8")

    class _FakeModel:
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.vlm = object()

        def state_dict(self):
            return {
                "vlm.model.layers.0.self_attn.q_proj.weight": 1,
            }

        def load_state_dict(self, state_dict, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def eval(self):
            return self

    monkeypatch.setattr(navhard_mod, "AutoVLA", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(
        navhard_mod.torch,
        "load",
        lambda *args, **kwargs: {
            "state_dict": {
                "autovla.vlm.language_model.model.layers.0.self_attn.q_proj.weight": 1,
            }
        },
    )

    model_cfg = OmegaConf.create(
        {
            "config_path": str(config_path),
            "checkpoint_path": "/tmp/fake-zero-exact.ckpt",
            "sensor_data_path": "/",
            "dataset_name": "navsim",
            "device": "cpu",
            "prediction_seed_mode": "none",
            "prediction_seed_base": 0,
            "lora_conf": {
                "use_lora": False,
            },
        }
    )
    trajectory_cfg = OmegaConf.create({"num_poses": 10, "interval_length": 0.5})

    with pytest.raises(RuntimeError, match="matched_exact=0"):
        navhard_mod.AutoVLAPredictor(model_cfg, trajectory_cfg)


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
