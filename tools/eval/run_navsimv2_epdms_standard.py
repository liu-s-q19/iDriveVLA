#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
import lzma
import os
import pickle
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


LOGGER = logging.getLogger("navsimv2_epdms_standard")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _repo_root() / "config" / "eval" / "navsimv2_epdms_standard.yaml"


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path.resolve()
    candidate = _repo_root() / config_path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Config file not found: {config_path}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config should be a mapping: {path}")
    return cfg


def _validate_top_level_config(cfg: Dict[str, Any]) -> None:
    for key in ["upstream", "run"]:
        if key not in cfg:
            raise ValueError(f"Missing required top-level key: {key}")

    upstream = cfg["upstream"]
    run = cfg["run"]
    required_upstream = ["navsim_root", "config_dir", "config_name", "overrides"]
    required_run = ["env_name", "output_dir", "log_path"]
    for key in required_upstream:
        if key not in upstream:
            raise ValueError(f"Missing upstream key: {key}")
    for key in required_run:
        if key not in run:
            raise ValueError(f"Missing run key: {key}")


def _ensure_upstream_navsim(navsim_root: str) -> Path:
    root = Path(navsim_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"upstream.navsim_root does not exist: {root}")
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    stale = [k for k in list(sys.modules.keys()) if k == "navsim" or k.startswith("navsim.")]
    for key in stale:
        del sys.modules[key]
    return root


def _has_override(overrides: List[str], key: str) -> bool:
    prefix = f"{key}="
    return any(item.startswith(prefix) for item in overrides)


def _extract_last_override(overrides: List[str], key: str) -> str:
    prefix = f"{key}="
    for item in reversed(overrides):
        if item.startswith(prefix):
            return item[len(prefix) :]
    return ""


def _build_effective_overrides(cfg: Dict[str, Any], cli_overrides: List[str]) -> List[str]:
    overrides = list(cfg["upstream"].get("overrides", []))
    overrides.extend(cli_overrides)
    if not _has_override(overrides, "output_dir"):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        default_out = Path(cfg["run"]["output_dir"]).resolve() / f"run_{timestamp}"
        overrides.append(f"output_dir={default_out}")
    return overrides


def _compose_upstream_cfg(cfg: Dict[str, Any], overrides: List[str]) -> Any:
    config_dir = str(Path(cfg["upstream"]["config_dir"]).resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None, job_name="epdms_standard_compose"):
        return compose(config_name=str(cfg["upstream"]["config_name"]), overrides=overrides)


def _load_metric_cache_obj(path: Path) -> Any:
    # navsim metric cache files are commonly lzma-compressed pickle files.
    try:
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    except (lzma.LZMAError, EOFError):
        with open(path, "rb") as f:
            return pickle.load(f)


def _sample_metric_cache_files(metric_cache_root: Path, sample_count: int) -> List[Path]:
    samples: List[Path] = []
    for root, _, files in os.walk(metric_cache_root):
        for name in files:
            if name == "metric_cache.pkl":
                samples.append(Path(root) / name)
                if len(samples) >= sample_count:
                    return samples
    return samples


def _validate_metric_cache_schema(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    val_cfg = cfg.get("validation", {})
    if not bool(val_cfg.get("enable_metric_cache_schema_check", True)):
        return {"enabled": False}

    metric_cache_path = _extract_last_override(overrides, "metric_cache_path")
    if not metric_cache_path:
        raise RuntimeError("metric_cache_path must be provided in upstream.overrides or --override.")
    metric_cache_root = Path(metric_cache_path).resolve()
    if not metric_cache_root.exists():
        raise FileNotFoundError(f"metric_cache_path does not exist: {metric_cache_root}")

    required_attrs = list(val_cfg.get("metric_cache_required_attrs", []))
    optional_attrs = list(val_cfg.get("metric_cache_optional_attrs", []))
    sample_count = int(val_cfg.get("metric_cache_sample_count", 3))
    samples = _sample_metric_cache_files(metric_cache_root, sample_count)
    if not samples:
        raise RuntimeError(f"No metric_cache.pkl files found under: {metric_cache_root}")

    failures: List[Dict[str, Any]] = []
    optional_missing: Dict[str, int] = {attr: 0 for attr in optional_attrs}

    for path in samples:
        obj = _load_metric_cache_obj(path)
        missing_required = [attr for attr in required_attrs if not hasattr(obj, attr)]
        if missing_required:
            failures.append({"path": str(path), "missing_required": missing_required})
        for attr in optional_attrs:
            if not hasattr(obj, attr):
                optional_missing[attr] += 1

    if failures:
        detail = json.dumps(failures, ensure_ascii=False, indent=2)
        raise RuntimeError(
            "Metric cache schema check failed. Missing required attributes in sampled files:\n"
            f"{detail}\n"
            "Expected v2-compatible metric cache. Please regenerate metric cache with upstream navsim v2."
        )

    return {
        "enabled": True,
        "metric_cache_path": str(metric_cache_root),
        "sampled_files": [str(x) for x in samples],
        "required_attrs": required_attrs,
        "optional_missing_count": optional_missing,
    }


def _build_fingerprint(hydra_cfg: Any, overrides: List[str]) -> Dict[str, Any]:
    import navsim
    import navsim.planning.script.run_pdm_score as run_pdm_score_module
    from navsim.evaluate import pdm_score as pdm_score_module
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

    pdm_sig = inspect.signature(pdm_score_module.pdm_score)
    score_sig = inspect.signature(PDMScorer.score_proposals)
    pdm_params = list(pdm_sig.parameters.keys())
    score_params = list(score_sig.parameters.keys())

    if "traffic_agents_policy" not in pdm_params:
        raise RuntimeError(
            "Detected non-standard pdm_score signature (missing traffic_agents_policy). "
            "Likely mixed into customized AutoVLA navsim path."
        )

    required_score_params = {
        "states",
        "observation",
        "centerline",
        "route_lane_ids",
        "drivable_area_map",
        "map_parameters",
    }
    if not required_score_params.issubset(set(score_params)):
        raise RuntimeError(
            "Detected non-standard PDMScorer.score_proposals signature. "
            "Likely mixed into customized AutoVLA navsim path."
        )

    scorer_obj = instantiate(hydra_cfg.scorer)
    agent_obj = instantiate(hydra_cfg.agent)

    return {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "navsim_file": str(Path(navsim.__file__).resolve()),
        "run_pdm_score_file": str(Path(run_pdm_score_module.__file__).resolve()),
        "pdm_score_signature": str(pdm_sig),
        "score_proposals_signature": str(score_sig),
        "scorer_type": f"{scorer_obj.__class__.__module__}.{scorer_obj.__class__.__name__}",
        "agent_type": f"{agent_obj.__class__.__module__}.{agent_obj.__class__.__name__}",
        "has_reactive_all_mapping": "reactive_all_mapping" in hydra_cfg.train_test_split,
        "effective_overrides": overrides,
    }


def _run_one_stage_fallback(hydra_cfg: Any) -> int:
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    from navsim.planning.script.run_pdm_score import run_pdm_score
    from nuplan.planning.utils.multithreading.worker_utils import worker_map

    worker = build_worker(hydra_cfg)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(hydra_cfg.navsim_log_path),
        synthetic_scenes_path=Path(hydra_cfg.synthetic_scenes_path),
        scene_filter=instantiate(hydra_cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(hydra_cfg.metric_cache_path))

    scene_tokens = [] if scene_loader.tokens is None else list(scene_loader.tokens)
    cache_tokens = [] if metric_cache_loader.tokens is None else list(metric_cache_loader.tokens)
    tokens_to_evaluate = list(set(scene_tokens) & set(cache_tokens))
    num_missing_metric_cache_tokens = len(set(scene_tokens) - set(cache_tokens))
    num_unused_metric_cache_tokens = len(set(cache_tokens) - set(scene_tokens))
    if num_missing_metric_cache_tokens > 0:
        LOGGER.warning("Missing metric cache for %d tokens. Skipping these tokens.", num_missing_metric_cache_tokens)
    if num_unused_metric_cache_tokens > 0:
        LOGGER.warning("Unused metric cache for %d tokens. Skipping these tokens.", num_unused_metric_cache_tokens)
    LOGGER.info(
        "Using one-stage fallback for split without reactive_all_mapping. Evaluating %d scenarios.",
        len(tokens_to_evaluate),
    )

    data_points = [
        {
            "cfg": hydra_cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows = worker_map(worker, run_pdm_score, data_points)
    pdm_score_df = pd.concat(score_rows, ignore_index=True)
    if "score" not in pdm_score_df.columns and "pdm_score" in pdm_score_df.columns:
        pdm_score_df["score"] = pdm_score_df["pdm_score"]

    num_successful = int(pdm_score_df["valid"].sum())
    num_failed = int(len(pdm_score_df) - num_successful)
    average_score = float(pdm_score_df["score"].mean(skipna=True)) if "score" in pdm_score_df.columns else float("nan")

    average_row: Dict[str, Any] = {col: None for col in pdm_score_df.columns}
    for col in pdm_score_df.columns:
        if col in {"token", "valid", "weighted_metrics", "weighted_metrics_array", "ego_simulated_states"}:
            continue
        if pd.api.types.is_numeric_dtype(pdm_score_df[col]):
            average_row[col] = float(pdm_score_df[col].mean(skipna=True))
    average_row["token"] = "average"
    average_row["valid"] = bool(pdm_score_df["valid"].all())
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(str(hydra_cfg.output_dir)).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path, index=False)

    LOGGER.info(
        "Finished fallback evaluation. successful=%d failed=%d average_score=%s csv=%s",
        num_successful,
        num_failed,
        str(average_score) if "score" in pdm_score_df.columns else "N/A",
        csv_path,
    )
    return 0


def _append_average_row(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    average_row: Dict[str, Any] = {col: None for col in pdm_score_df.columns}
    for col in pdm_score_df.columns:
        if col in {"token", "valid", "weighted_metrics", "weighted_metrics_array", "ego_simulated_states"}:
            continue
        if pd.api.types.is_numeric_dtype(pdm_score_df[col]):
            average_row[col] = float(pdm_score_df[col].mean(skipna=True))
    average_row["token"] = "average"
    average_row["valid"] = bool(pdm_score_df["valid"].all())
    pdm_score_df.loc[len(pdm_score_df)] = average_row
    return pdm_score_df


def _evaluate_autovla_one_stage_tokens(
    tokens: List[str],
    *,
    scene_loader: Any,
    metric_cache_loader: Any,
    predictor: Any,
    pdm_score_fn: Any,
    simulator: Any,
    scorer: Any,
    traffic_agents_policy: Any,
    sensor_root: Path,
    dataset_name: str,
    trajectory_num_poses: int,
    trajectory_interval: float,
    trajectory_cls: Any,
    payload_builder: Any,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for token in tokens:
        row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            payload = payload_builder(
                token=token,
                scene=scene,
                sensor_root=sensor_root,
                dataset_name=dataset_name,
                trajectory_num_poses=trajectory_num_poses,
            )
            prediction = predictor.predict(payload)
            if not isinstance(prediction, tuple):
                raise TypeError(f"predictor.predict must return tuple, got {type(prediction)!r}")
            if len(prediction) == 3:
                traj_np, traj_sampling, _cot = prediction
            elif len(prediction) == 2:
                traj_np, traj_sampling = prediction
                _cot = None
            else:
                raise ValueError(f"predictor.predict returned unexpected tuple length: {len(prediction)}")
            if traj_sampling is None:
                from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

                traj_sampling = TrajectorySampling(
                    num_poses=int(trajectory_num_poses),
                    interval_length=float(trajectory_interval),
                )
            trajectory = trajectory_cls(poses=traj_np, trajectory_sampling=traj_sampling)
            score_row = pdm_score_fn(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            if isinstance(score_row, tuple):
                score_row = score_row[0]
            if isinstance(score_row, pd.DataFrame):
                score_data = score_row.iloc[0].to_dict()
            else:
                score_data = dict(score_row)
            row.update(score_data)
        except Exception:
            LOGGER.warning("AutoVLA standard evaluation failed for token=%s", token)
            LOGGER.warning(traceback.format_exc())
            row["valid"] = False
        rows.append(row)

    pdm_score_df = pd.DataFrame(rows)
    return _append_average_row(pdm_score_df)


def _run_autovla_one_stage_from_components(
    *,
    tokens: List[str],
    output_dir: Path,
    scene_loader: Any,
    metric_cache_loader: Any,
    predictor: Any,
    pdm_score_fn: Any,
    simulator: Any,
    scorer: Any,
    traffic_agents_policy: Any,
    sensor_root: Path,
    dataset_name: str,
    trajectory_num_poses: int,
    trajectory_interval: float,
    trajectory_cls: Any,
    payload_builder: Any,
) -> int:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdm_score_df = _evaluate_autovla_one_stage_tokens(
        tokens=tokens,
        scene_loader=scene_loader,
        metric_cache_loader=metric_cache_loader,
        predictor=predictor,
        pdm_score_fn=pdm_score_fn,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
        sensor_root=sensor_root,
        dataset_name=dataset_name,
        trajectory_num_poses=trajectory_num_poses,
        trajectory_interval=trajectory_interval,
        trajectory_cls=trajectory_cls,
        payload_builder=payload_builder,
    )
    if "score" not in pdm_score_df.columns and "pdm_score" in pdm_score_df.columns:
        pdm_score_df["score"] = pdm_score_df["pdm_score"]

    scenario_df = pdm_score_df[pdm_score_df["token"] != "average"].copy()
    successful = int(scenario_df["valid"].fillna(False).sum()) if "valid" in scenario_df.columns else 0
    failed = int(len(scenario_df) - successful)
    invalid_sum = 0
    if "invalid" in scenario_df.columns:
        invalid_sum = int(pd.to_numeric(scenario_df["invalid"], errors="coerce").fillna(0).sum())
    score_mean = float(scenario_df["score"].mean(skipna=True)) if "score" in scenario_df.columns else float("nan")

    timestamp = datetime.utcnow().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = output_dir / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path, index=False)

    summary = {
        "successful": successful,
        "failed": failed,
        "invalid_sum": invalid_sum,
        "score_mean": score_mean,
        "csv_path": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Finished AutoVLA one-stage evaluation. successful=%d failed=%d invalid_sum=%d score_mean=%s csv=%s",
        successful,
        failed,
        invalid_sum,
        score_mean,
        csv_path,
    )
    return 0


def _run_autovla_one_stage(
    *,
    cfg: Dict[str, Any],
    hydra_cfg: Any,
    navsim_root: Path,
    overrides: List[str],
    dry_run: bool,
) -> int:
    del navsim_root, overrides

    if "reactive_all_mapping" in hydra_cfg.train_test_split:
        raise RuntimeError("evaluation.mode=autovla_one_stage only supports one-stage splits.")
    if "model" not in cfg:
        raise ValueError("evaluation.mode=autovla_one_stage requires a top-level model section.")

    from navsim.common.dataclasses import SensorConfig, Trajectory
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from tools.eval.run_navhard_two_stage_autovla import AutoVLAPredictor, _scene_to_autovla_payload

    simulator = instantiate(hydra_cfg.simulator)
    scorer = instantiate(hydra_cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer proposal_sampling must be identical.")

    traffic_cfg = cfg.get("traffic_agents", {})
    traffic_mode = str(traffic_cfg.get("mode", "non_reactive")).strip().lower()
    if traffic_mode == "reactive":
        traffic_agents_policy = instantiate(hydra_cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    elif traffic_mode == "non_reactive":
        traffic_agents_policy = instantiate(hydra_cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    else:
        raise ValueError(f"Unsupported traffic_agents.mode={traffic_mode}")

    sensor_config = SensorConfig(
        cam_f0=True,
        cam_l0=True,
        cam_l1=True,
        cam_l2=True,
        cam_r0=True,
        cam_r1=True,
        cam_r2=True,
        cam_b0=True,
        lidar_pc=False,
    )
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(str(hydra_cfg.synthetic_sensor_path)),
        original_sensor_path=Path(str(hydra_cfg.original_sensor_path)),
        data_path=Path(str(hydra_cfg.navsim_log_path)),
        synthetic_scenes_path=Path(str(hydra_cfg.synthetic_scenes_path)),
        scene_filter=instantiate(hydra_cfg.train_test_split.scene_filter),
        sensor_config=sensor_config,
    )
    metric_cache_loader = MetricCacheLoader(Path(str(hydra_cfg.metric_cache_path)))

    scene_tokens = [] if scene_loader.tokens is None else list(scene_loader.tokens)
    cache_token_set = set([] if metric_cache_loader.tokens is None else list(metric_cache_loader.tokens))
    tokens_to_evaluate = [token for token in scene_tokens if token in cache_token_set]
    missing_metric_cache = len([token for token in scene_tokens if token not in cache_token_set])
    if missing_metric_cache > 0:
        LOGGER.warning("Missing metric cache for %d tokens. Skipping these tokens.", missing_metric_cache)
    LOGGER.info(
        "Execution mode: autovla_one_stage, traffic_mode=%s, num_tokens=%d",
        traffic_mode,
        len(tokens_to_evaluate),
    )

    if dry_run:
        LOGGER.info("Dry-run enabled for AutoVLA one-stage mode. Skip model loading and scoring.")
        return 0

    model_cfg = OmegaConf.create(cfg["model"])
    predictor = AutoVLAPredictor(model_cfg, model_cfg.trajectory_sampling)

    return _run_autovla_one_stage_from_components(
        tokens=tokens_to_evaluate,
        output_dir=Path(str(hydra_cfg.output_dir)),
        scene_loader=scene_loader,
        metric_cache_loader=metric_cache_loader,
        predictor=predictor,
        pdm_score_fn=pdm_score,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
        sensor_root=Path(str(hydra_cfg.original_sensor_path)),
        dataset_name=str(model_cfg.get("dataset_name", "navsim")),
        trajectory_num_poses=int(model_cfg.trajectory_sampling.num_poses),
        trajectory_interval=float(model_cfg.trajectory_sampling.interval_length),
        trajectory_cls=Trajectory,
        payload_builder=_scene_to_autovla_payload,
    )


def _run_upstream_run_pdm_score(navsim_root: Path, overrides: List[str], dry_run: bool, hydra_cfg: Any) -> int:
    run_script = navsim_root / "navsim" / "planning" / "script" / "run_pdm_score.py"
    if not run_script.exists():
        raise FileNotFoundError(f"Cannot find upstream run_pdm_score.py at: {run_script}")
    cmd = [sys.executable, str(run_script)] + overrides
    has_reactive_all_mapping = "reactive_all_mapping" in hydra_cfg.train_test_split
    LOGGER.info("Execution mode: %s", "upstream_two_stage" if has_reactive_all_mapping else "wrapper_one_stage_fallback")
    LOGGER.info("Command: %s", shlex.join(cmd))
    if dry_run:
        LOGGER.info("Dry-run enabled, skip execution.")
        return 0
    if not has_reactive_all_mapping:
        return _run_one_stage_fallback(hydra_cfg)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    return proc.wait()


def _check_runtime_env(expected_env_name: str) -> None:
    current = os.environ.get("CONDA_DEFAULT_ENV", "")
    if expected_env_name and current and current != expected_env_name:
        LOGGER.warning(
            "Current conda env is '%s', while config.run.env_name is '%s'. Continue anyway.",
            current,
            expected_env_name,
        )


def _apply_process_env(cfg: Dict[str, Any]) -> None:
    env_vars = cfg.get("run", {}).get("env_vars", {})
    if not isinstance(env_vars, dict):
        raise ValueError("run.env_vars must be a mapping if provided.")
    for key, value in env_vars.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)


def _get_evaluation_mode(cfg: Dict[str, Any]) -> str:
    evaluation = cfg.get("evaluation", {})
    mode = str(evaluation.get("mode", "standard_baseline")).strip().lower()
    if mode not in {"standard_baseline", "autovla_one_stage"}:
        raise ValueError(f"Unsupported evaluation.mode={mode}")
    return mode


def _run_evaluation(
    *,
    cfg: Dict[str, Any],
    hydra_cfg: Any,
    navsim_root: Path,
    overrides: List[str],
    dry_run: bool,
    run_upstream_fn: Any,
    run_autovla_fn: Any,
) -> int:
    mode = _get_evaluation_mode(cfg)
    if mode == "autovla_one_stage":
        return run_autovla_fn(
            cfg=cfg,
            hydra_cfg=hydra_cfg,
            navsim_root=navsim_root,
            overrides=overrides,
            dry_run=dry_run,
        )
    return run_upstream_fn(
        navsim_root=navsim_root,
        overrides=overrides,
        dry_run=dry_run,
        hydra_cfg=hydra_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard navsim v2 EPDMS local template runner.")
    parser.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="Template config path. Default: config/eval/navsimv2_epdms_standard.yaml",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override passed to upstream run_pdm_score.py, e.g. metric_cache_path=/path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print fingerprint and command only.")
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = _load_yaml(cfg_path)
    _validate_top_level_config(cfg)

    log_path = Path(cfg["run"]["log_path"]).resolve()
    _setup_logging(log_path)
    _check_runtime_env(str(cfg["run"].get("env_name", "")))
    _apply_process_env(cfg)

    LOGGER.info("Using template config: %s", cfg_path)
    navsim_root = _ensure_upstream_navsim(str(cfg["upstream"]["navsim_root"]))
    LOGGER.info("Forced upstream navsim_root: %s", navsim_root)

    overrides = _build_effective_overrides(cfg, args.override)
    hydra_cfg = _compose_upstream_cfg(cfg, overrides)
    schema_result = _validate_metric_cache_schema(cfg, overrides)
    LOGGER.info("Metric cache schema check: %s", json.dumps(schema_result, ensure_ascii=False))

    fingerprint = _build_fingerprint(hydra_cfg, overrides)
    LOGGER.info("Evaluation fingerprint: %s", json.dumps(fingerprint, ensure_ascii=False))

    run_output_dir = Path(cfg["run"]["output_dir"]).resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    fingerprint_path = run_output_dir / f"fingerprint_{stamp}.json"
    with open(fingerprint_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, ensure_ascii=False, indent=2)
    LOGGER.info("Fingerprint saved: %s", fingerprint_path)

    code = _run_evaluation(
        cfg=cfg,
        hydra_cfg=hydra_cfg,
        navsim_root=navsim_root,
        overrides=overrides,
        dry_run=args.dry_run,
        run_upstream_fn=_run_upstream_run_pdm_score,
        run_autovla_fn=_run_autovla_one_stage,
    )
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
