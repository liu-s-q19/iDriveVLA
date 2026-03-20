#!/usr/bin/env python3
import argparse
import importlib
import json
import logging
import os
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("navsimv2_submission")


class _SceneFromAgentInput:
    """Adapter for official challenge loaders that only expose AgentInput."""

    def __init__(self, agent_input: Any):
        self._agent_input = agent_input

    def get_agent_input(self) -> Any:
        return self._agent_input

    def get_future_trajectory(self) -> Any:
        raise RuntimeError("future trajectory is unavailable for submission-only private inputs")

    def get_history_trajectory(self) -> Any:
        raise RuntimeError("history trajectory is unavailable for submission-only private inputs")


def _repo_root() -> Path:
    return PROJECT_ROOT


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


def _load_config(path: Path, dotlist: List[str]) -> Any:
    cfg = OmegaConf.load(path)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


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


def _apply_process_env(cfg: Any) -> None:
    for key, value in OmegaConf.to_container(cfg.run.env_vars, resolve=True).items():
        if value is not None:
            os.environ[str(key)] = str(value)


def _ensure_upstream_navsim(navsim_root: str) -> Path:
    root = Path(navsim_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"upstream.navsim_root does not exist: {root}")
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    stale = [key for key in list(sys.modules.keys()) if key == "navsim" or key.startswith("navsim.")]
    for key in stale:
        del sys.modules[key]
    return root


def _scene_loader_module_name(split_name: str) -> str:
    if split_name in {"warmup_two_stage", "navhard_two_stage"}:
        return "navsim.common.dataloader"
    if split_name == "private_test_hard_two_stage":
        return "navsim.common.dataloader_private"
    raise ValueError(f"Unsupported submission split: {split_name}")


def _load_scene_loader_cls(split_name: str):
    module = importlib.import_module(_scene_loader_module_name(split_name))
    return module.SceneLoader


def _camera_sensor_config():
    from navsim.common.dataclasses import SensorConfig

    return SensorConfig(
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


def _build_upstream_overrides(cfg: Any, output_dir: Path) -> List[str]:
    overrides = [str(x) for x in cfg.upstream.get("overrides", [])]
    overrides.extend(
        [
            f"train_test_split={cfg.data.train_test_split}",
            f"navsim_log_path={cfg.data.navsim_log_path}",
            f"original_sensor_path={cfg.data.original_sensor_path}",
            f"synthetic_sensor_path={cfg.data.synthetic_sensor_path}",
            f"synthetic_scenes_path={cfg.data.synthetic_scenes_path}",
            f"output_dir={output_dir}",
        ]
    )
    return overrides


def _compose_upstream_cfg(cfg: Any, output_dir: Path) -> Any:
    with initialize_config_dir(config_dir=str(Path(cfg.upstream.config_dir).resolve()), version_base=None):
        return compose(
            config_name=str(cfg.upstream.submission_config_name),
            overrides=_build_upstream_overrides(cfg, output_dir),
        )


def _build_submission_metadata(submission_cfg: Any) -> Dict[str, str]:
    metadata = {
        "team_name": str(submission_cfg.team_name),
        "authors": str(submission_cfg.authors),
        "email": str(submission_cfg.email),
        "institution": str(submission_cfg.institution),
        "country / region": str(submission_cfg.country),
    }
    missing = [key for key, value in metadata.items() if not value or value == "None" or value == "MUST_SET"]
    if missing:
        raise ValueError(f"Submission metadata must be configured before generation: {', '.join(missing)}")
    return metadata


def _build_submission_payload(
    metadata: Dict[str, str],
    first_stage_predictions: Dict[str, Any],
    second_stage_predictions: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(metadata)
    payload["first_stage_predictions"] = [first_stage_predictions]
    payload["second_stage_predictions"] = [second_stage_predictions]
    return payload


def _collect_tokens_from_mapping(raw_mapping: List[Tuple[str, str, List[Tuple[str, str]]]]) -> Tuple[List[str], List[str]]:
    first_stage_tokens: List[str] = []
    second_stage_tokens: List[str] = []
    first_seen = set()
    second_seen = set()

    for orig_token, prev_token, two_stage_pairs in raw_mapping:
        for token in (orig_token, prev_token):
            if token not in first_seen:
                first_seen.add(token)
                first_stage_tokens.append(token)
        for stage_one_token, stage_two_token in two_stage_pairs:
            for token in (stage_one_token, stage_two_token):
                if token not in second_seen:
                    second_seen.add(token)
                    second_stage_tokens.append(token)
    return first_stage_tokens, second_stage_tokens


def _collect_split_tokens(upstream_cfg: Any) -> Tuple[List[str], List[str]]:
    scene_filter = upstream_cfg.train_test_split.scene_filter
    first_stage_tokens = list(scene_filter.get("tokens") or [])
    second_stage_tokens = list(scene_filter.get("reactive_synthetic_initial_tokens") or [])

    data_split = getattr(upstream_cfg.train_test_split, "data_split", None)
    if data_split is None and hasattr(upstream_cfg.train_test_split, "get"):
        data_split = upstream_cfg.train_test_split.get("data_split", "")

    if str(data_split or "") == "private_test_hard":
        first_stage_tokens = []

    if first_stage_tokens and second_stage_tokens:
        return first_stage_tokens, second_stage_tokens

    raw_mapping = upstream_cfg.train_test_split.get("reactive_all_mapping")
    if raw_mapping is not None:
        return _collect_tokens_from_mapping(list(raw_mapping))

    return first_stage_tokens, second_stage_tokens


def _build_run_summary(
    split_name: str,
    loader_module_name: str,
    output_dir: Path,
    first_stage_predictions: Dict[str, Any],
    second_stage_predictions: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    summary = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "split_name": split_name,
        "loader_module_name": loader_module_name,
        "first_stage_tokens": len(first_stage_predictions),
        "second_stage_tokens": len(second_stage_predictions),
        "dry_run": dry_run,
    }
    if not dry_run:
        summary["submission_file"] = str(output_dir / "submission.pkl")
    return summary


def _get_run_id(cfg: Any) -> str:
    run_id = cfg.run.get("run_id")
    if run_id:
        return str(run_id)
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def _resolve_run_paths(cfg: Any) -> Tuple[Path, Path]:
    base_output_dir = Path(str(cfg.run.output_dir)).resolve()
    run_dir = base_output_dir / f"run_{_get_run_id(cfg)}"
    num_shards = int(cfg.run.get("num_shards", 1) or 1)
    shard_index = int(cfg.run.get("shard_index", 0) or 0)
    if num_shards > 1:
        shard_dir = run_dir / f"shard_{shard_index:02d}_of_{num_shards:02d}"
    else:
        shard_dir = run_dir
    return run_dir, shard_dir


def _slice_tokens_for_shard(tokens: List[str], shard_index: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return list(tokens)
    return list(tokens)[shard_index::num_shards]


def _load_prediction_dicts(submission_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    first_stage = submission_payload["first_stage_predictions"]
    second_stage = submission_payload["second_stage_predictions"]
    if not isinstance(first_stage, list) or not isinstance(second_stage, list):
        raise ValueError("Shard submission payload is malformed.")
    if len(first_stage) != 1 or len(second_stage) != 1:
        raise ValueError("Shard submission payload must contain exactly one seed.")
    return first_stage[0], second_stage[0]


def _merge_submission_shards(cfg: Any) -> int:
    run_dir, _ = _resolve_run_paths(cfg)
    num_shards = int(cfg.run.get("num_shards", 1) or 1)
    if num_shards <= 1:
        raise ValueError("merge_only requires run.num_shards > 1")

    metadata = _build_submission_metadata(cfg.submission)
    merged_first: Dict[str, Any] = {}
    merged_second: Dict[str, Any] = {}
    shard_summaries: List[Dict[str, Any]] = []

    for shard_index in range(num_shards):
        shard_dir = run_dir / f"shard_{shard_index:02d}_of_{num_shards:02d}"
        shard_submission_path = shard_dir / "submission.pkl"
        shard_summary_path = shard_dir / "summary.json"
        if not shard_submission_path.exists():
            raise FileNotFoundError(f"Missing shard submission: {shard_submission_path}")
        if not shard_summary_path.exists():
            raise FileNotFoundError(f"Missing shard summary: {shard_summary_path}")
        with open(shard_submission_path, "rb") as f:
            shard_payload = pickle.load(f)
        with open(shard_summary_path, "r", encoding="utf-8") as f:
            shard_summaries.append(json.load(f))
        first_stage, second_stage = _load_prediction_dicts(shard_payload)
        overlap_first = sorted(set(merged_first) & set(first_stage))
        overlap_second = sorted(set(merged_second) & set(second_stage))
        if overlap_first or overlap_second:
            raise ValueError(
                f"Duplicate shard tokens detected. first={overlap_first[:5]} second={overlap_second[:5]}"
            )
        merged_first.update(first_stage)
        merged_second.update(second_stage)

    resolved_parts = []
    for shard_index in range(num_shards):
        shard_dir = run_dir / f"shard_{shard_index:02d}_of_{num_shards:02d}"
        resolved_parts.append(
            f"# shard_{shard_index:02d}\n" + (shard_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )

    summary = _build_run_summary(
        split_name=str(cfg.data.train_test_split),
        loader_module_name=_scene_loader_module_name(str(cfg.data.train_test_split)),
        output_dir=run_dir,
        first_stage_predictions=merged_first,
        second_stage_predictions=merged_second,
        dry_run=False,
    )
    summary["num_shards"] = num_shards
    summary["merged_from_shards"] = [str(run_dir / f"shard_{i:02d}_of_{num_shards:02d}") for i in range(num_shards)]
    summary = _write_submission_artifacts(
        output_dir=run_dir,
        resolved_config_text="\n\n".join(resolved_parts) + "\n",
        submission_payload=_build_submission_payload(metadata, merged_first, merged_second),
        summary=summary,
    )
    LOGGER.info("Merged submission written to %s", summary["submission_file"])
    return 0


def _write_submission_artifacts(
    output_dir: Path,
    resolved_config_text: str,
    submission_payload: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.yaml"
    summary_path = output_dir / "summary.json"
    submission_path = output_dir / "submission.pkl"

    resolved_path.write_text(resolved_config_text, encoding="utf-8")
    with open(submission_path, "wb") as f:
        pickle.dump(submission_payload, f)

    summary = dict(summary)
    summary["resolved_config"] = str(resolved_path)
    summary["summary_file"] = str(summary_path)
    summary["submission_file"] = str(submission_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _predict_for_tokens(
    tokens: List[str],
    scene_loader: Any,
    predictor: Any,
    sensor_root: Path,
    dataset_name: str,
    split_name: str,
    trajectory_num_poses: int,
    trajectory_interval: float,
    trajectory_cls: Any,
    payload_builder: Any,
) -> Dict[str, Any]:
    predictions: Dict[str, Any] = {}
    failed_tokens: List[str] = []
    sampling = TrajectorySampling(num_poses=trajectory_num_poses, interval_length=trajectory_interval)

    for token in tokens:
        try:
            if split_name == "private_test_hard_two_stage":
                scene = _SceneFromAgentInput(scene_loader.get_agent_input_from_token(token))
            else:
                scene = scene_loader.get_scene_from_token(token)
            payload = payload_builder(
                token=token,
                scene=scene,
                sensor_root=sensor_root,
                dataset_name=dataset_name,
                trajectory_num_poses=trajectory_num_poses,
            )
            pred_out = predictor.predict(payload)
            if isinstance(pred_out, tuple):
                if len(pred_out) == 2:
                    traj_np, _cot = pred_out
                elif len(pred_out) >= 3:
                    traj_np = pred_out[0]
                else:
                    raise ValueError("predictor returned empty tuple")
            else:
                traj_np = pred_out
            predictions[token] = trajectory_cls(poses=traj_np, trajectory_sampling=sampling)
        except Exception:
            failed_tokens.append(token)
            LOGGER.warning("Failed to generate submission trajectory for token=%s", token)
            LOGGER.warning(traceback.format_exc())

    if failed_tokens:
        raise RuntimeError(f"Failed to generate trajectories for tokens: {', '.join(failed_tokens)}")
    return predictions


def _limit_tokens(tokens: List[str], max_scenes: Any) -> List[str]:
    token_list = list(tokens)
    if max_scenes is None:
        return token_list
    return token_list[: int(max_scenes)]


def _normalize_tokens(tokens: List[str]) -> List[str]:
    """Ensure deterministic shard assignment across independent processes."""
    return sorted(str(token) for token in tokens)


def _run_submission_generation(cfg: Any, dry_run: bool) -> int:
    _apply_process_env(cfg)
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))
    if cfg.run.get("merge_only"):
        return _merge_submission_shards(cfg)

    from navsim.common.dataclasses import Trajectory
    from tools.eval.run_navhard_two_stage_autovla import AutoVLAPredictor, _scene_to_autovla_payload

    run_dir, artifact_dir = _resolve_run_paths(cfg)
    upstream_cfg = _compose_upstream_cfg(cfg, run_dir)
    num_shards = int(cfg.run.get("num_shards", 1) or 1)
    shard_index = int(cfg.run.get("shard_index", 0) or 0)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")

    loader_cls = _load_scene_loader_cls(str(cfg.data.train_test_split))
    sensor_config = _camera_sensor_config()
    if dry_run:
        first_stage_tokens, second_stage_tokens = _collect_split_tokens(upstream_cfg)
        scene_loader = None
    else:
        scene_loader = loader_cls(
            data_path=Path(str(upstream_cfg.navsim_log_path)),
            scene_filter=instantiate(upstream_cfg.train_test_split.scene_filter),
            synthetic_sensor_path=Path(str(upstream_cfg.synthetic_sensor_path)),
            original_sensor_path=Path(str(upstream_cfg.original_sensor_path)),
            synthetic_scenes_path=Path(str(upstream_cfg.synthetic_scenes_path)),
            sensor_config=sensor_config,
        )
        first_stage_tokens = list(scene_loader.tokens_stage_one)
        second_stage_tokens = list(scene_loader.reactive_tokens_stage_two)

    first_stage_tokens = _normalize_tokens(first_stage_tokens)
    second_stage_tokens = _normalize_tokens(second_stage_tokens)
    first_stage_tokens = _limit_tokens(first_stage_tokens, cfg.run.get("max_scenes"))
    second_stage_tokens = _limit_tokens(second_stage_tokens, cfg.run.get("max_scenes"))
    first_stage_tokens = _slice_tokens_for_shard(first_stage_tokens, shard_index=shard_index, num_shards=num_shards)
    second_stage_tokens = _slice_tokens_for_shard(second_stage_tokens, shard_index=shard_index, num_shards=num_shards)

    resolved_config_text = OmegaConf.to_yaml(cfg)
    resolved_config_text += "\n# upstream_resolved\n"
    resolved_config_text += OmegaConf.to_yaml(upstream_cfg)

    summary = _build_run_summary(
        split_name=str(cfg.data.train_test_split),
        loader_module_name=_scene_loader_module_name(str(cfg.data.train_test_split)),
        output_dir=artifact_dir,
        first_stage_predictions={token: None for token in first_stage_tokens},
        second_stage_predictions={token: None for token in second_stage_tokens},
        dry_run=dry_run,
    )
    summary["num_shards"] = num_shards
    summary["shard_index"] = shard_index

    if dry_run:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "resolved_config.yaml").write_text(resolved_config_text, encoding="utf-8")
        with open(artifact_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        LOGGER.info("Dry-run summary written to %s", artifact_dir / "summary.json")
        return 0

    metadata = _build_submission_metadata(cfg.submission)
    predictor = AutoVLAPredictor(cfg.model, cfg.model.trajectory_sampling)
    first_stage_predictions = _predict_for_tokens(
        tokens=first_stage_tokens,
        scene_loader=scene_loader,
        predictor=predictor,
        sensor_root=Path(str(cfg.data.original_sensor_path)),
        dataset_name=str(cfg.model.dataset_name),
        split_name=str(cfg.data.train_test_split),
        trajectory_num_poses=int(cfg.model.trajectory_sampling.num_poses),
        trajectory_interval=float(cfg.model.trajectory_sampling.interval_length),
        trajectory_cls=Trajectory,
        payload_builder=_scene_to_autovla_payload,
    )
    second_stage_predictions = _predict_for_tokens(
        tokens=second_stage_tokens,
        scene_loader=scene_loader,
        predictor=predictor,
        sensor_root=Path(str(cfg.data.synthetic_sensor_path)),
        dataset_name=str(cfg.model.dataset_name),
        split_name=str(cfg.data.train_test_split),
        trajectory_num_poses=int(cfg.model.trajectory_sampling.num_poses),
        trajectory_interval=float(cfg.model.trajectory_sampling.interval_length),
        trajectory_cls=Trajectory,
        payload_builder=_scene_to_autovla_payload,
    )

    submission_payload = _build_submission_payload(metadata, first_stage_predictions, second_stage_predictions)
    summary = _build_run_summary(
        split_name=str(cfg.data.train_test_split),
        loader_module_name=_scene_loader_module_name(str(cfg.data.train_test_split)),
        output_dir=artifact_dir,
        first_stage_predictions=first_stage_predictions,
        second_stage_predictions=second_stage_predictions,
        dry_run=False,
    )
    summary["num_shards"] = num_shards
    summary["shard_index"] = shard_index
    summary = _write_submission_artifacts(artifact_dir, resolved_config_text, submission_payload, summary)
    LOGGER.info("Submission written to %s", summary["submission_file"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate official NavSim v2 submission.pkl for AutoVLA.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/submission/navsimv2_autovla_warmup.yaml",
        help="Path to submission config YAML.",
    )
    parser.add_argument("--set", action="append", default=[], help="Override config values with key=value.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and token counts without model inference.")
    args = parser.parse_args()

    cfg = _load_config(_resolve_config_path(args.config), args.set)
    _setup_logging(Path(str(cfg.run.log_path)))
    return _run_submission_generation(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
