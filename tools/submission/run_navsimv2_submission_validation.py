#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.submission.run_navsimv2_autovla_submission import (
    _apply_process_env,
    _collect_tokens_from_mapping,
    _collect_split_tokens,
    _ensure_upstream_navsim,
    _load_config,
    _repo_root,
    _resolve_config_path,
    _scene_loader_module_name,
    _setup_logging,
)


def _validate_submission_schema(submission_data: Dict[str, Any]) -> None:
    required = {
        "team_name",
        "authors",
        "email",
        "institution",
        "country / region",
        "first_stage_predictions",
        "second_stage_predictions",
    }
    missing = sorted(required - set(submission_data.keys()))
    if missing:
        raise ValueError(f"Missing required submission fields: {', '.join(missing)}")


def _load_prediction_dicts(submission_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    first_stage = submission_data["first_stage_predictions"]
    second_stage = submission_data["second_stage_predictions"]
    if not isinstance(first_stage, list) or not isinstance(second_stage, list):
        raise ValueError("Submission predictions must be lists of single-seed dictionaries.")
    if len(first_stage) != 1 or len(second_stage) != 1:
        raise ValueError("Submission predictions must contain exactly one seed.")
    if not isinstance(first_stage[0], dict) or not isinstance(second_stage[0], dict):
        raise ValueError("Submission predictions must contain dictionaries keyed by token.")
    return first_stage[0], second_stage[0]


def _build_expected_tokens(cfg: Any) -> Tuple[List[str], List[str]]:
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))
    overrides = [str(x) for x in cfg.upstream.get("overrides", [])]
    overrides.extend(
        [
            f"train_test_split={cfg.data.train_test_split}",
            f"navsim_log_path={cfg.data.navsim_log_path}",
            f"original_sensor_path={cfg.data.original_sensor_path}",
            f"synthetic_sensor_path={cfg.data.synthetic_sensor_path}",
            f"synthetic_scenes_path={cfg.data.synthetic_scenes_path}",
            f"output_dir={Path(str(cfg.validation.output_dir)).resolve()}",
        ]
    )
    with initialize_config_dir(config_dir=str(Path(cfg.upstream.config_dir).resolve()), version_base=None):
        upstream_cfg = compose(config_name=str(cfg.upstream.submission_config_name), overrides=overrides)
    return _collect_split_tokens(upstream_cfg)


def _build_expected_sampling(cfg: Any) -> Tuple[int, float]:
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))
    overrides = [str(x) for x in cfg.upstream.get("overrides", [])]
    overrides.extend(
        [
            f"train_test_split={cfg.data.train_test_split}",
            f"navsim_log_path={cfg.data.navsim_log_path}",
            f"original_sensor_path={cfg.data.original_sensor_path}",
            f"synthetic_sensor_path={cfg.data.synthetic_sensor_path}",
            f"synthetic_scenes_path={cfg.data.synthetic_scenes_path}",
            f"output_dir={Path(str(cfg.validation.output_dir)).resolve()}",
        ]
    )
    with initialize_config_dir(config_dir=str(Path(cfg.upstream.config_dir).resolve()), version_base=None):
        upstream_cfg = compose(config_name=str(cfg.upstream.submission_config_name), overrides=overrides)
    sampling = upstream_cfg.agent.trajectory_sampling
    interval = float(getattr(sampling, "interval_length", 0.0) or 0.0)
    num_poses = int(getattr(sampling, "num_poses", 0) or 0)
    if num_poses <= 0:
        time_horizon = float(getattr(sampling, "time_horizon", 0.0) or 0.0)
        if time_horizon > 0 and interval > 0:
            num_poses = int(round(time_horizon / interval))
    if num_poses <= 0 or interval <= 0:
        raise ValueError("Failed to resolve expected official trajectory sampling from upstream config.")
    return num_poses, interval


def _collect_sampling_mismatches(
    predictions: Dict[str, Any],
    expected_num_poses: int,
    expected_interval_length: float,
) -> List[str]:
    mismatches: List[str] = []
    for token, trajectory in predictions.items():
        poses = getattr(trajectory, "poses", None)
        sampling = getattr(trajectory, "trajectory_sampling", None)
        num_poses = getattr(sampling, "num_poses", None)
        interval_length = getattr(sampling, "interval_length", None)
        pose_rows = getattr(poses, "shape", [None])[0] if poses is not None else None
        if (
            poses is None
            or sampling is None
            or pose_rows != expected_num_poses
            or num_poses != expected_num_poses
            or float(interval_length) != float(expected_interval_length)
        ):
            mismatches.append(str(token))
    return mismatches


def _validate_private_submission(
    submission_data: Dict[str, Any],
    submission_path: Path,
    expected_first_stage_tokens: List[str],
    expected_second_stage_tokens: List[str],
    expected_num_poses: int,
    expected_interval_length: float,
    split_name: str,
) -> Dict[str, Any]:
    _validate_submission_schema(submission_data)
    first_stage, second_stage = _load_prediction_dicts(submission_data)
    first_stage_tokens = set(first_stage.keys())
    second_stage_tokens = set(second_stage.keys())
    duplicate_tokens = sorted(first_stage_tokens & second_stage_tokens)
    missing_first = sorted(set(expected_first_stage_tokens) - first_stage_tokens)
    missing_second = sorted(set(expected_second_stage_tokens) - second_stage_tokens)
    extra_first = sorted(first_stage_tokens - set(expected_first_stage_tokens))
    extra_second = sorted(second_stage_tokens - set(expected_second_stage_tokens))
    sampling_mismatch_first = _collect_sampling_mismatches(
        first_stage, expected_num_poses=expected_num_poses, expected_interval_length=expected_interval_length
    )
    sampling_mismatch_second = _collect_sampling_mismatches(
        second_stage, expected_num_poses=expected_num_poses, expected_interval_length=expected_interval_length
    )

    valid = (
        submission_path.name == "submission.pkl"
        and not duplicate_tokens
        and not missing_first
        and not missing_second
        and not extra_first
        and not extra_second
        and not sampling_mismatch_first
        and not sampling_mismatch_second
    )

    return {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "split_name": split_name,
        "submission_file": str(submission_path),
        "valid": valid,
        "first_stage_tokens": len(first_stage_tokens),
        "second_stage_tokens": len(second_stage_tokens),
        "missing_first_stage_tokens": missing_first,
        "missing_second_stage_tokens": missing_second,
        "extra_first_stage_tokens": extra_first,
        "extra_second_stage_tokens": extra_second,
        "duplicate_tokens_across_stages": duplicate_tokens,
        "sampling_mismatch_first_stage_tokens": sampling_mismatch_first,
        "sampling_mismatch_second_stage_tokens": sampling_mismatch_second,
        "expected_num_poses": expected_num_poses,
        "expected_interval_length": expected_interval_length,
        "requires_submission_filename": True,
        "filename_ok": submission_path.name == "submission.pkl",
    }


def _build_warmup_summary(score_df: pd.DataFrame, csv_path: Path, split_name: str) -> Dict[str, Any]:
    token_rows = score_df[~score_df["token"].astype(str).str.startswith("extended_pdm_score")]
    final_row = score_df.loc[score_df["token"] == "extended_pdm_score_combined"]
    final_score = float(final_row["score"].iloc[0]) if not final_row.empty else float("nan")
    successful = int(token_rows["valid"].fillna(False).sum())
    failed = int((~token_rows["valid"].fillna(False)).sum())
    return {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "split_name": split_name,
        "valid": pd.notna(final_score),
        "successful": successful,
        "failed": failed,
        "final_extended_pdm_score": final_score,
        "csv_path": str(csv_path),
    }


def _build_warmup_validation_command(cfg: Any, submission_path: Path, output_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    navsim_root = str(Path(cfg.upstream.navsim_root).resolve())
    env["PYTHONPATH"] = f"{navsim_root}:{existing_pythonpath}" if existing_pythonpath else navsim_root
    cmd = [
        sys.executable,
        str(Path(cfg.upstream.navsim_root) / "navsim" / "planning" / "script" / "run_pdm_score_from_submission.py"),
        f"train_test_split={cfg.data.train_test_split}",
        f"submission_file_path={submission_path}",
        f"metric_cache_path={cfg.data.metric_cache_path}",
        f"navsim_log_path={cfg.data.navsim_log_path}",
        f"original_sensor_path={cfg.data.original_sensor_path}",
        f"synthetic_sensor_path={cfg.data.synthetic_sensor_path}",
        f"synthetic_scenes_path={cfg.data.synthetic_scenes_path}",
        f"output_dir={output_dir}",
    ]
    return cmd, env


def _run_warmup_validation(cfg: Any, submission_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd, env = _build_warmup_validation_command(cfg, submission_path=submission_path, output_dir=output_dir)
    subprocess.run(cmd, check=True, cwd=str(_repo_root()), env=env)
    csv_files = sorted(output_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("Warmup validation did not produce a csv result.")
    csv_path = csv_files[-1]
    score_df = pd.read_csv(csv_path)
    return _build_warmup_summary(score_df, csv_path=csv_path, split_name=str(cfg.data.train_test_split))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate official NavSim v2 submission.pkl locally.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/submission/navsimv2_autovla_warmup.yaml",
        help="Path to submission config YAML.",
    )
    parser.add_argument("--set", action="append", default=[], help="Override config values with key=value.")
    parser.add_argument("--submission-file", type=str, required=True, help="Path to submission.pkl to validate.")
    args = parser.parse_args()

    cfg = _load_config(_resolve_config_path(args.config), args.set)
    _setup_logging(Path(str(cfg.run.log_path)))
    _apply_process_env(cfg)

    submission_path = Path(args.submission_file).resolve()
    with open(submission_path, "rb") as f:
        submission_data = pickle.load(f)

    output_dir = Path(str(cfg.validation.output_dir)).resolve() / f"run_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    if str(cfg.validation.mode) == "warmup_local_score":
        summary = _run_warmup_validation(cfg, submission_path=submission_path, output_dir=output_dir)
    else:
        expected_first, expected_second = _build_expected_tokens(cfg)
        expected_num_poses, expected_interval_length = _build_expected_sampling(cfg)
        summary = _validate_private_submission(
            submission_data=submission_data,
            submission_path=submission_path,
            expected_first_stage_tokens=expected_first,
            expected_second_stage_tokens=expected_second,
            expected_num_poses=expected_num_poses,
            expected_interval_length=expected_interval_length,
            split_name=str(cfg.data.train_test_split),
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0 if summary.get("valid", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
