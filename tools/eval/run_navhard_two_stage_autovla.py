#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model

from models.autovla import AutoVLA
from tools.eval.navhard_two_stage_sharded import (
    PREDICTION_DIAGNOSTIC_COLUMNS,
    collect_all_mappings,
    filter_mappings_by_available_tokens,
    finalize_merged_results,
    intersect_tokens,
    select_stage_two_tokens,
    stable_token_seed,
)


LOGGER = logging.getLogger("navhard_two_stage_autovla")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _ensure_upstream_navsim(navsim_root: str) -> None:
    root = str(Path(navsim_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    # Ensure navsim imports come from upstream path in this process.
    stale = [k for k in sys.modules.keys() if k == "navsim" or k.startswith("navsim.")]
    for key in stale:
        del sys.modules[key]


def _safe_token_list(tokens: Optional[List[str]]) -> List[str]:
    return [] if tokens is None else list(tokens)


def _checkpoint_uses_lora(state_dict: Dict[str, Any]) -> bool:
    for key in state_dict.keys():
        if ".lora_A." in key or ".lora_B." in key or ".base_model.model." in key:
            return True
    return False


def _maybe_wrap_with_lora(model: AutoVLA, lora_conf: Optional[DictConfig], checkpoint_uses_lora: bool) -> bool:
    if not lora_conf:
        return False

    use_lora = lora_conf.get("use_lora")
    if use_lora is None:
        use_lora = checkpoint_uses_lora
        if use_lora:
            LOGGER.warning(
                "Detected LoRA-form checkpoint while model.lora_conf.use_lora is unset; enabling LoRA wrapper automatically."
            )

    if not use_lora:
        return False

    lora_config = LoraConfig(
        task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
        target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        r=int(lora_conf.get("r", 8)),
        lora_alpha=int(lora_conf.get("lora_alpha", 8)),
        lora_dropout=float(lora_conf.get("lora_dropout", 0.1)),
        bias=lora_conf.get("bias", "none"),
    )
    model.vlm = get_peft_model(model.vlm, lora_config)
    return True


def _summarize_checkpoint_compatibility(
    model: AutoVLA,
    cleaned_state_dict: Dict[str, Any],
    load_msg: Any,
) -> Dict[str, Any]:
    ignored_keys = {"training_reward_buffer", "sliding_idx", "window_count"}
    model_keys = set(model.state_dict().keys()) - ignored_keys
    checkpoint_keys = set(cleaned_state_dict.keys()) - ignored_keys
    exact_keys = model_keys & checkpoint_keys
    missing_keys = [key for key in list(load_msg.missing_keys) if key not in ignored_keys]
    unexpected_keys = [key for key in list(load_msg.unexpected_keys) if key not in ignored_keys]
    return {
        "matched_exact": len(exact_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def _raise_on_checkpoint_incompatibility(summary: Dict[str, Any], checkpoint_uses_lora: bool) -> None:
    missing_keys = list(summary["missing_keys"])
    unexpected_keys = list(summary["unexpected_keys"])
    matched_exact = int(summary["matched_exact"])
    if not missing_keys and not unexpected_keys and matched_exact > 0:
        return

    raise RuntimeError(
        "Checkpoint incompatible with evaluation model: "
        f"matched_exact={matched_exact} "
        f"missing={len(missing_keys)} "
        f"unexpected={len(unexpected_keys)} "
        f"checkpoint_uses_lora={checkpoint_uses_lora}"
    )


def _intersect_tokens(lhs: Optional[List[str]], rhs: Optional[List[str]]) -> List[str]:
    return intersect_tokens(lhs, rhs)


def _filter_mappings_by_available_tokens(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    available_tokens: List[str],
) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    available = set(available_tokens)
    filtered: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for (orig_token, prev_token), pairs in all_mappings.items():
        if orig_token not in available or prev_token not in available:
            continue
        valid_pairs = [pair for pair in pairs if len(pair) >= 2 and pair[0] in available and pair[1] in available]
        if valid_pairs:
            filtered[(orig_token, prev_token)] = valid_pairs
    return filtered


def _command_to_text(driving_command: Any) -> str:
    command = np.asarray(driving_command)
    if command.ndim == 0:
        idx = int(command)
    else:
        idx = int(np.argmax(command))
    if idx == 0:
        return "TURN LEFT"
    if idx == 1:
        return "KEEP FORWARD"
    if idx == 2:
        return "TURN RIGHT"
    return "UNKNOWN"


def _resolve_camera_path(camera_path: Any, sensor_root: Path) -> str:
    if camera_path is None:
        raise ValueError("camera_path is None")
    p = Path(str(camera_path))
    if p.is_absolute():
        return str(p)
    return str((sensor_root / p).resolve())


def _pad_trajectory(poses: np.ndarray, target_len: int) -> np.ndarray:
    if poses.shape[0] == 0:
        return np.zeros((target_len, 3), dtype=np.float32)
    if poses.shape[0] >= target_len:
        return poses[:target_len].astype(np.float32)
    pad = np.repeat(poses[-1:, :], target_len - poses.shape[0], axis=0)
    return np.concatenate([poses, pad], axis=0).astype(np.float32)


def _build_prediction_diagnostics(
    protocol_result: Optional[Dict[str, Any]],
    raw_pose_count: int,
    target_num_poses: int,
) -> Dict[str, Any]:
    result = dict(protocol_result or {})
    raw_count = max(int(raw_pose_count), 0)
    target_len = max(int(target_num_poses), 0)
    protocol_valid = int(result.get("protocol_valid", 0))
    protocol_reason = str(result.get("invalid_reason", "not_run") or "")
    action_tokens_count = int(result.get("answer_action_tokens_len", 0) or 0)
    was_padded = int(raw_count < target_len)
    was_truncated = int(raw_count > target_len)
    used_zero_fallback = int(raw_count == 0 and target_len > 0)
    return {
        "protocol_valid": protocol_valid,
        "protocol_reason": protocol_reason,
        "action_tokens_count": action_tokens_count,
        "raw_pose_count": raw_count,
        "was_padded": was_padded,
        "was_truncated": was_truncated,
        "used_zero_fallback": used_zero_fallback,
    }


def _scene_to_autovla_payload(
    token: str,
    scene: Any,
    sensor_root: Path,
    dataset_name: str,
    trajectory_num_poses: int,
) -> Dict[str, Any]:
    agent_input = scene.get_agent_input()
    ego_status = agent_input.ego_statuses[-1]

    cams = agent_input.cameras
    if len(cams) < 4:
        raise ValueError(f"expected >=4 history frames, got {len(cams)}")

    front = [_resolve_camera_path(c.cam_f0.camera_path, sensor_root) for c in cams]
    front_left = [_resolve_camera_path(c.cam_l0.camera_path, sensor_root) for c in cams]
    front_right = [_resolve_camera_path(c.cam_r0.camera_path, sensor_root) for c in cams]
    left = [_resolve_camera_path(c.cam_l1.camera_path, sensor_root) for c in cams]
    right = [_resolve_camera_path(c.cam_r1.camera_path, sensor_root) for c in cams]
    back = [_resolve_camera_path(c.cam_b0.camera_path, sensor_root) for c in cams]
    back_left = [_resolve_camera_path(c.cam_l2.camera_path, sensor_root) for c in cams]
    back_right = [_resolve_camera_path(c.cam_r2.camera_path, sensor_root) for c in cams]

    try:
        gt_poses = scene.get_future_trajectory().poses
    except Exception:
        gt_poses = np.zeros((0, 3), dtype=np.float32)
    gt_trajectory = _pad_trajectory(np.asarray(gt_poses, dtype=np.float32), trajectory_num_poses).tolist()

    try:
        his_trajectory = scene.get_history_trajectory().poses.tolist()
    except Exception:
        his_trajectory = []

    return {
        "token": token,
        "dataset_name": dataset_name,
        "cot_output": [],
        "velocity": np.asarray(ego_status.ego_velocity, dtype=np.float32).tolist(),
        "acceleration": np.asarray(ego_status.ego_acceleration, dtype=np.float32).tolist(),
        "instruction": _command_to_text(ego_status.driving_command),
        "gt_trajectory": gt_trajectory,
        "his_trajectory": his_trajectory,
        "front_camera_paths": front,
        "front_left_camera_paths": front_left,
        "front_right_camera_paths": front_right,
        "left_camera_paths": left,
        "right_camera_paths": right,
        "back_camera_paths": back,
        "back_left_camera_paths": back_left,
        "back_right_camera_paths": back_right,
    }


class AutoVLAPredictor:
    def __init__(self, model_cfg: DictConfig, trajectory_sampling_cfg: DictConfig):
        with open(str(model_cfg.config_path), "r", encoding="utf-8") as f:
            autovla_config = yaml.safe_load(f)
        device = str(model_cfg.get("device", "cuda"))
        self.model = AutoVLA(autovla_config, device=device)
        self.sensor_data_path = str(model_cfg.sensor_data_path)
        self.interval_length = float(trajectory_sampling_cfg.interval_length)
        self.num_poses = int(trajectory_sampling_cfg.num_poses)
        self.prediction_seed_mode = str(model_cfg.get("prediction_seed_mode", "token_hash")).lower()
        self.prediction_seed_base = int(model_cfg.get("prediction_seed_base", 0))
        self.last_protocol_result: Dict[str, Any] = {}
        self.last_prediction_diagnostics: Dict[str, Any] = _build_prediction_diagnostics({}, 0, self.num_poses)

        state: Dict[str, Any] = torch.load(str(model_cfg.checkpoint_path), map_location=self.model.device)
        state_dict = state.get("state_dict", state)
        checkpoint_uses_lora = _checkpoint_uses_lora(state_dict)
        _maybe_wrap_with_lora(self.model, model_cfg.get("lora_conf"), checkpoint_uses_lora)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("autovla."):
                cleaned_state_dict[key.replace("autovla.", "", 1)] = value
            else:
                cleaned_state_dict[key] = value
        msg = self.model.load_state_dict(cleaned_state_dict, strict=False)
        compatibility = _summarize_checkpoint_compatibility(
            model=self.model,
            cleaned_state_dict=cleaned_state_dict,
            load_msg=msg,
        )
        LOGGER.info(
            "Checkpoint compatibility summary: matched_exact=%d missing=%d unexpected=%d checkpoint_uses_lora=%s",
            compatibility["matched_exact"],
            len(compatibility["missing_keys"]),
            len(compatibility["unexpected_keys"]),
            checkpoint_uses_lora,
        )
        _raise_on_checkpoint_incompatibility(compatibility, checkpoint_uses_lora)
        self.model.eval()

    def predict(self, payload: Dict[str, Any]) -> Tuple[Any, str]:
        if self.prediction_seed_mode == "token_hash":
            seed = stable_token_seed(str(payload.get("token", "")), seed_base=self.prediction_seed_base)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        elif self.prediction_seed_mode not in {"none", ""}:
            raise ValueError(f"Unsupported prediction_seed_mode={self.prediction_seed_mode}")

        features = {
            "vehicle_velocity": payload["velocity"],
            "vehicle_acceleration": payload["acceleration"],
            "driving_command": payload["instruction"],
            "images": {
                "front_camera": payload["front_camera_paths"],
                "front_left_camera": payload["front_left_camera_paths"],
                "front_right_camera": payload["front_right_camera_paths"],
                "left_camera": payload["left_camera_paths"],
                "right_camera": payload["right_camera_paths"],
                "back_camera": payload["back_camera_paths"],
                "back_left_camera": payload["back_left_camera_paths"],
                "back_right_camera": payload["back_right_camera_paths"],
            },
            "dataset_name": payload["dataset_name"],
            "gt_trajectory": payload["gt_trajectory"],
            "history_trajectory": payload["his_trajectory"] if payload["his_trajectory"] else payload["gt_trajectory"],
            # Using "/" keeps get_prompt path-join active while preserving absolute image paths.
            "sensor_data_path": self.sensor_data_path,
        }
        with torch.no_grad():
            traj_tensor, cot = self.model.predict(features)
        self.last_protocol_result = dict(getattr(self.model, "_last_protocol_result", {}))

        traj_np = traj_tensor.detach().cpu().numpy().astype(np.float32)
        if traj_np.ndim != 2 or traj_np.shape[1] != 3:
            raise ValueError(f"invalid trajectory shape from model: {traj_np.shape}")
        self.last_prediction_diagnostics = _build_prediction_diagnostics(
            self.last_protocol_result,
            raw_pose_count=int(traj_np.shape[0]),
            target_num_poses=self.num_poses,
        )
        traj_np = _pad_trajectory(traj_np, self.num_poses)

        sampling = TrajectorySampling(num_poses=self.num_poses, interval_length=self.interval_length)
        return traj_np, sampling, cot


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    df = pdm_score_df.reset_index()
    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."
    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores
    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)
    return df


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]
    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()
    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)
    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame,
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():
        first_tokens = [pair[0] for pair in second_stage_pairs if len(pair) > 0]
        second_tokens = [pair[1] for pair in second_stage_pairs if len(pair) > 1]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[pdm_score_df["token"].isin(first_tokens)]

        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[pdm_score_df["token"].isin(second_tokens)]

        group1_stage1_scores = calculate_weighted_average_score(group1_stage1_df)
        group1_stage2_scores = calculate_weighted_average_score(group1_stage2_df)
        group2_stage1_scores = calculate_weighted_average_score(group2_stage1_df)
        group2_stage2_scores = calculate_weighted_average_score(group2_stage2_df)

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)
        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores
        group2_scores = group2_stage1_scores * group2_stage2_scores
        all_group_scores.append((group1_scores + group2_scores) / 2)

    if not all_group_scores:
        nan_series = pd.Series(dtype=float)
        return nan_series, nan_series, nan_series

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()
        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])
    return full_score_df


def _evaluate_token(
    token: str,
    stage_name: str,
    scene_loader: Any,
    metric_cache_loader: Any,
    predictor: AutoVLAPredictor,
    pdm_score_fn: Any,
    traffic_agents_policy: Any,
    simulator: Any,
    scorer: Any,
    sensor_root: Path,
    dataset_name: str,
    trajectory_num_poses: int,
    trajectory_interval: float,
    trajectory_cls: Any,
    pdm_results_cls: Any,
    debug_dump_dir: Optional[Path],
    debug_dump_limit: int,
    debug_counter: List[int],
) -> pd.DataFrame:
    metric_cache = None
    diagnostics = _build_prediction_diagnostics(
        getattr(predictor, "last_protocol_result", {}),
        raw_pose_count=0,
        target_num_poses=trajectory_num_poses,
    )
    try:
        metric_cache = metric_cache_loader.get_from_token(token)
        scene = scene_loader.get_scene_from_token(token)
        payload = _scene_to_autovla_payload(
            token=token,
            scene=scene,
            sensor_root=sensor_root,
            dataset_name=dataset_name,
            trajectory_num_poses=trajectory_num_poses,
        )
        if debug_dump_dir is not None and debug_counter[0] < debug_dump_limit:
            debug_dump_dir.mkdir(parents=True, exist_ok=True)
            with open(debug_dump_dir / f"{token}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            debug_counter[0] += 1

        traj_np, traj_sampling, _cot = predictor.predict(payload)
        diagnostics = dict(getattr(predictor, "last_prediction_diagnostics", diagnostics))
        trajectory = trajectory_cls(poses=traj_np, trajectory_sampling=traj_sampling)

        try:
            score_row, ego_simulated_states = pdm_score_fn(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
        except TypeError as e:
            if "score_proposals" not in str(e):
                raise
            LOGGER.warning(
                "Falling back to scorer compatibility path for token=%s stage=%s due to signature mismatch: %s",
                token,
                stage_name,
                e,
            )
            from navsim.evaluate import pdm_score as pdm_score_module

            pred_trajectory = pdm_score_module.transform_trajectory(trajectory, metric_cache.ego_state)
            initial_ego_state = metric_cache.ego_state
            pdm_states = pdm_score_module.get_trajectory_as_array(
                metric_cache.trajectory,
                simulator.proposal_sampling,
                initial_ego_state.time_point,
            )
            pred_states = pdm_score_module.get_trajectory_as_array(
                pred_trajectory,
                simulator.proposal_sampling,
                initial_ego_state.time_point,
            )
            trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
            simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
            simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
                simulated_states[1], metric_cache
            )
            if len(simulated_agent_detections_tracks) != trajectory_states.shape[1]:
                raise ValueError(
                    f"traffic length mismatch: expected={trajectory_states.shape[1]} "
                    f"got={len(simulated_agent_detections_tracks)}"
                )

            call_kwargs: Dict[str, Any] = {}
            try:
                params = inspect.signature(scorer.score_proposals).parameters
            except Exception:
                params = {}
            if "map_parameters" in params:
                call_kwargs["map_parameters"] = metric_cache.map_parameters
            if "simulated_agent_detections_tracks" in params:
                call_kwargs["simulated_agent_detections_tracks"] = simulated_agent_detections_tracks
            if "human_past_trajectory" in params and hasattr(metric_cache, "past_human_trajectory"):
                call_kwargs["human_past_trajectory"] = metric_cache.past_human_trajectory

            try:
                score_list = scorer.score_proposals(
                    states=simulated_states,
                    observation=metric_cache.observation,
                    centerline=metric_cache.centerline,
                    route_lane_ids=metric_cache.route_lane_ids,
                    drivable_area_map=metric_cache.drivable_area_map,
                    **call_kwargs,
                )
            except TypeError:
                score_list = scorer.score_proposals(
                    simulated_states,
                    metric_cache.observation,
                    metric_cache.centerline,
                    metric_cache.route_lane_ids,
                    metric_cache.drivable_area_map,
                )

            pred_idx = 1
            score_row = score_list[pred_idx]
            ego_simulated_states = simulated_states[pred_idx]
        if not isinstance(score_row, pd.DataFrame):
            score_row = pd.DataFrame([score_row])

        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        end_pose = StateSE2(
            x=float(trajectory.poses[-1, 0]),
            y=float(trajectory.poses[-1, 1]),
            heading=float(trajectory.poses[-1, 2]),
        )
        absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
        score_row["endpoint_x"] = absolute_endpoint.x
        score_row["endpoint_y"] = absolute_endpoint.y
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_simulated_states]
    except Exception:
        LOGGER.warning("Agent failed for token=%s stage=%s", token, stage_name)
        LOGGER.warning(traceback.format_exc())
        score_row = pd.DataFrame([asdict(pdm_results_cls.get_empty_results())])
        score_row["valid"] = False
        if metric_cache is not None:
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
        else:
            score_row["frame_type"] = np.nan

    score_row["token"] = token
    for key in PREDICTION_DIAGNOSTIC_COLUMNS:
        if key in diagnostics:
            score_row[key] = diagnostics[key]
    return score_row


def _load_config(config_path: Path, dotlist: List[str]) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Scene-driven navhard two-stage evaluation for AutoVLA.")
    parser.add_argument(
        "--config",
        type=str,
        default="/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dotlist, e.g. --set eval.max_stage_one_scenarios=20",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config), args.set)
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))

    from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

    with initialize_config_dir(config_dir=str(cfg.upstream.config_dir), version_base=None):
        upstream_cfg = compose(
            config_name=str(cfg.upstream.config_name),
            overrides=[str(x) for x in cfg.upstream.overrides],
        )

    output_dir = Path(str(cfg.eval.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n# upstream_resolved\n")
        f.write(OmegaConf.to_yaml(upstream_cfg))

    simulator: PDMSimulator = instantiate(upstream_cfg.simulator)
    scorer: PDMScorer = instantiate(upstream_cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer proposal_sampling must be identical.")

    traffic_mode = str(cfg.traffic_agents.mode).lower()
    if traffic_mode == "reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            upstream_cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
        )
    elif traffic_mode == "non_reactive":
        traffic_agents_policy = instantiate(upstream_cfg.traffic_agents_policy.non_reactive)
    else:
        raise ValueError(f"Unsupported traffic_agents.mode={traffic_mode}")

    predictor = AutoVLAPredictor(cfg.model, cfg.model.trajectory_sampling)
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
        synthetic_sensor_path=Path(str(upstream_cfg.synthetic_sensor_path)),
        original_sensor_path=Path(str(upstream_cfg.original_sensor_path)),
        data_path=Path(str(upstream_cfg.navsim_log_path)),
        synthetic_scenes_path=Path(str(upstream_cfg.synthetic_scenes_path)),
        scene_filter=instantiate(upstream_cfg.train_test_split.scene_filter),
        sensor_config=sensor_config,
    )
    metric_cache_loader = MetricCacheLoader(Path(str(upstream_cfg.metric_cache_path)))
    scene_filter_cfg = upstream_cfg.train_test_split.scene_filter

    stage1_tokens = _intersect_tokens(scene_loader.tokens_stage_one, metric_cache_loader.tokens)
    if cfg.eval.max_stage_one_scenarios is not None:
        stage1_tokens = stage1_tokens[: int(cfg.eval.max_stage_one_scenarios)]

    stage2_tokens = select_stage_two_tokens(
        traffic_mode=traffic_mode,
        available_synthetic_tokens=list(scene_loader.synthetic_scenes.keys()),
        reactive_initial_tokens=scene_filter_cfg.get("reactive_synthetic_initial_tokens"),
        non_reactive_initial_tokens=scene_filter_cfg.get("non_reactive_synthetic_initial_tokens"),
        metric_cache_tokens=metric_cache_loader.tokens,
    )
    if cfg.eval.max_stage_two_scenarios is not None:
        stage2_tokens = stage2_tokens[: int(cfg.eval.max_stage_two_scenarios)]

    LOGGER.info(
        "Eval setup: stage1_tokens=%d stage2_tokens=%d traffic_mode=%s metric_tokens=%d scene_tokens=%d",
        len(stage1_tokens),
        len(stage2_tokens),
        traffic_mode,
        len(metric_cache_loader.tokens),
        len(scene_loader.tokens),
    )

    synthetic_tokens = set(scene_loader.synthetic_scenes.keys())
    orig_sensor_root = Path(str(upstream_cfg.original_sensor_path))
    synth_sensor_root = Path(str(upstream_cfg.synthetic_sensor_path))
    dataset_name = str(cfg.model.dataset_name)
    traj_num_poses = int(cfg.model.trajectory_sampling.num_poses)
    traj_interval = float(cfg.model.trajectory_sampling.interval_length)

    debug_dump_dir = None
    if cfg.debug.dump_scene_data_dir:
        debug_dump_dir = Path(str(cfg.debug.dump_scene_data_dir))
    debug_dump_limit = int(cfg.debug.dump_max_items)
    debug_counter = [0]

    pdm_results: List[pd.DataFrame] = []
    t0 = time.time()

    for idx, token in enumerate(stage1_tokens, start=1):
        LOGGER.info("Stage1 token %d/%d: %s", idx, len(stage1_tokens), token)
        score_row = _evaluate_token(
            token=token,
            stage_name="stage_one",
            scene_loader=scene_loader,
            metric_cache_loader=metric_cache_loader,
            predictor=predictor,
            pdm_score_fn=pdm_score,
            traffic_agents_policy=traffic_agents_policy,
            simulator=simulator,
            scorer=scorer,
            sensor_root=orig_sensor_root,
            dataset_name=dataset_name,
            trajectory_num_poses=traj_num_poses,
            trajectory_interval=traj_interval,
            trajectory_cls=Trajectory,
            pdm_results_cls=PDMResults,
            debug_dump_dir=debug_dump_dir,
            debug_dump_limit=debug_dump_limit,
            debug_counter=debug_counter,
        )
        pdm_results.append(score_row)

    for idx, token in enumerate(stage2_tokens, start=1):
        LOGGER.info("Stage2 token %d/%d: %s", idx, len(stage2_tokens), token)
        sensor_root = synth_sensor_root if token in synthetic_tokens else orig_sensor_root
        score_row = _evaluate_token(
            token=token,
            stage_name="stage_two",
            scene_loader=scene_loader,
            metric_cache_loader=metric_cache_loader,
            predictor=predictor,
            pdm_score_fn=pdm_score,
            traffic_agents_policy=traffic_agents_policy,
            simulator=simulator,
            scorer=scorer,
            sensor_root=sensor_root,
            dataset_name=dataset_name,
            trajectory_num_poses=traj_num_poses,
            trajectory_interval=traj_interval,
            trajectory_cls=Trajectory,
            pdm_results_cls=PDMResults,
            debug_dump_dir=debug_dump_dir,
            debug_dump_limit=debug_dump_limit,
            debug_counter=debug_counter,
        )
        pdm_results.append(score_row)

    if not pdm_results:
        raise RuntimeError("No scenarios were evaluated. Check split/cache/path settings.")

    pdm_score_df = pd.concat(pdm_results, ignore_index=True)

    all_mappings = collect_all_mappings(
        raw_mapping=upstream_cfg.train_test_split.reactive_all_mapping,
        scene_tokens=scene_loader.tokens,
    )
    available_tokens = [str(x) for x in pdm_score_df["token"].tolist()]
    eval_mappings = filter_mappings_by_available_tokens(all_mappings, available_tokens)
    LOGGER.info(
        "Mapping coverage: raw=%d filtered=%d available_tokens=%d",
        len(all_mappings),
        len(eval_mappings),
        len(set(available_tokens)),
    )
    pdm_score_df, summary = finalize_merged_results(
        combined_rows=pdm_score_df,
        all_mappings=all_mappings,
        proposal_sampling=instantiate(upstream_cfg.simulator.proposal_sampling),
        scene_frame_type_original=SceneFrameType.ORIGINAL,
        scene_frame_type_synthetic=SceneFrameType.SYNTHETIC,
        pdm_result_field_names=[score.name for score in fields(PDMResults)],
        output_dir=run_dir,
        write_artifacts=False,
        logger=LOGGER,
    )

    csv_path = run_dir / f"{datetime.utcnow().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
    pdm_score_df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    summary["csv_path"] = str(csv_path)
    summary["elapsed_sec"] = elapsed
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("Finished evaluation: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
