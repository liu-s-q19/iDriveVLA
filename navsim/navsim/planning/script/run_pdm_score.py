from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import math

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.utils import cal_polygon_contour, transform_to_global, wrap_angle
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.visualization.plots import plot_cameras_frame_with_bev_agent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _load_quantization_codebook(cfg: DictConfig) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load codebook tensors used for GT quantization in evaluation, if enabled."""
    quant_cfg = cfg.get("quantization")
    if not quant_cfg or not quant_cfg.get("enable", False):
        return None, None

    codebook_path = quant_cfg.get("codebook_path")
    if not codebook_path:
        raise ValueError("quantization.codebook_path must be set when quantization.enable=true")

    device = torch.device(str(quant_cfg.get("device", "cpu")))
    with open(Path(codebook_path), "rb") as f:
        data = pickle.load(f)

    token_traj = torch.tensor(data["token_all"]["veh"], dtype=torch.float32, device=device)[:, -1]  # [n, 4, 2]
    agent_width = float(quant_cfg.get("agent_width", 2.0))
    agent_length = float(quant_cfg.get("agent_length", 4.8))
    agent_shape = torch.tensor([[agent_width, agent_length]], dtype=torch.float32, device=device)
    return token_traj, agent_shape


def _quantize_trajectory_with_codebook(
    trajectory: Trajectory,
    token_traj: torch.Tensor,
    agent_shape: torch.Tensor,
) -> Tuple[Trajectory, Dict[str, float]]:
    """Quantize trajectory step-by-step with nearest codebook token and return metrics."""
    poses = torch.tensor(trajectory.poses, dtype=torch.float32, device=token_traj.device)
    pos = poses[None, :, :2]
    heading = poses[None, :, 2]
    token_traj = token_traj.unsqueeze(0)  # [1, n_token, 4, 2]

    prev_pos = torch.zeros((1, 2), dtype=torch.float32, device=token_traj.device)
    prev_head = torch.zeros(1, dtype=torch.float32, device=token_traj.device)

    quantized_poses: List[List[float]] = []
    center_errors: List[float] = []
    bbox_errors: List[float] = []
    heading_errors: List[float] = []

    for step in range(poses.shape[0]):
        gt_contour = cal_polygon_contour(pos[:, step], heading[:, step], agent_shape)  # [1, 4, 2]
        token_world = transform_to_global(
            pos_local=token_traj.flatten(1, 2),
            head_local=None,
            pos_now=prev_pos,
            head_now=prev_head,
        )[0].view(*token_traj.shape)

        dists = torch.norm(token_world - gt_contour.unsqueeze(1), dim=-1).sum(-1)  # [1, n_token]
        token_idx = torch.argmin(dists, dim=-1)  # [1]
        token_contour = token_world[0, token_idx]  # [1, 4, 2]

        pred_center = token_contour.mean(1).squeeze(0)  # [2]
        dxy = token_contour[:, 0] - token_contour[:, 3]
        pred_heading = torch.arctan2(dxy[:, 1], dxy[:, 0]).squeeze(0)

        gt_center = gt_contour[0].mean(0)
        center_err = torch.norm(pred_center - gt_center).item()
        bbox_err = torch.norm(token_contour.squeeze(0) - gt_contour.squeeze(0), dim=-1).mean().item()
        heading_err = torch.abs(wrap_angle(heading[:, step] - pred_heading.unsqueeze(0))).item()

        quantized_poses.append([pred_center[0].item(), pred_center[1].item(), pred_heading.item()])
        center_errors.append(center_err)
        bbox_errors.append(bbox_err)
        heading_errors.append(heading_err)

        prev_pos = pred_center.unsqueeze(0)
        prev_head = pred_heading.unsqueeze(0)

    quantized_np = np.asarray(quantized_poses, dtype=np.float32)
    quantized_traj = Trajectory(poses=quantized_np, trajectory_sampling=trajectory.trajectory_sampling)

    num_steps = len(center_errors)
    metrics = {
        "quant_num_steps": float(num_steps),
        "quant_ade_m": float(sum(center_errors) / num_steps),
        "quant_fde_m": float(center_errors[-1]),
        "quant_heading_mae_deg": float(sum(heading_errors) / num_steps * 180.0 / math.pi),
        "quant_bbox_l2": float(sum(bbox_errors) / num_steps),
        "quant_max_center_err": float(max(center_errors)),
    }
    return quantized_traj, metrics


def _trajectory_from_metric_cache(metric_cache: MetricCache, future_sampling: TrajectorySampling) -> Trajectory:
    """Build local-frame trajectory directly from cached PDM-Closed trajectory."""
    pdm_states = get_trajectory_as_array(metric_cache.trajectory, future_sampling, metric_cache.ego_state.time_point)
    future_global_se2 = pdm_states[1:, StateIndex.STATE_SE2]
    future_local_se2 = convert_absolute_to_relative_se2_array(metric_cache.ego_state.rear_axle, future_global_se2)
    return Trajectory(poses=future_local_se2.astype(np.float32), trajectory_sampling=future_sampling)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    trajectory_source = str(cfg.get("trajectory_source", "agent"))
    if trajectory_source not in {"agent", "metric_cache"}:
        raise ValueError(f"Unsupported trajectory_source={trajectory_source}, expected one of ['agent', 'metric_cache']")

    agent: Optional[AbstractAgent] = None
    sensor_config = SensorConfig.build_no_sensors()
    if trajectory_source == "agent":
        agent = instantiate(cfg.agent)
        agent.initialize()
        sensor_config = agent.get_sensor_config()

    token_traj, agent_shape = _load_quantization_codebook(cfg)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            if trajectory_source == "metric_cache":
                trajectory = _trajectory_from_metric_cache(metric_cache, simulator.proposal_sampling)
            else:
                assert agent is not None
                agent_input = scene_loader.get_agent_input_from_token(token)
                if agent.requires_scene:
                    scene = scene_loader.get_scene_from_token(token)
                    trajectory = agent.compute_trajectory(agent_input, scene)
                else:
                    trajectory = agent.compute_trajectory(agent_input)

            if token_traj is not None and agent_shape is not None:
                trajectory, quant_metrics = _quantize_trajectory_with_codebook(trajectory, token_traj, agent_shape)
                score_row.update(quant_metrics)

            # scene = scene_loader.get_scene_from_token(token)
            # frame_idx = scene.scene_metadata.num_history_frames - 1
            # fig, _ = plot_cameras_frame_with_bev_agent(scene, frame_idx, agent_trajectory=trajectory)
            # vis_dir = Path(cfg.output_dir) / "Visualization"
            # vis_dir.mkdir(parents=True, exist_ok=True)
            # vis_path = vis_dir / f"{token}_bevagent.png"
            # fig.savefig(vis_path, bbox_inches="tight")
            # plt.close(fig)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    quant_cols = {
        "quant_num_steps",
        "quant_ade_m",
        "quant_fde_m",
        "quant_heading_mae_deg",
        "quant_bbox_l2",
        "quant_max_center_err",
    }
    if quant_cols.issubset(set(pdm_score_df.columns)):
        valid_quant_df = pdm_score_df[pdm_score_df["valid"]]
        valid_quant_df = valid_quant_df[valid_quant_df["quant_num_steps"] > 0]
        if not valid_quant_df.empty:
            total_steps = valid_quant_df["quant_num_steps"].sum()
            average_row["quant_num_steps"] = total_steps
            average_row["quant_ade_m"] = (valid_quant_df["quant_ade_m"] * valid_quant_df["quant_num_steps"]).sum() / total_steps
            average_row["quant_fde_m"] = valid_quant_df["quant_fde_m"].mean()
            average_row["quant_heading_mae_deg"] = (
                valid_quant_df["quant_heading_mae_deg"] * valid_quant_df["quant_num_steps"]
            ).sum() / total_steps
            average_row["quant_bbox_l2"] = (
                valid_quant_df["quant_bbox_l2"] * valid_quant_df["quant_num_steps"]
            ).sum() / total_steps
            average_row["quant_max_center_err"] = valid_quant_df["quant_max_center_err"].max()
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
