import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class PDM_Reward:
    """
    A class that encapsulates the RL PDM reward calculation for gievn token.
    """

    def __init__(self, metric_cache_path: Path, reward_cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize the reward calculator with the given configuration.

        :param metric_cache_path: Path to the metric cache.
        """
        self.reward_cfg = reward_cfg or {}
        self.reward_type = self.reward_cfg.get("type", "epdms_stage1")
        self.traffic_agents_mode = self.reward_cfg.get("traffic_agents", "non_reactive")
        self._warned_epdms_full_fallback = False

        self.metric_cache_loader = MetricCacheLoader(metric_cache_path)
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.future_sampling)
        self.scorer = PDMScorer(
            self.future_sampling,
            config=self._build_scorer_config(self.reward_cfg),
        )
        self.traffic_agents_policy = self._build_traffic_agents_policy(self.traffic_agents_mode)

    @staticmethod
    def _build_scorer_config(reward_cfg: Dict[str, Any]) -> PDMScorerConfig:
        return PDMScorerConfig(
            progress_weight=float(reward_cfg.get("progress_weight", 5.0)),
            ttc_weight=float(reward_cfg.get("ttc_weight", 5.0)),
            lane_keeping_weight=float(reward_cfg.get("lane_keeping_weight", 2.0)),
            history_comfort_weight=float(reward_cfg.get("history_comfort_weight", 2.0)),
            two_frame_extended_comfort_weight=float(reward_cfg.get("two_frame_extended_comfort_weight", 2.0)),
            human_penalty_filter=bool(reward_cfg.get("human_penalty_filter", True)),
        )

    def _build_traffic_agents_policy(self, traffic_agents_mode: str):
        if traffic_agents_mode == "non_reactive":
            return LogReplayTrafficAgents(self.future_sampling)

        if traffic_agents_mode == "reactive":
            reactive_cfg = self.reward_cfg.get("reactive", {})
            idm_observation = NavsimIDMAgents(
                target_velocity=float(reactive_cfg.get("target_velocity", 10.0)),
                min_gap_to_lead_agent=float(reactive_cfg.get("min_gap_to_lead_agent", 1.0)),
                headway_time=float(reactive_cfg.get("headway_time", 1.5)),
                accel_max=float(reactive_cfg.get("accel_max", 1.0)),
                decel_max=float(reactive_cfg.get("decel_max", 2.0)),
                open_loop_detections_types=list(reactive_cfg.get("open_loop_detections_types", [])),
                minimum_path_length=float(reactive_cfg.get("minimum_path_length", 20)),
                planned_trajectory_samples=reactive_cfg.get("planned_trajectory_samples", None),
                planned_trajectory_sample_interval=reactive_cfg.get("planned_trajectory_sample_interval", None),
                radius=float(reactive_cfg.get("radius", 100)),
                add_open_loop_parked_vehicles=bool(reactive_cfg.get("add_open_loop_parked_vehicles", True)),
                idm_snap_threshold=float(reactive_cfg.get("idm_snap_threshold", 3.0)),
            )
            return NavsimIDMTrafficAgents(
                self.future_sampling,
                idm_agents_observation=idm_observation,
                map_root_override=reactive_cfg.get("map_root_override", None),
            )

        raise ValueError(
            f"Unsupported traffic_agents mode: {traffic_agents_mode}. "
            f"Expected one of ['non_reactive', 'reactive']."
        )

    @staticmethod
    def _extract_stage1_epdms(pdm_result_df) -> float:
        row = pdm_result_df.iloc[0]

        # v2 scorer output: stage-1 EPDMS (without two-frame EC)
        if "pdm_score" in row.index and np.isfinite(row["pdm_score"]):
            return float(row["pdm_score"])

        # Fallback if the dataframe shape/columns change in future.
        if "multiplicative_metrics_prod" in row.index and "weighted_metrics" in row.index and "weighted_metrics_array" in row.index:
            weighted_metrics = np.asarray(row["weighted_metrics"], dtype=np.float64).copy()
            weighted_metrics_array = np.asarray(row["weighted_metrics_array"], dtype=np.float64).copy()

            ec_idx = int(WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT)
            if 0 <= ec_idx < len(weighted_metrics):
                weighted_metrics[ec_idx] = 0.0
                weighted_metrics_array[ec_idx] = 0.0

            denom = weighted_metrics_array.sum()
            if denom <= 0.0:
                return 0.0

            weighted_score = (weighted_metrics * weighted_metrics_array).sum() / denom
            return float(row["multiplicative_metrics_prod"] * weighted_score)

        return 0.0

    def _select_reward_value(self, pdm_result_df) -> float:
        if self.reward_type == "epdms_stage1":
            return self._extract_stage1_epdms(pdm_result_df)

        # Placeholder mode: full EPDMS requires two-frame aggregation context.
        if self.reward_type == "epdms_full":
            if not self._warned_epdms_full_fallback:
                print("[PDM_Reward] reward.type=epdms_full currently falls back to epdms_stage1.")
                self._warned_epdms_full_fallback = True
            return self._extract_stage1_epdms(pdm_result_df)

        raise ValueError(
            f"Unsupported reward type: {self.reward_type}. "
            f"Expected one of ['epdms_stage1', 'epdms_full']."
        )

    def rl_pdm_score(self, trajectory, token):
        """
        Compute the rl pdm reward for a given token using the pdm_score metrics, excluding the two_frame_extended_comfort metric.

        :param trajectory: model output.
        :param token: The scene token.
        """
        try:
            metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        except KeyError:
            print(f"[PDM_Reward] Missing token in metric cache: {token}")
            return 0.0

        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        try:
            pdm_result_df, _ = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.future_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
                traffic_agents_policy=self.traffic_agents_policy,
            )

            return self._select_reward_value(pdm_result_df)
        except Exception as exc:
            print(f"[PDM_Reward] Reward calculation failed for token={token}: {exc}")

            return 0.0
