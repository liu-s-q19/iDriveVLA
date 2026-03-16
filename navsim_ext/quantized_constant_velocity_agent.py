import math
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _cal_polygon_contour(pos: torch.Tensor, heading: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Build 4-corner box contour from center pose and box shape."""
    width = shape[:, 0]
    length = shape[:, 1]
    half_w = width / 2.0
    half_l = length / 2.0
    local = torch.stack(
        [
            torch.stack([half_l, half_w], dim=-1),
            torch.stack([half_l, -half_w], dim=-1),
            torch.stack([-half_l, -half_w], dim=-1),
            torch.stack([-half_l, half_w], dim=-1),
        ],
        dim=1,
    )  # [B, 4, 2]

    c = torch.cos(heading)
    s = torch.sin(heading)
    rot = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )  # [B, 2, 2]
    return torch.matmul(local, rot.transpose(-1, -2)) + pos.unsqueeze(1)


def _transform_to_global(pos_local: torch.Tensor, pos_now: torch.Tensor, head_now: torch.Tensor) -> torch.Tensor:
    """Transform local 2D points into global frame using current pose."""
    c = torch.cos(head_now).unsqueeze(-1)
    s = torch.sin(head_now).unsqueeze(-1)
    x = pos_local[..., 0]
    y = pos_local[..., 1]
    gx = c * x - s * y + pos_now[..., 0:1]
    gy = s * x + c * y + pos_now[..., 1:2]
    return torch.stack([gx, gy], dim=-1)


class QuantizedConstantVelocityAgent(AbstractAgent):
    """Constant-velocity baseline followed by codebook trajectory quantization."""

    requires_scene = False

    def __init__(
        self,
        codebook_path: str,
        device: str = "cpu",
        agent_width: float = 2.0,
        agent_length: float = 4.8,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)
        self._codebook_path = Path(codebook_path)
        self._device = torch.device(device)
        self._agent_width = float(agent_width)
        self._agent_length = float(agent_length)
        self._base_agent = ConstantVelocityAgent(trajectory_sampling)
        self._token_traj: torch.Tensor = torch.empty(0)
        self._agent_shape: torch.Tensor = torch.empty(0)
        self.last_quant_metrics = {}

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._base_agent.initialize()
        with open(self._codebook_path, "rb") as f:
            data = pickle.load(f)
        self._token_traj = torch.tensor(data["token_all"]["veh"], dtype=torch.float32, device=self._device)[:, -1]
        self._agent_shape = torch.tensor(
            [[self._agent_width, self._agent_length]],
            dtype=torch.float32,
            device=self._device,
        )

    def get_sensor_config(self) -> SensorConfig:
        return self._base_agent.get_sensor_config()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        base_traj = self._base_agent.compute_trajectory(agent_input)
        quantized, metrics = self._quantize_trajectory(base_traj)
        self.last_quant_metrics = metrics
        return quantized

    def _quantize_trajectory(self, trajectory: Trajectory):
        poses = torch.tensor(trajectory.poses, dtype=torch.float32, device=self._token_traj.device)
        pos = poses[None, :, :2]
        heading = poses[None, :, 2]
        token_traj = self._token_traj.unsqueeze(0)

        prev_pos = torch.zeros((1, 2), dtype=torch.float32, device=token_traj.device)
        prev_head = torch.zeros(1, dtype=torch.float32, device=token_traj.device)

        quantized_poses: List[List[float]] = []
        center_errors: List[float] = []
        bbox_errors: List[float] = []
        heading_errors: List[float] = []

        for step in range(poses.shape[0]):
            gt_contour = _cal_polygon_contour(pos[:, step], heading[:, step], self._agent_shape)
            token_world = _transform_to_global(
                pos_local=token_traj.flatten(1, 2),
                pos_now=prev_pos,
                head_now=prev_head,
            ).view(*token_traj.shape)

            dists = torch.norm(token_world - gt_contour.unsqueeze(1), dim=-1).sum(-1)
            token_idx = torch.argmin(dists, dim=-1)
            token_contour = token_world[0, token_idx]

            pred_center = token_contour.mean(1).squeeze(0)
            dxy = token_contour[:, 0] - token_contour[:, 3]
            pred_heading = torch.arctan2(dxy[:, 1], dxy[:, 0]).squeeze(0)

            gt_center = gt_contour[0].mean(0)
            center_err = torch.norm(pred_center - gt_center).item()
            bbox_err = torch.norm(token_contour.squeeze(0) - gt_contour.squeeze(0), dim=-1).mean().item()
            heading_err = torch.abs(_wrap_angle(heading[:, step] - pred_heading.unsqueeze(0))).item()

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
