"""Quantization error diagnostics for the action token codebook.

This script replays preprocessed scene JSON files, re-tokenizes the ground-truth
trajectory using the current action codebook, and reports the residual error
introduced by the discrete representation (an upper bound on policy quality).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

# Add repo root to sys.path so this script works when invoked standalone.
REPO_ROOT = Path(__file__).resolve().parents[2]
NAVSIM_ROOT = REPO_ROOT / "navsim"

import sys

for path in (REPO_ROOT, NAVSIM_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

    if "PYTHONPATH" in os.environ:
        if str(path) not in os.environ["PYTHONPATH"].split(":"):
            os.environ["PYTHONPATH"] = f"{path}:{os.environ['PYTHONPATH']}"
    else:
        os.environ["PYTHONPATH"] = str(path)

from navsim.agents.utils import cal_polygon_contour, transform_to_global, wrap_angle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute action token quantization error stats")
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Directory that contains preprocessed *.json scenes",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=Path("codebook_cache/agent_vocab.pkl"),
        help="Path to the action codebook pickle (default: codebook_cache/agent_vocab.pkl)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional limit on the number of scenes to process (for quick sanity checks)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for the distance calculations",
    )
    return parser.parse_args()


def load_codebook(path: Path, device: torch.device) -> torch.Tensor:
    with open(path, "rb") as f:
        data = pickle.load(f)
    codebook = torch.tensor(data["token_all"]["veh"], dtype=torch.float32, device=device)
    # We only need the terminal polygon per token for matching (see TokenProcessor)
    return codebook[:, -1]  # [n_token, 4, 2] in the ego frame


def iter_scene_files(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.glob("*.json") if p.is_file())


def scene_quant_error(
    gt_traj: torch.Tensor,
    token_traj: torch.Tensor,
    agent_shape: torch.Tensor,
    device: torch.device,
    thresholds: Tuple[float, ...],
) -> Dict[str, object]:
    """Replays a single trajectory and records quantization residues."""

    prev_pos = torch.zeros((1, 2), dtype=torch.float32, device=device)
    prev_head = torch.zeros(1, dtype=torch.float32, device=device)

    center_errors: List[float] = []
    bbox_errors: List[float] = []
    heading_errors: List[float] = []
    threshold_hits = {thr: 0 for thr in thresholds}

    token_traj = token_traj.unsqueeze(0)  # [1, n_token, 4, 2]

    for step in range(gt_traj.shape[0]):
        gt_pose = gt_traj[step]
        if torch.isnan(gt_pose).any():
            continue

        gt_pos = gt_pose[:2].view(1, 1, 2)
        gt_head = gt_pose[2].view(1, 1)
        gt_contour = cal_polygon_contour(gt_pos, gt_head, agent_shape).unsqueeze(1)

        token_world = transform_to_global(
            pos_local=token_traj.flatten(1, 2),
            head_local=None,
            pos_now=prev_pos,
            head_now=prev_head,
        )[0].view_as(token_traj)

        # RISK: averaging over (-1, -2) collapses token dimension as well; downstream
        # argmin/indexing assumes a per-token distance vector.
        dist = torch.norm(token_world - gt_contour, dim=-1).mean((-1, -2))
        best_idx = torch.argmin(dist, dim=-1).item()
        best_dist = dist[0, best_idx]
        best_contour = token_world[0, best_idx]

        pred_center = best_contour.mean(dim=0)
        gt_center = gt_contour[0, 0].mean(dim=0)
        center_err = torch.norm(pred_center - gt_center).item()

        dxy = best_contour[0] - best_contour[3]
        pred_heading = torch.atan2(dxy[1], dxy[0])
        heading_err = torch.abs(wrap_angle(gt_head.squeeze() - pred_heading)).item()

        center_errors.append(center_err)
        bbox_errors.append(best_dist.item())
        heading_errors.append(heading_err)

        for thr in thresholds:
            if center_err <= thr:
                threshold_hits[thr] += 1

        prev_pos = pred_center.unsqueeze(0)
        prev_head = pred_heading.view(1)

    if not center_errors:
        return {}

    return {
        "num_steps": len(center_errors),
        "center_errors": center_errors,
        "bbox_errors": bbox_errors,
        "heading_errors": heading_errors,
        "threshold_hits": threshold_hits,
    }


def summarize(results: List[Dict[str, object]], thresholds: Tuple[float, ...]) -> Dict[str, float]:
    total_steps = sum(r["num_steps"] for r in results)
    if total_steps == 0:
        return {}

    ade_sum = sum(sum(r["center_errors"]) for r in results)
    heading_sum = sum(sum(r["heading_errors"]) for r in results)
    bbox_sum = sum(sum(r["bbox_errors"]) for r in results)
    fde_list = [r["center_errors"][-1] for r in results]
    max_center = max(max(r["center_errors"]) for r in results)

    thresh_hits = {thr: sum(r["threshold_hits"][thr] for r in results) for thr in thresholds}

    summary = {
        "num_scenes": len(results),
        "num_steps": total_steps,
        "ade_m": ade_sum / total_steps,
        "fde_m": sum(fde_list) / len(fde_list),
        "heading_mae_deg": (heading_sum / total_steps) * 180.0 / math.pi,
        "bbox_l2": bbox_sum / total_steps,
        "max_center_err": max_center,
    }

    for thr, count in thresh_hits.items():
        summary[f"pct_le_{thr:.2f}m"] = 100.0 * count / total_steps

    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    thresholds = (0.05, 0.10, 0.25, 0.50)

    token_traj = load_codebook(args.codebook, device)
    agent_shape = torch.tensor([[2.0, 4.8]], dtype=torch.float32, device=device)

    scene_files = list(iter_scene_files(args.data_path))
    if args.max_scenes is not None:
        scene_files = scene_files[: args.max_scenes]

    if not scene_files:
        raise FileNotFoundError(f"No JSON files found under {args.data_path}")

    stats: List[Dict[str, object]] = []
    progress = tqdm(scene_files, desc="Quantizing scenes")
    for scene_path in progress:
        with open(scene_path, "r") as f:
            scene = json.load(f)

        gt = scene.get("gt_trajectory")
        if gt is None:
            continue

        traj = torch.tensor(gt, dtype=torch.float32, device=device)
        if traj.ndim != 2 or traj.shape[1] < 3:
            continue

        scene_stats = scene_quant_error(traj[:, :3], token_traj, agent_shape, device, thresholds)
        if scene_stats:
            stats.append(scene_stats)

    summary = summarize(stats, thresholds)
    if not summary:
        raise RuntimeError("No valid steps collected; double-check the dataset path")

    print("=== Action Token Quantization Summary ===")
    for k, v in summary.items():
        if k.startswith("pct_le_"):
            print(f"{k:>16s}: {v:6.2f}%")
        else:
            print(f"{k:>16s}: {v:.4f}")


if __name__ == "__main__":
    main()
