#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tools.eval.run_navhard_two_stage_autovla import AutoVLAPredictor, _ensure_upstream_navsim, _evaluate_token


LOGGER = logging.getLogger("navhard_two_stage_autovla_shard")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shard of navhard two-stage AutoVLA evaluation.")
    parser.add_argument("--plan-dir", type=str, required=True, help="Shard plan directory created by prepare_navhard_two_stage_shards.py")
    parser.add_argument("--shard-index", type=int, required=True, help="Shard index to execute")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir).resolve()
    plan = json.loads((plan_dir / "shard_plan.json").read_text(encoding="utf-8"))
    manifest_path = Path(plan["manifests_dir"]) / f"shard_{int(args.shard_index):02d}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_dir = Path(manifest["partial_dir"]).resolve()
    _setup_logging(shard_dir / "runner.log")

    cfg = OmegaConf.load(plan["resolved_config_path"])
    upstream_cfg = OmegaConf.load(plan["upstream_resolved_path"])
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))

    from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

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

    synthetic_tokens = set(scene_loader.synthetic_scenes.keys())
    orig_sensor_root = Path(str(upstream_cfg.original_sensor_path))
    synth_sensor_root = Path(str(upstream_cfg.synthetic_sensor_path))
    dataset_name = str(cfg.model.dataset_name)
    traj_num_poses = int(cfg.model.trajectory_sampling.num_poses)
    traj_interval = float(cfg.model.trajectory_sampling.interval_length)
    debug_dump_dir = Path(str(cfg.debug.dump_scene_data_dir)) if cfg.debug.dump_scene_data_dir else None
    debug_dump_limit = int(cfg.debug.dump_max_items)
    debug_counter = [0]

    pdm_results = []
    t0 = time.time()
    jobs = manifest["jobs"]
    LOGGER.info("Running shard %d with %d jobs", int(args.shard_index), len(jobs))
    for idx, job in enumerate(jobs, start=1):
        token = str(job["token"])
        stage_name = str(job["stage_name"])
        LOGGER.info("Shard %d job %d/%d: %s", int(args.shard_index), idx, len(jobs), job["job_id"])
        sensor_root = orig_sensor_root
        if stage_name == "stage_two":
            sensor_root = synth_sensor_root if token in synthetic_tokens else orig_sensor_root
        score_row = _evaluate_token(
            token=token,
            stage_name=stage_name,
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
        score_row["stage_name"] = stage_name
        score_row["job_id"] = str(job["job_id"])
        score_row["shard_index"] = int(args.shard_index)
        pdm_results.append(score_row)

    if pdm_results:
        partial_df = pd.concat(pdm_results, ignore_index=True)
    else:
        partial_df = pd.DataFrame(columns=["token", "valid", "stage_name", "job_id", "shard_index"])

    partial_pickle = shard_dir / "partial_rows.pkl"
    partial_csv = shard_dir / "partial_rows.csv"
    partial_df.to_pickle(partial_pickle)
    partial_df.to_csv(partial_csv, index=False)

    summary = {
        "shard_index": int(args.shard_index),
        "num_jobs": len(jobs),
        "num_rows": int(len(partial_df)),
        "num_successful_rows": int(partial_df["valid"].sum()) if "valid" in partial_df.columns else 0,
        "num_failed_rows": int(len(partial_df) - partial_df["valid"].sum()) if "valid" in partial_df.columns else int(len(partial_df)),
        "partial_pickle": str(partial_pickle),
        "partial_csv": str(partial_csv),
        "elapsed_sec": time.time() - t0,
    }
    with open(shard_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("Shard finished: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
