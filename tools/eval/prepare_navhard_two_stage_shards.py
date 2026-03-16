#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tools.eval.navhard_two_stage_sharded import (
    build_eval_jobs,
    intersect_tokens,
    select_stage_two_tokens,
    shard_eval_jobs,
)


def _ensure_upstream_navsim(navsim_root: str) -> None:
    root = str(Path(navsim_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    stale = [k for k in list(sys.modules.keys()) if k == "navsim" or k.startswith("navsim.")]
    for key in stale:
        del sys.modules[key]


def _load_config(config_path: Path, dotlist: List[str]):
    cfg = OmegaConf.load(config_path)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sharded navhard two-stage AutoVLA evaluation manifests.")
    parser.add_argument("--config", type=str, required=True, help="Path to top-level evaluation config.")
    parser.add_argument("--set", action="append", default=[], help="OmegaConf dotlist override.")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of shards / GPUs.")
    parser.add_argument("--plan-dir", type=str, default=None, help="Optional output directory for shard manifests.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path, args.set)
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))

    with initialize_config_dir(config_dir=str(Path(str(cfg.upstream.config_dir)).resolve()), version_base=None):
        upstream_cfg = compose(
            config_name=str(cfg.upstream.config_name),
            overrides=[str(x) for x in cfg.upstream.overrides],
        )

    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

    sensor_config = SensorConfig.build_no_sensors()
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

    stage1_tokens = intersect_tokens(scene_loader.tokens_stage_one, metric_cache_loader.tokens)
    if cfg.eval.max_stage_one_scenarios is not None:
        stage1_tokens = stage1_tokens[: int(cfg.eval.max_stage_one_scenarios)]

    traffic_mode = str(cfg.traffic_agents.mode).lower()
    stage2_tokens = select_stage_two_tokens(
        traffic_mode=traffic_mode,
        available_synthetic_tokens=list(scene_loader.synthetic_scenes.keys()),
        reactive_initial_tokens=scene_filter_cfg.get("reactive_synthetic_initial_tokens"),
        non_reactive_initial_tokens=scene_filter_cfg.get("non_reactive_synthetic_initial_tokens"),
        metric_cache_tokens=metric_cache_loader.tokens,
    )
    if cfg.eval.max_stage_two_scenarios is not None:
        stage2_tokens = stage2_tokens[: int(cfg.eval.max_stage_two_scenarios)]

    jobs = build_eval_jobs(stage1_tokens=stage1_tokens, stage2_tokens=stage2_tokens)
    shards = shard_eval_jobs(jobs, num_shards=int(args.num_shards))

    if args.plan_dir:
        plan_dir = Path(args.plan_dir).resolve()
    else:
        run_ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        plan_dir = Path(str(cfg.eval.output_dir)).resolve() / f"sharded_run_{run_ts}"
    manifests_dir = plan_dir / "manifests"
    partials_dir = plan_dir / "partials"
    merged_dir = plan_dir / "merged"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_path = plan_dir / "resolved_config.yaml"
    upstream_cfg_path = plan_dir / "upstream_resolved.yaml"
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))
    with open(upstream_cfg_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(upstream_cfg))

    manifest_paths = []
    for shard_index, shard_jobs in enumerate(shards):
        shard_dir = partials_dir / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "shard_index": shard_index,
            "num_shards": int(args.num_shards),
            "jobs": shard_jobs,
            "partial_dir": str(shard_dir),
        }
        manifest_path = manifests_dir / f"shard_{shard_index:02d}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifest_paths.append(str(manifest_path))

    plan = {
        "plan_dir": str(plan_dir),
        "resolved_config_path": str(resolved_cfg_path),
        "upstream_resolved_path": str(upstream_cfg_path),
        "manifests_dir": str(manifests_dir),
        "partials_dir": str(partials_dir),
        "merged_dir": str(merged_dir),
        "num_shards": int(args.num_shards),
        "traffic_mode": traffic_mode,
        "stage1_tokens": [str(x) for x in stage1_tokens],
        "stage2_tokens": [str(x) for x in stage2_tokens],
        "scene_tokens": [str(x) for x in scene_loader.tokens],
        "jobs": jobs,
        "manifest_paths": manifest_paths,
    }
    plan_path = plan_dir / "shard_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "plan_dir": str(plan_dir),
        "plan_path": str(plan_path),
        "num_shards": int(args.num_shards),
        "num_stage1_tokens": len(stage1_tokens),
        "num_stage2_tokens": len(stage2_tokens),
        "num_jobs": len(jobs),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
