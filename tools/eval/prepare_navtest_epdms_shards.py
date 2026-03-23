#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tools.eval import navtest_epdms_sharded as shard_mod
from tools.eval import run_navsimv2_epdms_standard as standard_mod


def _remove_override(overrides, key: str):
    prefix = f"{key}="
    return [item for item in overrides if not str(item).startswith(prefix)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sharded navtest standard EPDMS evaluation manifests.")
    parser.add_argument("--config", type=str, required=True, help="Template config path.")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of shards / GPUs.")
    parser.add_argument("--plan-dir", type=str, default=None, help="Optional output directory for shard manifests.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Top-level OmegaConf dotlist override, e.g. model.checkpoint_path=/path/to/ckpt",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override forwarded to the standard wrapper.",
    )
    args = parser.parse_args()

    cfg_path = standard_mod._resolve_config_path(args.config)
    cfg = standard_mod._load_yaml(cfg_path)
    if args.set:
        merged_cfg = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(args.set))
        cfg = OmegaConf.to_container(merged_cfg, resolve=True)
        if not isinstance(cfg, dict):
            raise ValueError("--set produced non-mapping top-level config.")
    standard_mod._validate_top_level_config(cfg)
    standard_mod._apply_process_env(cfg)

    base_overrides = standard_mod._build_effective_overrides(cfg, args.override)
    base_overrides = _remove_override(base_overrides, "output_dir")
    base_overrides = _remove_override(base_overrides, "train_test_split.scene_filter.tokens")

    navsim_root = standard_mod._ensure_upstream_navsim(str(cfg["upstream"]["navsim_root"]))
    hydra_cfg = standard_mod._compose_upstream_cfg(cfg, base_overrides)

    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

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
    cache_tokens = [] if metric_cache_loader.tokens is None else list(metric_cache_loader.tokens)
    tokens = shard_mod.intersect_tokens(scene_tokens, cache_tokens)
    shards = shard_mod.shard_tokens(tokens, num_shards=int(args.num_shards))

    if args.plan_dir:
        plan_dir = Path(args.plan_dir).resolve()
    else:
        run_ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        plan_dir = Path(str(cfg["run"]["output_dir"])).resolve() / f"sharded_run_{run_ts}"
    manifests_dir = plan_dir / "manifests"
    partials_dir = plan_dir / "partials"
    merged_dir = plan_dir / "merged"
    resolved_cfg_path = plan_dir / "resolved_config.yaml"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")

    manifest_paths = []
    for shard_index, shard_tokens in enumerate(shards):
        shard_dir = partials_dir / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "shard_index": shard_index,
            "num_shards": int(args.num_shards),
            "tokens": shard_tokens,
            "partial_dir": str(shard_dir),
            "config_path": str(resolved_cfg_path),
            "base_overrides": base_overrides,
            "navsim_root": str(navsim_root),
        }
        manifest_path = manifests_dir / f"shard_{shard_index:02d}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_paths.append(str(manifest_path))

    plan = {
        "plan_dir": str(plan_dir),
        "manifests_dir": str(manifests_dir),
        "partials_dir": str(partials_dir),
        "merged_dir": str(merged_dir),
        "num_shards": int(args.num_shards),
        "config_path": str(cfg_path),
        "resolved_config_path": str(resolved_cfg_path),
        "set_overrides": list(args.set),
        "base_overrides": base_overrides,
        "num_tokens": len(tokens),
        "manifest_paths": manifest_paths,
    }
    (plan_dir / "shard_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False))


if __name__ == "__main__":
    main()
