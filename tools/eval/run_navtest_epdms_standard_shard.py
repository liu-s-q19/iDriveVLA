#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.eval import run_navsimv2_epdms_standard as standard_mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one shard of navtest standard EPDMS evaluation.")
    parser.add_argument("--plan-dir", type=str, required=True, help="Shard plan directory created by prepare_navtest_epdms_shards.py")
    parser.add_argument("--shard-index", type=int, required=True, help="Shard index to execute")
    parser.add_argument("--dry-run", action="store_true", help="Print fingerprint only.")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir).resolve()
    plan = json.loads((plan_dir / "shard_plan.json").read_text(encoding="utf-8"))
    manifest_path = Path(plan["manifests_dir"]) / f"shard_{int(args.shard_index):02d}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partial_dir = Path(manifest["partial_dir"]).resolve()
    partial_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = standard_mod._resolve_config_path(manifest["config_path"])
    cfg = standard_mod._load_yaml(cfg_path)
    cfg["run"]["output_dir"] = str(partial_dir)
    cfg["run"]["log_path"] = str(partial_dir / "runner.log")
    standard_mod._validate_top_level_config(cfg)
    standard_mod._setup_logging(Path(cfg["run"]["log_path"]).resolve())
    standard_mod._check_runtime_env(str(cfg["run"].get("env_name", "")))
    standard_mod._apply_process_env(cfg)

    overrides = list(manifest["base_overrides"])
    overrides.append(f"train_test_split.scene_filter.tokens={json.dumps(manifest['tokens'], ensure_ascii=False)}")
    overrides.append(f"output_dir={partial_dir}")

    navsim_root = standard_mod._ensure_upstream_navsim(str(cfg["upstream"]["navsim_root"]))
    hydra_cfg = standard_mod._compose_upstream_cfg(cfg, overrides)
    schema_result = standard_mod._validate_metric_cache_schema(cfg, overrides)
    standard_mod.LOGGER.info("Metric cache schema check: %s", json.dumps(schema_result, ensure_ascii=False))
    fingerprint = standard_mod._build_fingerprint(hydra_cfg, overrides)
    standard_mod.LOGGER.info("Evaluation fingerprint: %s", json.dumps(fingerprint, ensure_ascii=False))
    fingerprint_path = partial_dir / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    standard_mod.LOGGER.info("Fingerprint saved: %s", fingerprint_path)

    code = standard_mod._run_evaluation(
        cfg=cfg,
        hydra_cfg=hydra_cfg,
        navsim_root=navsim_root,
        overrides=overrides,
        dry_run=args.dry_run,
        run_upstream_fn=standard_mod._run_upstream_run_pdm_score,
        run_autovla_fn=standard_mod._run_autovla_one_stage,
    )
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
