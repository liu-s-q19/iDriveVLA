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

from tools.eval.navhard_two_stage_sharded import (
    collect_all_mappings,
    finalize_merged_results,
    validate_job_coverage,
)
from tools.eval.run_navhard_two_stage_autovla import _ensure_upstream_navsim


LOGGER = logging.getLogger("merge_navhard_two_stage_shards")


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
    parser = argparse.ArgumentParser(description="Merge sharded navhard two-stage AutoVLA evaluation outputs.")
    parser.add_argument("--plan-dir", type=str, required=True, help="Shard plan directory created by prepare_navhard_two_stage_shards.py")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir).resolve()
    plan = json.loads((plan_dir / "shard_plan.json").read_text(encoding="utf-8"))
    merged_dir = Path(plan["merged_dir"]).resolve()
    _setup_logging(merged_dir / "merge.log")

    cfg = OmegaConf.load(plan["resolved_config_path"])
    upstream_cfg = OmegaConf.load(plan["upstream_resolved_path"])
    _ensure_upstream_navsim(str(cfg.upstream.navsim_root))

    from navsim.common.dataclasses import PDMResults
    from navsim.common.enums import SceneFrameType

    partial_frames = []
    t0 = time.time()
    for shard_index in range(int(plan["num_shards"])):
        partial_path = Path(plan["partials_dir"]) / f"shard_{shard_index:02d}" / "partial_rows.pkl"
        if not partial_path.exists():
            raise FileNotFoundError(f"Missing shard partial pickle: {partial_path}")
        partial_frames.append(pd.read_pickle(partial_path))

    combined_rows = pd.concat(partial_frames, ignore_index=True) if partial_frames else pd.DataFrame()
    validate_job_coverage(expected_jobs=plan["jobs"], combined_rows=combined_rows)

    all_mappings = collect_all_mappings(
        raw_mapping=upstream_cfg.train_test_split.reactive_all_mapping,
        scene_tokens=plan["scene_tokens"],
    )

    final_df, summary = finalize_merged_results(
        combined_rows=combined_rows,
        all_mappings=all_mappings,
        proposal_sampling=instantiate(upstream_cfg.simulator.proposal_sampling),
        scene_frame_type_original=SceneFrameType.ORIGINAL,
        scene_frame_type_synthetic=SceneFrameType.SYNTHETIC,
        pdm_result_field_names=[score.name for score in PDMResults.__dataclass_fields__.values()],
        output_dir=merged_dir,
        write_artifacts=True,
        logger=LOGGER,
    )
    del final_df

    summary["num_shards"] = int(plan["num_shards"])
    summary["num_jobs"] = len(plan["jobs"])
    summary["elapsed_sec"] = time.time() - t0
    with open(merged_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("Merge finished: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
