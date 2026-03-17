#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.eval import navtest_epdms_sharded as shard_mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded navtest standard EPDMS outputs.")
    parser.add_argument("--plan-dir", type=str, required=True, help="Shard plan directory created by prepare_navtest_epdms_shards.py")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir).resolve()
    plan = json.loads((plan_dir / "shard_plan.json").read_text(encoding="utf-8"))
    partial_dirs = [Path(plan["partials_dir"]) / f"shard_{idx:02d}" for idx in range(int(plan["num_shards"]))]
    merged_dir = Path(plan["merged_dir"]).resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_df, summary = shard_mod.summarize_merged_results(partial_dirs)
    (merged_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not merged_df.empty:
        merged_df.to_csv(merged_dir / "merged.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
