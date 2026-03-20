#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autolsqv2/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml}"

NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/dataset/navsim/maps}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/dataset/navsim}"
NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

CKPT_PATH="${CKPT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_STAGE_ONE="${MAX_STAGE_ONE:-}"
MAX_STAGE_TWO="${MAX_STAGE_TWO:-}"

export NUPLAN_MAPS_ROOT
export OPENSCENE_DATA_ROOT
export NUPLAN_MAP_VERSION

cd "$ROOT_DIR"

CMD=(
  "$PYTHON_BIN" tools/eval/run_navhard_two_stage_autovla.py
  --config "$CONFIG_PATH"
)

if [[ -n "$CKPT_PATH" ]]; then
  CMD+=(--set "model.checkpoint_path=$CKPT_PATH")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--set "eval.output_dir=$OUTPUT_DIR")
fi

if [[ -n "$MAX_STAGE_ONE" ]]; then
  CMD+=(--set "eval.max_stage_one_scenarios=$MAX_STAGE_ONE")
fi

if [[ -n "$MAX_STAGE_TWO" ]]; then
  CMD+=(--set "eval.max_stage_two_scenarios=$MAX_STAGE_TWO")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Running single-machine navhard eval with current evaluator"
printf '  %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
