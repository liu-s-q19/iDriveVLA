#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autolsqv2/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/eval/navhard_two_stage_autovla_qwen_sft8_retry3_epoch4.yaml}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PLAN_DIR="${PLAN_DIR:-}"
CKPT_PATH="${CKPT_PATH:-}"
MAX_STAGE_ONE="${MAX_STAGE_ONE:-}"
MAX_STAGE_TWO="${MAX_STAGE_TWO:-}"

NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/dataset/navsim/maps}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/dataset/navsim}"
NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

export NUPLAN_MAPS_ROOT
export OPENSCENE_DATA_ROOT
export NUPLAN_MAP_VERSION

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=$NUM_SHARDS" >&2
  exit 1
fi

cd "$ROOT_DIR"

PREP_CMD=(
  "$PYTHON_BIN" tools/eval/prepare_navhard_two_stage_shards.py
  --config "$CONFIG_PATH"
  --num-shards "$NUM_SHARDS"
)

if [[ -n "$PLAN_DIR" ]]; then
  PREP_CMD+=(--plan-dir "$PLAN_DIR")
fi

if [[ -n "$CKPT_PATH" ]]; then
  PREP_CMD+=(--set "model.checkpoint_path=$CKPT_PATH")
fi

if [[ -n "$MAX_STAGE_ONE" ]]; then
  PREP_CMD+=(--set "eval.max_stage_one_scenarios=$MAX_STAGE_ONE")
fi

if [[ -n "$MAX_STAGE_TWO" ]]; then
  PREP_CMD+=(--set "eval.max_stage_two_scenarios=$MAX_STAGE_TWO")
fi

if [[ "$#" -gt 0 ]]; then
  PREP_CMD+=("$@")
fi

echo "Preparing current 8GPU navhard eval plan"
printf '  %q' "${PREP_CMD[@]}"
printf '\n'

PREP_OUTPUT="$("${PREP_CMD[@]}")"
echo "$PREP_OUTPUT"

PLAN_DIR_RESOLVED="$($PYTHON_BIN - <<'PY' "$PREP_OUTPUT"
import json
import sys

print(json.loads(sys.argv[1])["plan_dir"])
PY
)"

echo "Using plan dir: $PLAN_DIR_RESOLVED"

PIDS=()
for (( shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++ )); do
  gpu="${GPUS[$shard_idx]}"
  shard_dir="$PLAN_DIR_RESOLVED/partials/shard_$(printf '%02d' "$shard_idx")"
  mkdir -p "$shard_dir"
  log_path="$shard_dir/launcher_stdout.log"
  echo "Launching shard $shard_idx on GPU $gpu -> $log_path"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/eval/run_navhard_two_stage_autovla_shard.py \
    --plan-dir "$PLAN_DIR_RESOLVED" \
    --shard-index "$shard_idx" \
    >"$log_path" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

if [[ "$STATUS" -ne 0 ]]; then
  echo "At least one shard failed. Inspect $PLAN_DIR_RESOLVED/partials/shard_*/launcher_stdout.log" >&2
  exit "$STATUS"
fi

"$PYTHON_BIN" tools/eval/merge_navhard_two_stage_shards.py --plan-dir "$PLAN_DIR_RESOLVED"

echo "Merged results: $PLAN_DIR_RESOLVED/merged/summary.json"
