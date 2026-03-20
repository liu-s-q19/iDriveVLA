#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autolsqv2/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_sft8_epoch4.yaml}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PLAN_DIR="${PLAN_DIR:-$ROOT_DIR/logs/eval/navhard_recogdrive_vlm2b_8gpu_$(date -u +%F_%H-%M-%S)}"
CKPT_PATH="${CKPT_PATH:-}"
MAX_STAGE_ONE="${MAX_STAGE_ONE:-}"
MAX_STAGE_TWO="${MAX_STAGE_TWO:-}"

cd "$ROOT_DIR"

CMD=(
  bash scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh
)

if [[ -n "$PLAN_DIR" ]]; then
  CMD=("PLAN_DIR=$PLAN_DIR" "${CMD[@]}")
fi

CMD=("GPU_LIST=$GPU_LIST" "CONFIG_PATH=$CONFIG_PATH" "PYTHON_BIN=$PYTHON_BIN" "${CMD[@]}")

if [[ -n "$CKPT_PATH" ]]; then
  CMD=("CKPT_PATH=$CKPT_PATH" "${CMD[@]}")
fi

if [[ -n "$MAX_STAGE_ONE" ]]; then
  CMD=("MAX_STAGE_ONE=$MAX_STAGE_ONE" "${CMD[@]}")
fi

if [[ -n "$MAX_STAGE_TWO" ]]; then
  CMD=("MAX_STAGE_TWO=$MAX_STAGE_TWO" "${CMD[@]}")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Running ReCogDrive navhard eval with dedicated config"
printf '  %q' "${CMD[@]}"
printf '\n'

exec env "${CMD[@]}"
