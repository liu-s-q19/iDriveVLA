#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autolsqv2/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_sft8_epoch4.yaml}"
CKPT_PATH="${CKPT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_STAGE_ONE="${MAX_STAGE_ONE:-}"
MAX_STAGE_TWO="${MAX_STAGE_TWO:-}"

cd "$ROOT_DIR"

CMD=(
  bash scripts/eval/run_navhard_two_stage_autovla_single.sh
)

CMD=("CONFIG_PATH=$CONFIG_PATH" "PYTHON_BIN=$PYTHON_BIN" "${CMD[@]}")

if [[ -n "$CKPT_PATH" ]]; then
  CMD=("CKPT_PATH=$CKPT_PATH" "${CMD[@]}")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD=("OUTPUT_DIR=$OUTPUT_DIR" "${CMD[@]}")
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

echo "Running single-machine ReCogDrive navhard eval with dedicated config"
printf '  %q' "${CMD[@]}"
printf '\n'

exec env "${CMD[@]}"
