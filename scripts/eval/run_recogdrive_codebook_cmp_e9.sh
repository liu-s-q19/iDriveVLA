#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autolsqv2/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

CKPT_PATH="${CKPT_PATH:-/data/liushiqi/AutoVLA/runs/sft/navsimv2_recogdrive_sft_epoch10_speedupab_mainline_2026-03-23_12-29-45/epoch=9-loss=1.1338.ckpt}"
OLD_MODEL_CONFIG_PATH="${OLD_MODEL_CONFIG_PATH:-config/training/recogdrive-vlm-2b-navsimv2-mix-sft-local8gpu.yaml}"
NEW_MODEL_CONFIG_PATH="${NEW_MODEL_CONFIG_PATH:-config/training/recogdrive-vlm-2b-navsimv2-mix-sft-local8gpu-epoch10-speedupab-noprobe-20260323.yaml}"

SMOKE_MAX_STAGE_ONE="${SMOKE_MAX_STAGE_ONE:-40}"
SMOKE_MAX_STAGE_TWO="${SMOKE_MAX_STAGE_TWO:-40}"
FULL_CONFIG_TEMPLATE="${FULL_CONFIG_TEMPLATE:-config/eval/navsimv2_epdms_standard_autovla_recogdrive_vlm2b_sft8_epoch4.yaml}"
PLAN_DIR="${PLAN_DIR:-$ROOT_DIR/logs/eval/codebook_cmp_e9_$(date -u +%F_%H-%M-%S)}"

cd "$ROOT_DIR"

"$PYTHON_BIN" tools/eval/run_codebook_eval_compare.py \
  --root-dir "$ROOT_DIR" \
  --plan-dir "$PLAN_DIR" \
  --python-bin "$PYTHON_BIN" \
  --gpu-list "$GPU_LIST" \
  --ckpt-path "$CKPT_PATH" \
  --old-model-config-path "$OLD_MODEL_CONFIG_PATH" \
  --new-model-config-path "$NEW_MODEL_CONFIG_PATH" \
  --smoke-max-stage-one "$SMOKE_MAX_STAGE_ONE" \
  --smoke-max-stage-two "$SMOKE_MAX_STAGE_TWO" \
  --full-config-template "$FULL_CONFIG_TEMPLATE" \
  --run-smoke \
  --run-full \
  --run-final-rerun \
  "$@"

echo "compare_report.json: $PLAN_DIR/compare_report.json"
echo "compare_report.md: $PLAN_DIR/compare_report.md"

# tmux template (host: 10.199.7.32)
# cd /data/liushiqi/AutoVLA
# tmux new-session -d -s codebook_cmp_e9 '
# cd /data/liushiqi/AutoVLA && \
# bash scripts/eval/run_recogdrive_codebook_cmp_e9.sh 2>&1 | tee logs/eval/codebook_cmp_e9_tmux.log
# '
