#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"

cd "${ROOT_DIR}"
exec env CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/eval/navsimv2_epdms_standard_autovla_recogdrive_vlm2b_sft8_epoch4.yaml}" \
  bash scripts/eval/run_navtest_epdms_standard_8gpu.sh "$@"
