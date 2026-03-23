#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
export RFT_CONFIG="${RFT_CONFIG:-training/recogdrive-vlm-2b-navsimv2-grpo-cot-fast-rft20260321-reg-default-format}"

cd "$ROOT_DIR"
exec bash scripts/run_rft.sh

