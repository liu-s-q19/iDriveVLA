#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"

cd "${ROOT_DIR}"
exec env SFT_CONFIG="${SFT_CONFIG:-training/qwen2.5-vl-3B-navsimv2-mix-sft}" bash scripts/run_sft.sh "$@"
