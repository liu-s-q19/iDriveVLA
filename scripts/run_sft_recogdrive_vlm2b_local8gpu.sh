#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"

cd "${ROOT_DIR}"
exec env SFT_CONFIG="${SFT_CONFIG:-training/recogdrive-vlm-2b-navsimv2-mix-sft}" bash scripts/run_sft.sh "$@"
