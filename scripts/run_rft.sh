#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-autolsqv2}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/${CONDA_ENV}/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RFT_CONFIG="${RFT_CONFIG:-training/qwen2.5-vl-3B-navsimv2-grpo-cot-fast-rft20260312e4-answer-format}"
# Avoid distributed run_id sync-file races in tools/run_rft.py.
# Every worker inherits the same environment from the launcher process.
RFT_RUN_ID="${RFT_RUN_ID:-grpo_$(date -u +%F_%H-%M-%S)_$RANDOM}"

export NAVSIM_DEVKIT_ROOT=/data/liushiqi/navsim
export NAVSIM_DATA_ROOT=/data/dataset/navsim
export NUPLAN_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
export NAVSIM_EXP_ROOT=/data/dataset/navsim
export OPENSCENE_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export RFT_RUN_ID

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON_BIN}" tools/run_rft.py --config "${RFT_CONFIG}"
