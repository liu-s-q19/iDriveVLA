#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-autolsqv2}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/${CONDA_ENV}/bin/python}"
SFT_CONFIG="${SFT_CONFIG:-training/qwen2.5-vl-3B-navsimv2-mix-sft}"

export NAVSIM_DEVKIT_ROOT=/data/liushiqi/navsim
export NAVSIM_DATA_ROOT=/data/dataset/navsim
export NUPLAN_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
export NAVSIM_EXP_ROOT=/data/dataset/navsim
export OPENSCENE_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" tools/run_sft.py --config "${SFT_CONFIG}"
