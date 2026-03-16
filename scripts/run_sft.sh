#!/bin/bash
set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/data/liushiqi/navsim
export NAVSIM_DATA_ROOT=/data/dataset/navsim
export NUPLAN_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
export NAVSIM_EXP_ROOT=/data/dataset/navsim
export OPENSCENE_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

python tools/run_sft.py --config training/qwen2.5-vl-3B-navsimv2-mix-sft
