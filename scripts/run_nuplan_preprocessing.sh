#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

export PYTHONPATH=/data/liushiqi/AutoVLA/navsim:$PYTHONPATH
export PYTHONPATH=/data/liushiqi/AutoVLA:$PYTHONPATH

export NAVSIM_DEVKIT_ROOT=/data/liushiqi/AutoVLA/navsim
export NAVSIM_DATA_ROOT=/data/dataset/navsim
export NUPLAN_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps
export NAVSIM_EXP_ROOT=/data/dataset/navsim/navsim_logs
export OPENSCENE_DATA_ROOT=/data/dataset/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

INCLUDE_COT=false
CONFIG="dataset/qwen2.5-vl-72B-trainval"
OUTPUT_DIR="/data/dataset/navsim/preprocessed/navtrainval_nocot"

mkdir -p "$OUTPUT_DIR"

CONFIG_FILE="/data/liushiqi/AutoVLA/config/${CONFIG}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

# RISK: defaulting to navsim env may miss AutoVLA training deps (e.g. qwen_vl_utils);
# override PYTHON_BIN to autovla_codeclean python when running full preprocessing.
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/navsim/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Python binary not found or not executable: $PYTHON_BIN"
    exit 1
fi

DATASET_PATH=$(CONFIG_FILE="$CONFIG_FILE" "$PYTHON_BIN" - <<'PY'
import yaml
import os
cfg = yaml.safe_load(open(os.environ['CONFIG_FILE'], 'r'))
print(cfg['dataset_path'])
PY
)

SENSOR_BLOBS_PATH="${DATASET_PATH/placeholder/sensor_blobs}"
NAVSIM_LOGS_PATH="${DATASET_PATH/placeholder/navsim_logs}"

echo "[PathCheck] dataset_path=$DATASET_PATH"
echo "[PathCheck] navsim_logs_path=$NAVSIM_LOGS_PATH"
echo "[PathCheck] sensor_blobs_path=$SENSOR_BLOBS_PATH"

if [ ! -d "$NAVSIM_LOGS_PATH" ]; then
    echo "[ERROR] navsim logs path not found: $NAVSIM_LOGS_PATH"
    exit 1
fi

if [ ! -d "$SENSOR_BLOBS_PATH" ]; then
    echo "[ERROR] sensor blobs path not found: $SENSOR_BLOBS_PATH"
    exit 1
fi

if [[ "$SENSOR_BLOBS_PATH" == *"trainval_ini"* ]]; then
    echo "[ERROR] sensor_blobs_path points to trainval_ini: $SENSOR_BLOBS_PATH"
    echo "[ERROR] Please switch dataset_path to use trainval (not trainval_ini)."
    exit 1
fi

if [ "$INCLUDE_COT" = true ]; then
    echo "Preprocessing with Chain-of-Thought (CoT)..."
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" tools/preprocessing/cot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    "$PYTHON_BIN" tools/preprocessing/nocot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
fi
