#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_TAG="${RUN_TAG:-navtest_pdms_sft_8gpu_det_$(date -u +%F_%H-%M-%S)}"
SESSION_NAME="${SESSION_NAME:-${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/eval}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
LAUNCHER_LOG="${RUN_ROOT}/launcher.log"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/hezeyu/AutoVLA/runs/sft/2026-03-06_12-12-48/epoch=4-loss=1.0341.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/eval/qwen2.5-vl-3B-navsim-pdms-nocot-det.yaml}"
SENSOR_DATA_PATH="${SENSOR_DATA_PATH:-/data/dataset/navsim/sensor_blobs/test}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data/dataset/navsim/metric_cache/navtest}"
JSON_DATA_PATH="${JSON_DATA_PATH:-/data/dataset/navsim/preprocessed/navtest_nocot}"
SCENE_FILTER_CONFIG="${SCENE_FILTER_CONFIG:-${REPO_ROOT}/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml}"
SEED="${SEED:-17}"
NUM_SHARDS="${NUM_SHARDS:-8}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-8}"
MAX_USED_MEMORY_MB="${MAX_USED_MEMORY_MB:-2048}"
POLL_SECONDS="${POLL_SECONDS:-30}"
USE_LORA="${USE_LORA:-false}"

mkdir -p "${RUN_ROOT}"

echo "[$(date -u '+%F %T UTC')] waiter_start session=${SESSION_NAME} run_root=${RUN_ROOT}" | tee -a "${LAUNCHER_LOG}"

while true; do
  READY_GPU_LIST="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v max="${MAX_USED_MEMORY_MB}" '
          {
            gsub(/ /, "", $1);
            gsub(/ /, "", $2);
            if (($2 + 0) <= max) print $1;
          }
        ' \
      | head -n "${REQUIRED_GPU_COUNT}" \
      | paste -sd, -
  )"
  if [ -n "${READY_GPU_LIST}" ]; then
    READY_GPU_COUNT="$(awk -F',' '{print NF}' <<< "${READY_GPU_LIST}")"
  else
    READY_GPU_COUNT=0
  fi

  if [ "${READY_GPU_COUNT}" -ge "${REQUIRED_GPU_COUNT}" ]; then
    echo "[$(date -u '+%F %T UTC')] ready_gpus=${READY_GPU_LIST}" | tee -a "${LAUNCHER_LOG}"
    break
  fi

  echo "[$(date -u '+%F %T UTC')] waiting ready=${READY_GPU_COUNT}/${REQUIRED_GPU_COUNT} max_used_memory_mb=${MAX_USED_MEMORY_MB}" | tee -a "${LAUNCHER_LOG}"
  sleep "${POLL_SECONDS}"
done

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
LAUNCH_CMD=(
  /data/miniconda/envs/autovla_codeclean/bin/python
  "${REPO_ROOT}/tools/eval/navsim_eval_8gpu_launcher.py"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --config-path "${CONFIG_PATH}"
  --sensor-data-path "${SENSOR_DATA_PATH}"
  --metric-cache-path "${METRIC_CACHE_PATH}"
  --json-data-path "${JSON_DATA_PATH}"
  --scene-filter-config "${SCENE_FILTER_CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --session-name "${SESSION_NAME}"
  --run-tag "${RUN_TAG}"
  --seed "${SEED}"
  --num-shards "${NUM_SHARDS}"
  --required-gpu-count "${REQUIRED_GPU_COUNT}"
  --max-used-memory-mb "${MAX_USED_MEMORY_MB}"
  --poll-seconds "${POLL_SECONDS}"
  --gpu-indices "${READY_GPU_LIST}"
)
if [ "${USE_LORA}" = "true" ]; then
  LAUNCH_CMD+=(--use-lora)
fi

"${LAUNCH_CMD[@]}"
