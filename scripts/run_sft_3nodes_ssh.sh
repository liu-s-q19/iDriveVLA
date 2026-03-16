#!/usr/bin/env bash
set -euo pipefail

# 3-node SSH launcher for AutoVLA SFT (NavSim v2)
# - Launch worker nodes (rank 1..N-1) via SSH + nohup
# - Launch master rank0 locally (foreground or background)

# ----------------- SSH / cluster -----------------
SSH_PORT="${SSH_PORT:-2289}"
SSH_USER="${SSH_USER:-root}"

NODES=(
  "10.199.7.32"  # rank 0
  "10.199.7.33"  # rank 1
  "10.199.7.190" # rank 2
)

NNODES="${NNODES:-3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-10.199.7.32}"
MASTER_PORT="${MASTER_PORT:-29637}"

RUN_MASTER_BG="${RUN_MASTER_BG:-1}"
AUTO_CHECK="${AUTO_CHECK:-1}"
CHECK_TIMEOUT_SEC="${CHECK_TIMEOUT_SEC:-240}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-10}"

# ----------------- Project / runtime -----------------
PROJECT_ROOT="${PROJECT_ROOT:-/data/liushiqi/AutoVLA}"
CONDA_SH="${CONDA_SH:-/data/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-autovla_codeclean}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autovla_codeclean/bin/python}"
SFT_CONFIG="${SFT_CONFIG:-training/qwen2.5-vl-3B-navsimv2-mix-sft}"

NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/data/liushiqi/navsim}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/data/dataset/navsim}"
NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/data/dataset/navsim}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/dataset/navsim/maps}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/dataset/navsim}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/dataset/navsim}"
NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

# ----------------- Distributed network mode -----------------
# Modes:
# - stable_tcp: prioritize startup stability (legacy default)
# - fast_ib:    prioritize multi-node throughput on IB clusters
NCCL_NET_MODE="${NCCL_NET_MODE:-stable_tcp}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond4}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"

case "${NCCL_NET_MODE}" in
  stable_tcp)
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    ;;
  fast_ib)
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_0}"
    ;;
  *)
    echo "[ERROR] Unknown NCCL_NET_MODE=${NCCL_NET_MODE} (expected stable_tcp|fast_ib)"
    exit 1
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/logs/train}"
TS="${TS:-$(date +%F_%H-%M-%S)}"
RUN_TAG="sft_navsimv2_3nodes_${TS}"

SSH_OPTS=(
  -p "${SSH_PORT}"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ConnectTimeout=8
)

if (( NNODES > ${#NODES[@]} )); then
  echo "[ERROR] NNODES=${NNODES} exceeds configured nodes count=${#NODES[@]}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "🚀 AutoVLA SFT 3-node launcher"
echo "Nodes:        ${NODES[*]}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "NNODES:       ${NNODES}"
echo "GPUs/node:    ${GPUS_PER_NODE}"
echo "Config:       ${SFT_CONFIG}"
echo "Conda env:    ${CONDA_ENV}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Run tag:      ${RUN_TAG}"
echo "Master BG:    ${RUN_MASTER_BG}"
echo "Auto check:   ${AUTO_CHECK}"
echo "NCCL mode:    ${NCCL_NET_MODE}"
echo "NCCL IFACE:   ${NCCL_SOCKET_IFNAME} (IB disabled=${NCCL_IB_DISABLE}, HCA=${NCCL_IB_HCA:-n/a})"
echo "=================================================="

if [[ "${NCCL_NET_MODE}" == "fast_ib" ]]; then
  echo "[CHECK] Probing IB device (${NCCL_IB_HCA}) on all nodes ..."
  for node_ip in "${NODES[@]}"; do
    if ! ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node_ip}" \
      "test -d /sys/class/infiniband/${NCCL_IB_HCA}" >/dev/null 2>&1; then
      echo "[WARN] ${node_ip}: /sys/class/infiniband/${NCCL_IB_HCA} not found (will still try launching)."
    else
      echo "[OK]   ${node_ip}: found /sys/class/infiniband/${NCCL_IB_HCA}"
    fi
  done
fi

# ----------------- Step 1: launch workers -----------------
for node_rank in $(seq 1 $((NNODES - 1))); do
  node_ip="${NODES[$node_rank]}"
  echo "==> launch worker rank=${node_rank} node=${node_ip}"

  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node_ip}" \
    "MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}' NNODES='${NNODES}' GPUS_PER_NODE='${GPUS_PER_NODE}' NODE_RANK='${node_rank}' \
     PROJECT_ROOT='${PROJECT_ROOT}' CONDA_SH='${CONDA_SH}' CONDA_ENV='${CONDA_ENV}' PYTHON_BIN='${PYTHON_BIN}' \
     SFT_CONFIG='${SFT_CONFIG}' OUTPUT_DIR='${OUTPUT_DIR}' RUN_TAG='${RUN_TAG}' TS='${TS}' \
     NAVSIM_DEVKIT_ROOT='${NAVSIM_DEVKIT_ROOT}' NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' NUPLAN_DATA_ROOT='${NUPLAN_DATA_ROOT}' \
     NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' \
     NUPLAN_MAP_VERSION='${NUPLAN_MAP_VERSION}' \
     NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}' NCCL_IB_DISABLE='${NCCL_IB_DISABLE}' \
     NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}' NCCL_P2P_DISABLE='${NCCL_P2P_DISABLE}' NCCL_SHM_DISABLE='${NCCL_SHM_DISABLE}' \
     NCCL_CROSS_NIC='${NCCL_CROSS_NIC}' NCCL_IB_HCA='${NCCL_IB_HCA}' \
     NCCL_ASYNC_ERROR_HANDLING='${NCCL_ASYNC_ERROR_HANDLING}' TORCH_NCCL_ASYNC_ERROR_HANDLING='${TORCH_NCCL_ASYNC_ERROR_HANDLING}' \
     NCCL_DEBUG='${NCCL_DEBUG}' bash -s" <<'REMOTE'
set -euo pipefail

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

export NAVSIM_DEVKIT_ROOT NAVSIM_DATA_ROOT NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAP_VERSION
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PL_NUM_NODES="${NNODES}"
export SFT_RUN_TS="${TS}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG
export NCCL_SOCKET_FAMILY NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_IB_HCA

LOG_FILE="${OUTPUT_DIR}/${RUN_TAG}_rank${NODE_RANK}.log"
PID_FILE="${OUTPUT_DIR}/${RUN_TAG}_rank${NODE_RANK}.pid"

nohup torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  tools/run_sft.py \
  --config "${SFT_CONFIG}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[LAUNCHED] worker rank=${NODE_RANK} pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
REMOTE
done

# ----------------- Step 2: launch master -----------------
MASTER_LOG="${OUTPUT_DIR}/${RUN_TAG}_rank0.log"
MASTER_PID="${OUTPUT_DIR}/${RUN_TAG}_rank0.pid"

MASTER_CMD=(torchrun
  --nnodes="${NNODES}"
  --node_rank=0
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  --nproc_per_node="${GPUS_PER_NODE}"
  tools/run_sft.py
  --config "${SFT_CONFIG}")

if [[ "${RUN_MASTER_BG}" == "1" ]]; then
  master_node="${NODES[0]}"
  echo "==> launch master rank=0 node=${master_node} via ssh"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${master_node}" \
    "MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}' NNODES='${NNODES}' GPUS_PER_NODE='${GPUS_PER_NODE}' NODE_RANK='0' \
     PROJECT_ROOT='${PROJECT_ROOT}' CONDA_SH='${CONDA_SH}' CONDA_ENV='${CONDA_ENV}' PYTHON_BIN='${PYTHON_BIN}' \
     SFT_CONFIG='${SFT_CONFIG}' OUTPUT_DIR='${OUTPUT_DIR}' RUN_TAG='${RUN_TAG}' TS='${TS}' \
     NAVSIM_DEVKIT_ROOT='${NAVSIM_DEVKIT_ROOT}' NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' NUPLAN_DATA_ROOT='${NUPLAN_DATA_ROOT}' \
     NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' \
     NUPLAN_MAP_VERSION='${NUPLAN_MAP_VERSION}' \
     NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}' NCCL_IB_DISABLE='${NCCL_IB_DISABLE}' \
     NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}' NCCL_P2P_DISABLE='${NCCL_P2P_DISABLE}' NCCL_SHM_DISABLE='${NCCL_SHM_DISABLE}' \
     NCCL_CROSS_NIC='${NCCL_CROSS_NIC}' NCCL_IB_HCA='${NCCL_IB_HCA}' \
     NCCL_ASYNC_ERROR_HANDLING='${NCCL_ASYNC_ERROR_HANDLING}' TORCH_NCCL_ASYNC_ERROR_HANDLING='${TORCH_NCCL_ASYNC_ERROR_HANDLING}' \
     NCCL_DEBUG='${NCCL_DEBUG}' bash -s" <<'REMOTE'
set -euo pipefail

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

export NAVSIM_DEVKIT_ROOT NAVSIM_DATA_ROOT NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAP_VERSION
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PL_NUM_NODES="${NNODES}"
export SFT_RUN_TS="${TS}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG
export NCCL_SOCKET_FAMILY NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_IB_HCA

LOG_FILE="${OUTPUT_DIR}/${RUN_TAG}_rank${NODE_RANK}.log"
PID_FILE="${OUTPUT_DIR}/${RUN_TAG}_rank${NODE_RANK}.pid"

nohup torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  tools/run_sft.py \
  --config "${SFT_CONFIG}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[LAUNCHED] master rank=${NODE_RANK} pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
REMOTE
else
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  cd "${PROJECT_ROOT}"

  export NAVSIM_DEVKIT_ROOT NAVSIM_DATA_ROOT NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAP_VERSION
  export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
  export PL_NUM_NODES="${NNODES}"
  export SFT_RUN_TS="${TS}"
  export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG
  export NCCL_SOCKET_FAMILY NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_CROSS_NIC NCCL_IB_HCA

  "${MASTER_CMD[@]}" | tee "${MASTER_LOG}"
fi

if [[ "${RUN_MASTER_BG}" == "1" && "${AUTO_CHECK}" == "1" ]]; then
  echo "[CHECK] Waiting for rank logs / pids ..."
  deadline=$((SECONDS + CHECK_TIMEOUT_SEC))
  ok=0
  while (( SECONDS < deadline )); do
    ok=1
    for r in $(seq 0 $((NNODES - 1))); do
      pf="${OUTPUT_DIR}/${RUN_TAG}_rank${r}.pid"
      lf="${OUTPUT_DIR}/${RUN_TAG}_rank${r}.log"
      if [[ ! -f "${pf}" ]] || [[ ! -s "${pf}" ]]; then
        ok=0
        break
      fi
      pid="$(cat "${pf}" 2>/dev/null || true)"
      if [[ -z "${pid}" ]]; then
        ok=0
        break
      fi
      if [[ "${r}" == "0" ]]; then
        if ! kill -0 "${pid}" 2>/dev/null; then
          ok=0
          break
        fi
      fi
      if [[ ! -f "${lf}" ]]; then
        ok=0
        break
      fi
    done
    if [[ "${ok}" == "1" ]]; then
      break
    fi
    sleep "${CHECK_INTERVAL_SEC}"
  done

  if [[ "${ok}" == "1" ]]; then
    echo "[CHECK][OK] Launcher pids/logs are ready."
    for r in $(seq 0 $((NNODES - 1))); do
      echo "--- tail rank${r} ---"
      tail -n 20 "${OUTPUT_DIR}/${RUN_TAG}_rank${r}.log" || true
    done
  else
    echo "[CHECK][WARN] Timeout while waiting rank logs/pids."
  fi
fi

echo "[DONE] launch finished"
echo "Logs prefix: ${OUTPUT_DIR}/${RUN_TAG}_rank*.log"
