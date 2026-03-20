#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/liushiqi/AutoVLA}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda/envs/autolsqv2/bin/python}"
SSH_HOST="${SSH_HOST:-10.199.7.190}"
SSH_PORT="${SSH_PORT:-2289}"
SSH_USER="${SSH_USER:-root}"
QUEUE_TAG="${QUEUE_TAG:-navhard_eval_queue_ip190_$(date -u +%Y-%m-%d_%H-%M-%S)}"
QUEUE_DIR="${QUEUE_DIR:-$ROOT_DIR/logs/eval/$QUEUE_TAG}"
SMOKE_STAGE_ONE="${SMOKE_STAGE_ONE:-1}"
SMOKE_STAGE_TWO="${SMOKE_STAGE_TWO:-1}"
TASK_IDS_CSV="${TASK_IDS_CSV:-}"
EXECUTION_MODE="${EXECUTION_MODE:-8gpu}"
POLL_MISSING="${POLL_MISSING:-1}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-600}"
MAX_WAIT_CYCLES="${MAX_WAIT_CYCLES:-0}"

TASK_ARGS=""
if [[ -n "$TASK_IDS_CSV" ]]; then
  IFS=',' read -r -a TASK_IDS <<< "$TASK_IDS_CSV"
  for task_id in "${TASK_IDS[@]}"; do
    TASK_ARGS+=" --task-id $task_id"
  done
fi

RUN_EXTRA_ARGS=""
if [[ "$POLL_MISSING" == "1" ]]; then
  RUN_EXTRA_ARGS+=" --poll-missing"
fi
RUN_EXTRA_ARGS+=" --execution-mode $EXECUTION_MODE"
RUN_EXTRA_ARGS+=" --poll-interval-sec $POLL_INTERVAL_SEC"
RUN_EXTRA_ARGS+=" --max-wait-cycles $MAX_WAIT_CYCLES"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd "$ROOT_DIR"
mkdir -p "$QUEUE_DIR"
"$PYTHON_BIN" tools/eval/run_navhard_eval_queue.py init --queue-dir "$QUEUE_DIR"
nohup "$PYTHON_BIN" tools/eval/run_navhard_eval_queue.py run --queue-dir "$QUEUE_DIR" --smoke-stage-one "$SMOKE_STAGE_ONE" --smoke-stage-two "$SMOKE_STAGE_TWO"$TASK_ARGS$RUN_EXTRA_ARGS >"$QUEUE_DIR/worker_stdout.log" 2>&1 < /dev/null &
echo \$! > "$QUEUE_DIR/worker.pid"
echo "QUEUE_DIR=$QUEUE_DIR"
echo "WORKER_PID=\$(cat "$QUEUE_DIR/worker.pid")"
echo "STDOUT_LOG=$QUEUE_DIR/worker_stdout.log"
EOF
)

ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "bash -lc $(printf '%q' "$REMOTE_CMD")"
