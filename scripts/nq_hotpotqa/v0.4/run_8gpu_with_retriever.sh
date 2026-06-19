#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/nq_hotpotqa/v0.4/run_8gpu_with_retriever.sh \
    --algo <ppo|grpo> \
    --base-model <model_path> \
    --experiment-name <experiment_name>

Optional environment variables:
  RETRIEVER_PYTHON
  TRAIN_PYTHON
  RETRIEVER_CUDA_VISIBLE_DEVICES
  TRAIN_CUDA_VISIBLE_DEVICES
  RETRIEVER_URL
  RETRIEVER_DOCS_URL
  RETRIEVER_READY_TIMEOUT_SECS
  RETRIEVER_READY_SLEEP_SECS
  LOG_DIR
EOF
}

ALGO=""
BASE_MODEL_ARG=""
EXPERIMENT_NAME_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)
            ALGO="${2:-}"
            shift 2
            ;;
        --base-model)
            BASE_MODEL_ARG="${2:-}"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME_ARG="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$ALGO" || -z "$BASE_MODEL_ARG" || -z "$EXPERIMENT_NAME_ARG" ]]; then
    echo "--algo, --base-model, and --experiment-name are required." >&2
    usage >&2
    exit 1
fi

case "$ALGO" in
    ppo)
        TRAIN_SCRIPT_NAME="train_ppo_entropy_trigger.sh"
        ;;
    grpo)
        TRAIN_SCRIPT_NAME="train_grpo_entropy_trigger.sh"
        ;;
    *)
        echo "--algo must be either 'ppo' or 'grpo'. Got: $ALGO" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/nfs/volume-904-5/zhaowx/miniconda3/envs/retriever/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/nfs/volume-904-5/zhaowx/miniconda3/envs/searchr1-debug/bin/python}"

RETRIEVER_CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_DOCS_URL="${RETRIEVER_DOCS_URL:-http://127.0.0.1:8000/docs}"
RETRIEVER_READY_TIMEOUT_SECS="${RETRIEVER_READY_TIMEOUT_SECS:-1000}"
RETRIEVER_READY_SLEEP_SECS="${RETRIEVER_READY_SLEEP_SECS:-2}"

BASE_MODEL="$BASE_MODEL_ARG"
EXPERIMENT_NAME="$EXPERIMENT_NAME_ARG"

LOG_ROOT_DIR="${LOG_DIR:-$ROOT_DIR/log}"
LOG_DIR="$LOG_ROOT_DIR/$EXPERIMENT_NAME"

mkdir -p "$LOG_DIR"

RETRIEVER_LOG="$LOG_DIR/$EXPERIMENT_NAME.retriever-launch.log"
HEALTHCHECK_LOG="$LOG_DIR/$EXPERIMENT_NAME.retriever-healthcheck.json"

cd "$ROOT_DIR"

cleanup() {
    if [[ -n "${RETRIEVER_PID:-}" ]] && kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        kill "$RETRIEVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo "Starting retriever with $RETRIEVER_PYTHON"
echo "Retriever CUDA_VISIBLE_DEVICES=$RETRIEVER_CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES="$RETRIEVER_CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
PYTHON_BIN="$RETRIEVER_PYTHON" \
bash "$ROOT_DIR/retrieval_launch.sh" >"$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!

READY=0
for _ in $(seq 1 $((RETRIEVER_READY_TIMEOUT_SECS / RETRIEVER_READY_SLEEP_SECS))); do
    if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        echo "Retriever exited unexpectedly. Tail of $RETRIEVER_LOG:"
        tail -n 200 "$RETRIEVER_LOG" || true
        exit 1
    fi

    if curl -fsS -m 3 "$RETRIEVER_DOCS_URL" >/dev/null 2>&1; then
        if curl -fsS -m 10 -X POST "$RETRIEVER_URL" \
            -H "Content-Type: application/json" \
            -d '{"queries":["capital of France"],"topk":1}' >"$HEALTHCHECK_LOG"; then
            READY=1
            break
        fi
    fi
    sleep "$RETRIEVER_READY_SLEEP_SECS"
done

if [[ "$READY" -ne 1 ]]; then
    echo "Retriever health check failed. Tail of $RETRIEVER_LOG:"
    tail -n 200 "$RETRIEVER_LOG" || true
    exit 1
fi

echo "Retriever is ready"
cat "$HEALTHCHECK_LOG"

echo "Starting $ALGO training with $TRAIN_PYTHON"
echo "Train CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "Base model: $BASE_MODEL"
echo "Experiment name: $EXPERIMENT_NAME"

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
PYTHON_BIN="$TRAIN_PYTHON" \
LOG_DIR="$LOG_DIR" \
BASE_MODEL="$BASE_MODEL" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
bash "$SCRIPT_DIR/$TRAIN_SCRIPT_NAME"
