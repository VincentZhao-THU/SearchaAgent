#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/infer/run_infer_with_retriever.sh --question "<your question>"

Optional arguments:
  --model-path <path>
  --retriever-url <url>
  --retriever-docs-url <url>
  --topk <int>
  --max-new-tokens <int>
  --temperature <float>
  --dtype <auto|bfloat16|float16|float32>

Optional environment variables:
  RETRIEVER_PYTHON
  INFER_PYTHON
  HEALTHCHECK_PYTHON
  RETRIEVER_CUDA_VISIBLE_DEVICES
  INFER_CUDA_VISIBLE_DEVICES
  RETRIEVER_READY_TIMEOUT_SECS
  RETRIEVER_READY_SLEEP_SECS
  RETRIEVER_HEALTHCHECK_REQUEST_TIMEOUT_SECS
  LOG_DIR
EOF
}

QUESTION=""
MODEL_PATH_ARG=""
RETRIEVER_URL_ARG=""
RETRIEVER_DOCS_URL_ARG=""
TOPK_ARG=""
MAX_NEW_TOKENS_ARG=""
TEMPERATURE_ARG=""
DTYPE_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --question)
            QUESTION="${2:-}"
            shift 2
            ;;
        --model-path)
            MODEL_PATH_ARG="${2:-}"
            shift 2
            ;;
        --retriever-url)
            RETRIEVER_URL_ARG="${2:-}"
            shift 2
            ;;
        --retriever-docs-url)
            RETRIEVER_DOCS_URL_ARG="${2:-}"
            shift 2
            ;;
        --topk)
            TOPK_ARG="${2:-}"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS_ARG="${2:-}"
            shift 2
            ;;
        --temperature)
            TEMPERATURE_ARG="${2:-}"
            shift 2
            ;;
        --dtype)
            DTYPE_ARG="${2:-}"
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

if [[ -z "$QUESTION" ]]; then
    echo "--question is required." >&2
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/nfs/volume-904-5/zhaowx/miniconda3/envs/retriever/bin/python}"
INFER_PYTHON="${INFER_PYTHON:-/nfs/volume-904-5/zhaowx/miniconda3/envs/searchr1-debug/bin/python}"
HEALTHCHECK_PYTHON="${HEALTHCHECK_PYTHON:-$RETRIEVER_PYTHON}"

RETRIEVER_CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-0,1}"
INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_PATH="${MODEL_PATH_ARG:-$ROOT_DIR/verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-em-rerun/global_step_200/actor}"
RETRIEVER_URL="${RETRIEVER_URL_ARG:-http://127.0.0.1:8000/retrieve}"
RETRIEVER_DOCS_URL="${RETRIEVER_DOCS_URL_ARG:-http://127.0.0.1:8000/docs}"
RETRIEVER_READY_TIMEOUT_SECS="${RETRIEVER_READY_TIMEOUT_SECS:-1000}"
RETRIEVER_READY_SLEEP_SECS="${RETRIEVER_READY_SLEEP_SECS:-2}"
RETRIEVER_HEALTHCHECK_REQUEST_TIMEOUT_SECS="${RETRIEVER_HEALTHCHECK_REQUEST_TIMEOUT_SECS:-120}"
TOPK="${TOPK_ARG:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS_ARG:-1024}"
TEMPERATURE="${TEMPERATURE_ARG:-0.7}"
DTYPE="${DTYPE_ARG:-bfloat16}"

LOG_ROOT_DIR="${LOG_DIR:-$ROOT_DIR/log}"
RUN_NAME="infer_with_retriever"
RUN_LOG_DIR="$LOG_ROOT_DIR/$RUN_NAME"

mkdir -p "$RUN_LOG_DIR"

RETRIEVER_LOG="$RUN_LOG_DIR/$RUN_NAME.retriever-launch.log"
HEALTHCHECK_LOG="$RUN_LOG_DIR/$RUN_NAME.retriever-healthcheck.json"

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

    if "$HEALTHCHECK_PYTHON" - "$RETRIEVER_DOCS_URL" "$RETRIEVER_URL" "$HEALTHCHECK_LOG" "$RETRIEVER_HEALTHCHECK_REQUEST_TIMEOUT_SECS" <<'PY'
import json
import sys
import urllib.error
import urllib.request

docs_url, retriever_url, output_path, timeout_secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])

with urllib.request.urlopen(docs_url, timeout=5) as response:
    if response.status != 200:
        raise RuntimeError(f"docs check failed with status {response.status}")

payload = json.dumps({
    "queries": ["capital of France"],
    "topk": 1,
}).encode("utf-8")
request = urllib.request.Request(
    retriever_url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=timeout_secs) as response:
    body = response.read().decode("utf-8")
    if response.status != 200:
        raise RuntimeError(f"retrieve check failed with status {response.status}")

parsed = json.loads(body)
if "result" not in parsed:
    raise RuntimeError("health check response missing 'result'")

with open(output_path, "w") as f:
    f.write(body)
PY
    then
        READY=1
        break
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

echo "Starting inference with $INFER_PYTHON"
echo "Infer CUDA_VISIBLE_DEVICES=$INFER_CUDA_VISIBLE_DEVICES"
echo "Model path: $MODEL_PATH"
echo "Question: $QUESTION"

CUDA_VISIBLE_DEVICES="$INFER_CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
"$INFER_PYTHON" "$ROOT_DIR/infer.py" \
    --question "$QUESTION" \
    --model-path "$MODEL_PATH" \
    --retriever-url "$RETRIEVER_URL" \
    --topk "$TOPK" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --dtype "$DTYPE"
