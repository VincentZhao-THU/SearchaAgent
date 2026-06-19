#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/infer/run_batch_infer_existing_retriever_v04_smoke.sh --model-path <path>

This smoke script assumes the retriever service is already running.
EOF
}

MODEL_PATH_ARG=""
DATA_PATH_ARG=""
RETRIEVER_URL_ARG=""
OUTPUT_DIR_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH_ARG="${2:-}"; shift 2 ;;
        --data-path) DATA_PATH_ARG="${2:-}"; shift 2 ;;
        --retriever-url) RETRIEVER_URL_ARG="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR_ARG="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH_ARG" ]]; then
    echo "--model-path is required." >&2
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

INFER_PYTHON="${INFER_PYTHON:-/nfs/volume-904-5/zhaowx/miniconda3/envs/searchr1-debug/bin/python}"
INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="$MODEL_PATH_ARG"
DATA_PATH="${DATA_PATH_ARG:-$ROOT_DIR/data/nq_hotpotqa_train_entropy_triggered/test.parquet}"
RETRIEVER_URL="${RETRIEVER_URL_ARG:-http://127.0.0.1:8000/retrieve}"
OUTPUT_DIR="${OUTPUT_DIR_ARG:-${LOG_DIR:-$ROOT_DIR/log}/batch_infer_entropy_triggered_testsets_smoke}"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$INFER_CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
"$INFER_PYTHON" "$ROOT_DIR/scripts/infer/batch_infer_entropy_triggered_testsets.py" \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --retriever-url "$RETRIEVER_URL" \
    --output-dir "$OUTPUT_DIR" \
    --samples-per-source 1 \
    --seed 42 \
    --topk 3 \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --max-turns 2 \
    --max-prompt-length 3072 \
    --max-obs-length 384 \
    --entropy-top-k 10 \
    --ema-alpha 0.3 \
    --trigger-threshold 0.2 \
    --trigger-tail-k 3 \
    --dtype bfloat16
