#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/nq_hotpotqa/v0.2/run_batch_infer_existing_retriever.sh --model-path <path>

This script assumes the retriever service is already running.

Optional arguments:
  --model-path <path>
  --data-path <path>
  --retriever-url <url>
  --output-dir <path>
  --samples-per-source <int>
  --seed <int>
  --topk <int>
  --max-new-tokens <int>
  --temperature <float>
  --max-turns <int>
  --max-prompt-length <int>
  --max-obs-length <int>
  --dtype <auto|bfloat16|float16|float32>

Optional environment variables:
  INFER_PYTHON
  INFER_CUDA_VISIBLE_DEVICES
  LOG_DIR
EOF
}

MODEL_PATH_ARG=""
DATA_PATH_ARG=""
RETRIEVER_URL_ARG=""
OUTPUT_DIR_ARG=""
SAMPLES_PER_SOURCE_ARG=""
SEED_ARG=""
TOPK_ARG=""
MAX_NEW_TOKENS_ARG=""
TEMPERATURE_ARG=""
MAX_TURNS_ARG=""
MAX_PROMPT_LENGTH_ARG=""
MAX_OBS_LENGTH_ARG=""
DTYPE_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH_ARG="${2:-}"
            shift 2
            ;;
        --data-path)
            DATA_PATH_ARG="${2:-}"
            shift 2
            ;;
        --retriever-url)
            RETRIEVER_URL_ARG="${2:-}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_ARG="${2:-}"
            shift 2
            ;;
        --samples-per-source)
            SAMPLES_PER_SOURCE_ARG="${2:-}"
            shift 2
            ;;
        --seed)
            SEED_ARG="${2:-}"
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
        --max-turns)
            MAX_TURNS_ARG="${2:-}"
            shift 2
            ;;
        --max-prompt-length)
            MAX_PROMPT_LENGTH_ARG="${2:-}"
            shift 2
            ;;
        --max-obs-length)
            MAX_OBS_LENGTH_ARG="${2:-}"
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

if [[ -z "$MODEL_PATH_ARG" ]]; then
    echo "--model-path is required." >&2
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

INFER_PYTHON="${INFER_PYTHON:-/nfs/volume-904-5/zhaowx/miniconda3/envs/searchr1-debug/bin/python}"
INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_PATH="$MODEL_PATH_ARG"
DATA_PATH="${DATA_PATH_ARG:-$ROOT_DIR/data/nq_hotpotqa_train/test.parquet}"
RETRIEVER_URL="${RETRIEVER_URL_ARG:-http://127.0.0.1:8000/retrieve}"
OUTPUT_DIR="${OUTPUT_DIR_ARG:-${LOG_DIR:-$ROOT_DIR/log}/batch_infer_testsets_existing_retriever}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE_ARG:-10}"
SEED="${SEED_ARG:-42}"
TOPK="${TOPK_ARG:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS_ARG:-1024}"
TEMPERATURE="${TEMPERATURE_ARG:-0.7}"
MAX_TURNS="${MAX_TURNS_ARG:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH_ARG:-4096}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH_ARG:-500}"
DTYPE="${DTYPE_ARG:-bfloat16}"

mkdir -p "$OUTPUT_DIR"

echo "Starting batch inference with existing retriever"
echo "Infer python: $INFER_PYTHON"
echo "Infer CUDA_VISIBLE_DEVICES=$INFER_CUDA_VISIBLE_DEVICES"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Retriever url: $RETRIEVER_URL"
echo "Output dir: $OUTPUT_DIR"
echo "Samples per source: $SAMPLES_PER_SOURCE"
echo "Seed: $SEED"
echo "Topk: $TOPK"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Max turns: $MAX_TURNS"
echo "Max prompt length: $MAX_PROMPT_LENGTH"
echo "Max obs length: $MAX_OBS_LENGTH"
echo "Dtype: $DTYPE"

CUDA_VISIBLE_DEVICES="$INFER_CUDA_VISIBLE_DEVICES" \
PYTHONUNBUFFERED=1 \
"$INFER_PYTHON" "$ROOT_DIR/scripts/infer/batch_infer_testsets.py" \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --retriever-url "$RETRIEVER_URL" \
    --output-dir "$OUTPUT_DIR" \
    --samples-per-source "$SAMPLES_PER_SOURCE" \
    --seed "$SEED" \
    --topk "$TOPK" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --max-turns "$MAX_TURNS" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-obs-length "$MAX_OBS_LENGTH" \
    --dtype "$DTYPE"
