
file_path=/nfs/volume-904-5/zhaowx/datasets/wiki18
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/nfs/volume-904-5/zhaowx/models/e5-base-v2
CACHE_ROOT=/nfs/volume-904-5/zhaowx/.cache/search-r1

mkdir -p "$CACHE_ROOT"/hf_home
mkdir -p "$CACHE_ROOT"/hf_datasets
mkdir -p "$CACHE_ROOT"/hf_hub
mkdir -p "$CACHE_ROOT"/tmp

export HF_HOME="$CACHE_ROOT/hf_home"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf_hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$CACHE_ROOT/tmp"
export TEMP="$CACHE_ROOT/tmp"

PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
