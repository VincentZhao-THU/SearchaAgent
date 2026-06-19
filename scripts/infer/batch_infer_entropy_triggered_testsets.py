import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import torch
import transformers

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_entropy_triggered import (  # noqa: E402
    DEFAULT_ENTROPY_TOP_K,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_OBS_LENGTH,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RETRIEVER_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIGGER_TAIL_K,
    DEFAULT_TRIGGER_THRESHOLD,
    DEFAULT_EMA_ALPHA,
    resolve_dtype,
    run_single_example,
)
from verl.utils.reward_score.qa_em import em_check  # noqa: E402


DEFAULT_DATA_PATH = "data/nq_hotpotqa_train_entropy_triggered/test.parquet"
DEFAULT_OUTPUT_DIR = "log/batch_infer_entropy_triggered_testsets"
DEFAULT_TOPK = 3
DEFAULT_SAMPLES_PER_SOURCE = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TURNS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with entropy-triggered search.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--retriever-url", type=str, default=DEFAULT_RETRIEVER_URL)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-obs-length", type=int, default=DEFAULT_MAX_OBS_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--entropy-top-k", type=int, default=DEFAULT_ENTROPY_TOP_K)
    parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA)
    parser.add_argument("--trigger-threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD)
    parser.add_argument("--trigger-tail-k", type=int, default=DEFAULT_TRIGGER_TAIL_K)
    parser.add_argument("--samples-per-source", type=int, default=DEFAULT_SAMPLES_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args()


def select_examples(ds, samples_per_source: int, seed: int):
    grouped = defaultdict(list)
    for idx, source in enumerate(ds["data_source"]):
        grouped[source].append(idx)

    rng = random.Random(seed)
    selected = {}
    for source, indices in sorted(grouped.items()):
        sample_num = min(samples_per_source, len(indices))
        selected[source] = rng.sample(indices, sample_num)
    return selected


def print_and_log(message: str, log_fp):
    print(message, flush=True)
    log_fp.write(message + "\n")
    log_fp.flush()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "batch_infer_entropy_triggered_testsets.log"
    jsonl_path = output_dir / "batch_infer_entropy_triggered_testsets.jsonl"

    ds = datasets.load_dataset("parquet", data_files=args.data_path, split="train")
    selected = select_examples(ds, args.samples_per_source, args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model_kwargs = {"device_map": "auto"}
    torch_dtype = resolve_dtype(args.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total = 0
    total_correct = 0

    with log_path.open("w", encoding="utf-8") as log_fp, jsonl_path.open("w", encoding="utf-8") as jsonl_fp:
        print_and_log(f"Model path: {args.model_path}", log_fp)
        print_and_log(f"Data path: {args.data_path}", log_fp)
        print_and_log(f"Retriever url: {args.retriever_url}", log_fp)
        print_and_log(f"Search control mode: entropy_triggered", log_fp)
        print_and_log(f"Samples per source: {args.samples_per_source}", log_fp)
        print_and_log(f"Random seed: {args.seed}", log_fp)
        print_and_log(f"Max new tokens: {args.max_new_tokens}", log_fp)
        print_and_log(f"Max turns: {args.max_turns}", log_fp)
        print_and_log(f"Entropy top-k: {args.entropy_top_k}", log_fp)
        print_and_log(f"EMA alpha: {args.ema_alpha}", log_fp)
        print_and_log(f"Trigger threshold: {args.trigger_threshold}", log_fp)
        print_and_log(f"Trigger tail-k: {args.trigger_tail_k}", log_fp)
        print_and_log("", log_fp)

        for source in sorted(selected):
            print_and_log(f"{'=' * 30} DATA SOURCE: {source} {'=' * 30}", log_fp)
            source_correct = 0
            for sample_idx, ds_idx in enumerate(selected[source], start=1):
                item = ds[ds_idx]
                result = run_single_example(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    question=item["question"],
                    retriever_url=args.retriever_url,
                    topk=args.topk,
                    max_new_tokens=args.max_new_tokens,
                    max_turns=args.max_turns,
                    max_prompt_length=args.max_prompt_length,
                    max_obs_length=args.max_obs_length,
                    temperature=args.temperature,
                    entropy_top_k=args.entropy_top_k,
                    ema_alpha=args.ema_alpha,
                    trigger_threshold=args.trigger_threshold,
                    trigger_tail_k=args.trigger_tail_k,
                    model_path=args.model_path,
                )
                predicted_answer = result["extracted_answer"]
                is_correct = bool(predicted_answer is not None and em_check(predicted_answer, item["golden_answers"]))

                total += 1
                total_correct += int(is_correct)
                source_correct += int(is_correct)

                marker = "CORRECT" if is_correct else "WRONG"
                print_and_log(f"[{marker}] source={source} sample={sample_idx} dataset_index={ds_idx} id={item['id']}", log_fp)
                print_and_log(f"Question: {item['question']}", log_fp)
                print_and_log(f"Golden Answer: {item['golden_answers']}", log_fp)
                print_and_log("Model Full Answer Process:", log_fp)
                print_and_log(result["full_trace"], log_fp)
                print_and_log(f"Extracted Model Answer: {predicted_answer}", log_fp)
                print_and_log("", log_fp)

                output_record = {
                    "id": item["id"],
                    "data_source": source,
                    "dataset_index": ds_idx,
                    "question": item["question"],
                    "golden_answers": item["golden_answers"],
                    "model_answer": predicted_answer,
                    "is_correct": is_correct,
                    "num_rounds": len(result["turns"]),
                    "model_path": args.model_path,
                    "search_control_mode": "entropy_triggered",
                    "entropy_top_k": args.entropy_top_k,
                    "ema_alpha": args.ema_alpha,
                    "trigger_threshold": args.trigger_threshold,
                    "trigger_tail_k": args.trigger_tail_k,
                    "full_trace": result["full_trace"],
                    "turns": result["turns"],
                    "entropy_windows": result["entropy_windows"],
                }
                jsonl_fp.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                jsonl_fp.flush()

            print_and_log(
                f"Summary for {source}: {source_correct}/{len(selected[source])} correct, accuracy={source_correct / len(selected[source]):.4f}",
                log_fp,
            )
            print_and_log("", log_fp)

        print_and_log(f"Overall: {total_correct}/{total} correct, accuracy={total_correct / total:.4f}", log_fp)
        print_and_log(f"Saved text log to: {log_path}", log_fp)
        print_and_log(f"Saved jsonl log to: {jsonl_path}", log_fp)


if __name__ == "__main__":
    main()
