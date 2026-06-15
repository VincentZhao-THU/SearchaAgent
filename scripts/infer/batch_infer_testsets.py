import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import requests
import torch
import transformers

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from verl.utils.reward_score.qa_em import em_check


DEFAULT_MODEL_PATH = (
    "verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-em-rerun/"
    "global_step_200/actor"
)
DEFAULT_DATA_PATH = "data/nq_hotpotqa_train/test.parquet"
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_OUTPUT_DIR = "log/batch_infer_testsets"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOPK = 3
DEFAULT_SAMPLES_PER_SOURCE = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_PROMPT_LENGTH = 4096
DEFAULT_MAX_OBS_LENGTH = 500

ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference on 7 test subsets with Search-R1.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--retriever-url", type=str, default=DEFAULT_RETRIEVER_URL)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--samples-per-source", type=int, default=DEFAULT_SAMPLES_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-obs-length", type=int, default=DEFAULT_MAX_OBS_LENGTH)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


class StopOnSequence(transformers.StoppingCriteria):
    """DEPRECATED: Kept for API compatibility but no longer used.

    Training uses post-hoc string truncation (postprocess_response_text)
    instead of token-level stopping criteria, because exact token-ID matching
    is fragile — BPE token boundaries can shift depending on context, causing
    the criteria to silently fail.  The inference script now follows the same
    approach: generate up to max_new_tokens, then truncate at the first
    </answer> or </search> closing tag via postprocess_response_text.
    """
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    return None


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def postprocess_response_text(text: str) -> str:
    """Truncate at the first closing tag, with </search> priority over </answer>.

    Aligns with training's _postprocess_responses: if the response contains
    </search>, truncate there (the model was doing a search); else if it
    contains </answer>, truncate there; else keep the full response.
    """
    if "</search>" in text:
        return text.split("</search>")[0] + "</search>"
    if "</answer>" in text:
        return text.split("</answer>")[0] + "</answer>"
    return text


def parse_first_action(text: str):
    match = ACTION_PATTERN.search(text)
    if not match:
        return None, ""
    return match.group(1), match.group(2).strip()


def truncate_text_by_tokens(tokenizer, text: str, max_tokens: int, keep: str = "prefix"):
    """Truncate text to at most max_tokens tokens.

    keep="prefix" keeps the beginning (for observations).
    keep="suffix" keeps the end (for rolling prompts, matching training).
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return text, False
    if keep == "suffix":
        truncated_ids = input_ids[-max_tokens:]
    else:
        truncated_ids = input_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False), True


def build_invalid_action_feedback():
    return (
        "\nMy previous action is invalid. "
        "If I want to search, I should put the query between <search> and </search>. "
        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
        "Let me try again.\n"
    )


def search(query: str, retriever_url: str, topk: int):
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
    }
    response = requests.post(retriever_url, json=payload, timeout=60)
    response.raise_for_status()
    results = response.json()["result"]

    def _passages2string(retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def build_prompt(tokenizer, question: str):
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def run_single_example(
    model,
    tokenizer,
    device,
    question,
    retriever_url,
    topk,
    max_new_tokens,
    temperature,
    max_turns,
    max_prompt_length,
    max_obs_length,
):
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    prompt = build_prompt(tokenizer, question)
    trace_parts = [
        "\n\n################# [Start Reasoning + Searching] ##################\n\n",
        prompt,
    ]
    rounds = 0
    extracted_answer = None

    def _generate_once(curr_prompt: str):
        trimmed_prompt, prompt_truncated = truncate_text_by_tokens(
            tokenizer, curr_prompt, max_prompt_length, keep="suffix"
        )
        if prompt_truncated:
            curr_prompt = trimmed_prompt
        input_ids = tokenizer.encode(curr_prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )
        generated_tokens = outputs[0][input_ids.shape[1]:]
        raw_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return curr_prompt, postprocess_response_text(raw_output_text)

    for _ in range(max_turns):
        prompt, output_text = _generate_once(prompt)
        trace_parts.append(output_text)

        action, content = parse_first_action(output_text)
        if action == "answer":
            extracted_answer = content
            break
        if action == "search":
            search_results = search(content, retriever_url, topk)
            observation = f"\n\n<information>{search_results}</information>\n\n"
            observation, _ = truncate_text_by_tokens(tokenizer, observation, max_obs_length)
            prompt += output_text + observation
            trace_parts.append(observation)
            rounds += 1
            continue

        invalid_feedback = build_invalid_action_feedback()
        observation, _ = truncate_text_by_tokens(tokenizer, invalid_feedback, max_obs_length)
        prompt += output_text + observation
        trace_parts.append(observation)

    else:
        prompt, output_text = _generate_once(prompt)
        trace_parts.append(output_text)
        action, content = parse_first_action(output_text)
        if action == "answer":
            extracted_answer = content

    if extracted_answer is None:
        extracted_answer = extract_answer("".join(trace_parts))

    full_trace = "".join(trace_parts)
    return {
        "full_trace": full_trace,
        "extracted_answer": extracted_answer,
        "num_rounds": rounds,
    }


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
    log_path = output_dir / "batch_infer_testsets.log"
    jsonl_path = output_dir / "batch_infer_testsets.jsonl"

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
        print_and_log(f"Samples per source: {args.samples_per_source}", log_fp)
        print_and_log(f"Random seed: {args.seed}", log_fp)
        print_and_log(f"Max new tokens: {args.max_new_tokens}", log_fp)
        print_and_log(f"Max turns: {args.max_turns}", log_fp)
        print_and_log(f"Max prompt length: {args.max_prompt_length}", log_fp)
        print_and_log(f"Max obs length: {args.max_obs_length}", log_fp)
        print_and_log("", log_fp)

        for source in sorted(selected):
            print_and_log(f"{'=' * 30} DATA SOURCE: {source} {'=' * 30}", log_fp)
            source_correct = 0
            for sample_idx, ds_idx in enumerate(selected[source], start=1):
                item = ds[ds_idx]
                question = item["question"]
                golden_answers = item["golden_answers"]
                result = run_single_example(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    question=question,
                    retriever_url=args.retriever_url,
                    topk=args.topk,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    max_turns=args.max_turns,
                    max_prompt_length=args.max_prompt_length,
                    max_obs_length=args.max_obs_length,
                )
                predicted_answer = result["extracted_answer"]
                is_correct = bool(predicted_answer is not None and em_check(predicted_answer, golden_answers))
                marker = "WRONG" if not is_correct else "CORRECT"

                total += 1
                total_correct += int(is_correct)
                source_correct += int(is_correct)

                print_and_log(f"[{marker}] source={source} sample={sample_idx} dataset_index={ds_idx} id={item['id']}", log_fp)
                print_and_log(f"Question: {question}", log_fp)
                print_and_log(f"Golden Answer: {golden_answers}", log_fp)
                print_and_log("Model Full Answer Process:", log_fp)
                print_and_log(result["full_trace"], log_fp)
                print_and_log(f"Extracted Model Answer: {predicted_answer}", log_fp)
                if not is_correct:
                    print_and_log("########## ANSWER MISMATCH ##########", log_fp)
                print_and_log("", log_fp)

                jsonl_fp.write(json.dumps({
                    "id": item["id"],
                    "data_source": source,
                    "dataset_index": ds_idx,
                    "question": question,
                    "golden_answers": golden_answers,
                    "model_answer": predicted_answer,
                    "is_correct": is_correct,
                    "num_rounds": result["num_rounds"],
                    "full_trace": result["full_trace"],
                }, ensure_ascii=False) + "\n")
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
