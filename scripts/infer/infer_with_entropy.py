import argparse
import json
import re
import sys
from pathlib import Path

import requests
import torch
import transformers
from transformers import LogitsProcessorList

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from search_r1.llm_agent.entropy import (
    EntropyTraceLogitsProcessor,
    build_entropy_windows,
    build_token_entropy_records,
    records_to_aligned_sequences,
)


DEFAULT_QUESTION = (
    "Mike Barnett negotiated many contracts including which player that went on to "
    "become general manager of CSKA Moscow of the Kontinental Hockey League?"
)
DEFAULT_MODEL_PATH = (
    "verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-em-rerun/"
    "global_step_200/actor"
)
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_ENTROPY_TOP_K = 10
DEFAULT_ENTROPY_WINDOW_SIZE = 1
DEFAULT_MAX_PROMPT_LENGTH = 4096
DEFAULT_MAX_OBS_LENGTH = 500


ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-R1 inference with token entropy tracing.")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--retriever-url", type=str, default=DEFAULT_RETRIEVER_URL)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-obs-length", type=int, default=DEFAULT_MAX_OBS_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--entropy-top-k", type=int, default=DEFAULT_ENTROPY_TOP_K)
    parser.add_argument("--entropy-window-size", type=int, default=DEFAULT_ENTROPY_WINDOW_SIZE)
    parser.add_argument("--output-path", type=str, default="log/entropy_single/example.json")
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


def postprocess_response_text(text: str) -> str:
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
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return text, False
    if keep == "suffix":
        truncated_ids = input_ids[-max_tokens:]
    else:
        truncated_ids = input_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False), True


def search(query: str, retriever_url: str, topk: int):
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    response = requests.post(retriever_url, json=payload, timeout=60)
    response.raise_for_status()
    results = response.json()["result"]

    passages = []
    for idx, doc_item in enumerate(results[0]):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        passages.append(f"Doc {idx + 1}(Title: {title}) {text}\n")
    return "".join(passages)


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


def _generate_with_entropy(model, tokenizer, device, prompt, max_new_tokens, temperature, entropy_top_k, max_prompt_length):
    prompt, _ = truncate_text_by_tokens(tokenizer, prompt, max_prompt_length, keep="suffix")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    entropy_processor = EntropyTraceLogitsProcessor(top_k=entropy_top_k)
    sequences = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        logits_processor=LogitsProcessorList([entropy_processor]),
    )

    generated_token_ids = sequences[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    postprocessed_text = postprocess_response_text(generated_text)
    retained_token_count = len(tokenizer.encode(postprocessed_text, add_special_tokens=False))
    retained_token_count = min(retained_token_count, generated_token_ids.numel(), len(entropy_processor.entropies))
    token_records = build_token_entropy_records(
        generated_token_ids=generated_token_ids[:retained_token_count],
        token_entropies=entropy_processor.entropies[:retained_token_count],
        tokenizer=tokenizer,
    )
    return prompt, generated_text, generated_token_ids, token_records


def run_single_example(
    model,
    tokenizer,
    device,
    question,
    retriever_url,
    topk,
    max_new_tokens,
    max_turns,
    max_prompt_length,
    max_obs_length,
    temperature,
    entropy_top_k,
    entropy_window_size,
    model_path=None,
):
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    prompt = build_prompt(tokenizer, question)
    trace_parts = ["\n\n################# [Start Reasoning + Searching] ##################\n\n", prompt]
    turns = []
    extracted_answer = None
    global_token_offset = 0

    all_records = []

    for turn_index in range(max_turns):
        prompt, generated_text, generated_token_ids, token_records = _generate_with_entropy(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            entropy_top_k=entropy_top_k,
            max_prompt_length=max_prompt_length,
        )

        for record in token_records:
            record["turn_index"] = turn_index
            record["token_index_global"] = global_token_offset + record["token_index_in_turn"]
        global_token_offset += len(token_records)

        postprocessed_text = postprocess_response_text(generated_text)
        action, content = parse_first_action(postprocessed_text)
        turn_records = records_to_aligned_sequences(token_records)
        all_records.extend(token_records)
        turns.append({
            "turn_index": turn_index,
            "action": action,
            "content": content,
            "generated_text": generated_text,
            "postprocessed_text": postprocessed_text,
            **turn_records,
        })
        trace_parts.append(postprocessed_text)

        if action == "answer":
            extracted_answer = content
            break
        if action == "search":
            search_results = search(content, retriever_url, topk)
            observation = f"\n\n<information>{search_results}</information>\n\n"
            observation, _ = truncate_text_by_tokens(tokenizer, observation, max_obs_length)
            prompt += postprocessed_text + observation
            trace_parts.append(observation)
            continue

        invalid_feedback = (
            "\nMy previous action is invalid. "
            "If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
            "Let me try again.\n"
        )
        observation, _ = truncate_text_by_tokens(tokenizer, invalid_feedback, max_obs_length)
        prompt += postprocessed_text + observation
        trace_parts.append(observation)

    full_trace = "".join(trace_parts)
    entropy_windows = build_entropy_windows(all_records, window_size=entropy_window_size)

    return {
        "question": question,
        "model_path": model_path,
        "entropy_top_k": entropy_top_k,
        "entropy_window_size": entropy_window_size,
        "normalized": False,
        "full_trace": full_trace,
        "turns": turns,
        "entropy_windows": entropy_windows,
        "extracted_answer": extracted_answer,
    }


def main():
    args = parse_args()
    question = args.question.strip()
    model_id = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model_kwargs = {"device_map": "auto"}
    torch_dtype = resolve_dtype(args.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    result = run_single_example(
        model=model,
        tokenizer=tokenizer,
        device=device,
        question=question,
        retriever_url=args.retriever_url,
        topk=args.topk,
        max_new_tokens=args.max_new_tokens,
        max_turns=args.max_turns,
        max_prompt_length=args.max_prompt_length,
        max_obs_length=args.max_obs_length,
        temperature=args.temperature,
        entropy_top_k=args.entropy_top_k,
        entropy_window_size=args.entropy_window_size,
        model_path=args.model_path,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.pop("extracted_answer", None)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
