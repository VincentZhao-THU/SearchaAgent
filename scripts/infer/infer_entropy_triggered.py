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

from search_r1.llm_agent.entropy import (  # noqa: E402
    EntropyTraceLogitsProcessor,
    build_entropy_windows,
    build_think_token_entropy_trace,
    build_token_entropy_records,
    ema_tail_trigger,
    records_to_aligned_sequences,
)


DEFAULT_MODEL_PATH = (
    "verl_checkpoints/nq_hotpotqa_train-search-r1-ppo-qwen2.5-3b-em-rerun/"
    "global_step_200/actor"
)
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_ENTROPY_TOP_K = 10
DEFAULT_MAX_PROMPT_LENGTH = 4096
DEFAULT_MAX_OBS_LENGTH = 500
DEFAULT_TRIGGER_THRESHOLD = 0.2
DEFAULT_TRIGGER_TAIL_K = 3
DEFAULT_EMA_ALPHA = 0.3

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Run entropy-triggered Search-R1 inference.")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--retriever-url", type=str, default=DEFAULT_RETRIEVER_URL)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-obs-length", type=int, default=DEFAULT_MAX_OBS_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--entropy-top-k", type=int, default=DEFAULT_ENTROPY_TOP_K)
    parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA)
    parser.add_argument("--trigger-threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD)
    parser.add_argument("--trigger-tail-k", type=int, default=DEFAULT_TRIGGER_TAIL_K)
    parser.add_argument("--output-path", type=str, default="log/entropy_triggered_single/example.json")
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


def truncate_text_by_tokens(tokenizer, text: str, max_tokens: int, keep: str = "prefix"):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) <= max_tokens:
        return text, False
    truncated_ids = input_ids[-max_tokens:] if keep == "suffix" else input_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False), True


def build_prompt(tokenizer, question: str):
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
Use <answer> and </answer> only when you are ready to provide the final answer. \
Search is controlled externally, so do not output <search> tags in your normal response. \
If you are explicitly asked to provide a search query, output only the raw query text and nothing else. \
Question: {question}\n"""
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def extract_answer(text: str):
    matches = list(ANSWER_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def postprocess_turn_response_text(text: str) -> str:
    if "</answer>" in text:
        return text.split("</answer>")[0] + "</answer>"
    if "</think>" in text:
        return text.split("</think>")[0] + "</think>"
    return text


def validate_query_text(query_text: str):
    stripped = query_text.strip()
    if not stripped:
        return {"valid": False, "query_text": None, "invalid_reason": "empty_query"}
    if "<" in stripped or ">" in stripped:
        return {"valid": False, "query_text": None, "invalid_reason": "query_contains_tags"}
    return {"valid": True, "query_text": stripped, "invalid_reason": None}


def parse_turn_response(response_text: str, token_records, ema_alpha: float, trigger_threshold: float, trigger_tail_k: int):
    answer = extract_answer(response_text)
    if answer is not None:
        return {
            "action": "answer",
            "answer": answer,
            "triggered_search": False,
            "think_entropy_ema": [],
            "think_entropy_records": [],
            "invalid_reason": None,
        }

    think_match = THINK_PATTERN.search(response_text)
    if not think_match:
        return {
            "action": "invalid",
            "answer": None,
            "triggered_search": False,
            "think_entropy_ema": [],
            "think_entropy_records": [],
            "invalid_reason": "missing_think_block",
        }

    trace = build_think_token_entropy_trace(token_records, ema_alpha=ema_alpha)
    trigger = ema_tail_trigger(
        trace["ema_values"],
        threshold=trigger_threshold,
        tail_k=trigger_tail_k,
    )
    return {
        "action": "search" if trigger["triggered"] else "continue",
        "answer": None,
        "triggered_search": trigger["triggered"],
        "think_entropy_ema": trace["ema_values"],
        "think_entropy_records": trace["filtered_records"],
        "invalid_reason": None,
    }


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


def generate_with_entropy(model, tokenizer, device, prompt, max_new_tokens, temperature, entropy_top_k, max_prompt_length):
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
    generated_text = postprocess_turn_response_text(generated_text)
    retained_token_count = min(generated_token_ids.numel(), len(entropy_processor.entropies))
    token_records = build_token_entropy_records(
        generated_token_ids=generated_token_ids[:retained_token_count],
        token_entropies=entropy_processor.entropies[:retained_token_count],
        tokenizer=tokenizer,
    )
    return prompt, generated_text, token_records


def generate_text(model, tokenizer, device, prompt, max_new_tokens, temperature, max_prompt_length):
    prompt, _ = truncate_text_by_tokens(tokenizer, prompt, max_prompt_length, keep="suffix")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    sequences = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
    )
    generated_token_ids = sequences[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return prompt, postprocess_turn_response_text(generated_text)


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
    ema_alpha,
    trigger_threshold,
    trigger_tail_k,
    model_path=None,
):
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    prompt = build_prompt(tokenizer, question)
    trace_parts = ["\n\n################# [Start Entropy Triggered Reasoning] ##################\n\n", prompt]
    turns = []
    all_records = []
    extracted_answer = None

    think_turn_index = 0
    followup_after_last_search = False

    while think_turn_index < max_turns or followup_after_last_search:
        prompt, generated_text, token_records = generate_with_entropy(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            entropy_top_k=entropy_top_k,
            max_prompt_length=max_prompt_length,
        )
        parsed = parse_turn_response(
            generated_text,
            token_records,
            ema_alpha=ema_alpha,
            trigger_threshold=trigger_threshold,
            trigger_tail_k=trigger_tail_k,
        )
        turn_record = {
            "think_turn_index": think_turn_index,
            "generated_text": generated_text,
            "postprocessed_text": generated_text,
            **records_to_aligned_sequences(token_records),
            "triggered_search": parsed["triggered_search"],
            "think_entropy_ema": parsed["think_entropy_ema"],
            "think_entropy_records": parsed["think_entropy_records"],
            "invalid_reason": parsed["invalid_reason"],
        }
        turns.append(turn_record)
        all_records.extend(token_records)
        trace_parts.append(generated_text)

        if parsed["action"] == "answer":
            extracted_answer = parsed["answer"]
            break

        if parsed["action"] == "search" and think_turn_index < max_turns:
            query_prompt = prompt + generated_text + "\nBased on the current context, provide only the raw search query text.\n"
            _, query_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=query_prompt,
                max_new_tokens=min(max_new_tokens, 64),
                temperature=temperature,
                max_prompt_length=max_prompt_length,
            )
            query_validation = validate_query_text(query_text)
            turns[-1]["query_text"] = query_validation.get("query_text")
            turns[-1]["query_generation_text"] = query_text
            if query_validation["valid"]:
                search_results = search(query_validation["query_text"], retriever_url, topk)
                observation = f"\n\n<information>{search_results}</information>\n\n"
            else:
                observation = "\nMy previous query is invalid. Please output only the raw search query text.\n"
                turns[-1]["invalid_reason"] = query_validation["invalid_reason"]
            observation, _ = truncate_text_by_tokens(tokenizer, observation, max_obs_length)
            prompt += generated_text + observation
            trace_parts.append(observation)
            think_turn_index += 1
            followup_after_last_search = think_turn_index >= max_turns
            continue

        if parsed["action"] == "continue":
            prompt += generated_text
            think_turn_index += 1
            followup_after_last_search = False
            continue

        invalid_feedback = (
            "\nMy previous action is invalid. "
            "Please reason inside <think> and </think>, or provide the final answer inside <answer> and </answer>. "
            "Let me try again.\n"
        )
        observation, _ = truncate_text_by_tokens(tokenizer, invalid_feedback, max_obs_length)
        prompt += generated_text + observation
        trace_parts.append(observation)
        think_turn_index += 1
        followup_after_last_search = False

    if extracted_answer is None:
        prompt, final_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_prompt_length=max_prompt_length,
        )
        trace_parts.append(final_text)
        extracted_answer = extract_answer("".join(trace_parts))

    return {
        "question": question,
        "model_path": model_path,
        "search_control_mode": "entropy_triggered",
        "entropy_top_k": entropy_top_k,
        "ema_alpha": ema_alpha,
        "trigger_threshold": trigger_threshold,
        "trigger_tail_k": trigger_tail_k,
        "full_trace": "".join(trace_parts),
        "turns": turns,
        "entropy_windows": build_entropy_windows(all_records, window_size=1),
        "extracted_answer": extracted_answer,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model_kwargs = {"device_map": "auto"}
    torch_dtype = resolve_dtype(args.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    result = run_single_example(
        model=model,
        tokenizer=tokenizer,
        device=device,
        question=args.question,
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

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
