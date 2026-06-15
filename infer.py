import argparse
import re

import requests
import torch
import transformers


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-R1 inference with a local retriever.")
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path or Hugging Face ID of the model checkpoint to load.",
    )
    parser.add_argument(
        "--retriever-url",
        type=str,
        default=DEFAULT_RETRIEVER_URL,
        help="Retriever server endpoint.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of passages to retrieve for each search call.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens per generation round.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype used to load the model.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)

# Define the custom stopping criterion (DEPRECATED — kept for reference)
class StopOnSequence(transformers.StoppingCriteria):
    """DEPRECATED: No longer used.  Training relies on post-hoc string
    truncation instead of token-level stopping criteria (see
    postprocess_response_text in batch_infer_testsets.py).  Exact
    token-ID matching is fragile because BPE boundaries can shift with
    context, causing the criteria to silently fail.
    """
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
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


def search(query: str, retriever_url: str, topk: int):
    payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
    response = requests.post(retriever_url, json=payload, timeout=60)
    response.raise_for_status()
    results = response.json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
def main():
    args = parse_args()

    question = args.question.strip()
    if question and question[-1] != '?':
        question += '?'

    model_id = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_eos = [151645, 151643]  # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model_kwargs = {"device_map": "auto"}
    torch_dtype = resolve_dtype(args.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # No stopping_criteria — align with training which uses post-hoc
    # string truncation instead of token-ID matching.

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(prompt)

    while True:
        prompt, _ = truncate_text_by_tokens(tokenizer, prompt, 4096, keep="suffix")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature
        )
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_text = postprocess_response_text(output_text)

        match = re.search(r"<(search|answer)>(.*?)</\1>", output_text, re.DOTALL)
        if match:
            action = match.group(1)
            content = match.group(2).strip()
        else:
            action = None
            content = ""

        if action == "answer":
            print(output_text)
            break
        if action == "search":
            search_results = search(content, args.retriever_url, args.topk)
            observation = f"\n\n<information>{search_results}</information>\n\n"
            observation, _ = truncate_text_by_tokens(tokenizer, observation, 500)
            prompt += output_text + observation
            print(output_text + observation)
            continue

        invalid_feedback = (
            "\nMy previous action is invalid. "
            "If I want to search, I should put the query between <search> and </search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
            "Let me try again.\n"
        )
        observation, _ = truncate_text_by_tokens(tokenizer, invalid_feedback, 500)
        prompt += output_text + observation
        print(output_text + observation)


if __name__ == "__main__":
    main()
