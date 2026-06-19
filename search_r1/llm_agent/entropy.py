import string
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import LogitsProcessor

from verl.utils.torch_functional import topk_entropy_from_logits


class EntropyTraceLogitsProcessor(LogitsProcessor):
    """Record per-step top-k entropy from raw next-token logits."""

    def __init__(self, top_k: int = 10):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.entropies: List[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step_entropy = topk_entropy_from_logits(scores, top_k=self.top_k)
        if step_entropy.dim() == 0:
            self.entropies.append(float(step_entropy.item()))
        else:
            if step_entropy.size(0) != 1:
                raise ValueError("EntropyTraceLogitsProcessor only supports batch size 1")
            self.entropies.append(float(step_entropy[0].item()))
        return scores


def _decode_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    return tokenizer.decode([token_id], skip_special_tokens=False)


def build_token_entropy_records(
    generated_token_ids: torch.Tensor,
    token_entropies: Sequence[float],
    tokenizer: Any,
    turn_index: int = 0,
    global_token_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Build simple per-token entropy records aligned by generated token index."""
    if generated_token_ids.dim() != 1:
        generated_token_ids = generated_token_ids.reshape(-1)

    limit = min(len(token_entropies), generated_token_ids.numel())
    records: List[Dict[str, Any]] = []
    for token_index in range(limit):
        token_id = int(generated_token_ids[token_index].item())
        records.append({
            "turn_index": turn_index,
            "token_index_in_turn": token_index,
            "token_index_global": global_token_offset + token_index,
            "token_id": token_id,
            "generated_token": _decode_token(tokenizer, token_id),
            "entropy": float(token_entropies[token_index]),
        })
    return records


def records_to_aligned_sequences(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Any]]:
    records = list(records)
    return {
        "generated_token_ids": [record["token_id"] for record in records],
        "generated_tokens": [record["generated_token"] for record in records],
        "token_entropies": [record["entropy"] for record in records],
    }


def build_entropy_windows(records: Sequence[Dict[str, Any]], window_size: int = 1) -> List[Dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    windows: List[Dict[str, Any]] = []
    for start in range(0, len(records), window_size):
        window_records = records[start:start + window_size]
        entropies = [float(record["entropy"]) for record in window_records]
        if not entropies:
            continue
        windows.append({
            "start_global_token_index": window_records[0]["token_index_global"],
            "end_global_token_index": window_records[-1]["token_index_global"],
            "num_tokens": len(window_records),
            "mean_entropy": sum(entropies) / len(entropies),
            "max_entropy": max(entropies),
            "min_entropy": min(entropies),
        })
    return windows


_THINK_TAG_PIECES = {"<", "</", "think", ">\n", ">", "\n", "<think>", "</think>"}


def is_meaningful_think_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    if token in _THINK_TAG_PIECES or stripped in _THINK_TAG_PIECES:
        return False
    if all(ch in string.punctuation for ch in stripped):
        return False
    return True


def compute_ema(values: Sequence[float], alpha: float = 0.3) -> List[float]:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    ema_values: List[float] = []
    cur = None
    for value in values:
        value = float(value)
        cur = value if cur is None else alpha * value + (1 - alpha) * cur
        ema_values.append(cur)
    return ema_values


def build_think_token_entropy_trace(
    records: Sequence[Dict[str, Any]],
    ema_alpha: float = 0.3,
) -> Dict[str, Any]:
    filtered_records = [
        dict(record) for record in records
        if is_meaningful_think_token(str(record.get("generated_token", "")))
    ]
    token_entropies = [float(record["entropy"]) for record in filtered_records]
    ema_values = compute_ema(token_entropies, alpha=ema_alpha) if token_entropies else []
    return {
        "filtered_records": filtered_records,
        "token_entropies": token_entropies,
        "ema_values": ema_values,
        "ema_alpha": float(ema_alpha),
    }


def ema_tail_trigger(
    ema_values: Sequence[float],
    threshold: float = 0.2,
    tail_k: int = 3,
) -> Dict[str, Any]:
    if tail_k <= 0:
        raise ValueError("tail_k must be positive")

    tail = [float(value) for value in ema_values[-tail_k:]]
    triggered = len(tail) == tail_k and all(value > threshold for value in tail)
    return {
        "triggered": triggered,
        "tail_ema": tail,
        "threshold": float(threshold),
        "tail_k": int(tail_k),
    }
