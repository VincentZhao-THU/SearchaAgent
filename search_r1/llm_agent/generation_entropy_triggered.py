import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from verl import DataProto

from .entropy import build_think_token_entropy_trace, ema_tail_trigger
from .generation import LLMGenerationManager


@dataclass
class EntropySearchControlConfig:
    entropy_top_k: int = 10
    ema_alpha: float = 0.3
    trigger_threshold: float = 0.2
    trigger_tail_k: int = 3
    query_max_new_tokens: int = 64


class EntropyTriggeredGenerationManager(LLMGenerationManager):
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(self, tokenizer, actor_rollout_wg, config, is_validation: bool = False):
        super().__init__(tokenizer=tokenizer, actor_rollout_wg=actor_rollout_wg, config=config, is_validation=is_validation)
        self.entropy_control = getattr(
            config,
            "entropy_control",
            EntropySearchControlConfig(
                entropy_top_k=getattr(config, "entropy_top_k", 10),
                ema_alpha=getattr(config, "entropy_ema_alpha", 0.3),
                trigger_threshold=getattr(config, "entropy_trigger_threshold", 0.2),
                trigger_tail_k=getattr(config, "entropy_trigger_tail_k", 3),
                query_max_new_tokens=getattr(config, "query_max_new_tokens", 64),
            ),
        )

    def _extract_think_span_records(
        self,
        response_text: str,
        token_records: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not token_records or self.ANSWER_PATTERN.search(response_text):
            return []
        think_match = self.THINK_PATTERN.search(response_text)
        if think_match is None:
            return []

        end_tag = "</think>"
        collected_records: List[Dict[str, Any]] = []
        reconstructed_text = ""
        for record in token_records:
            copied_record = dict(record)
            token_text = str(copied_record.get("generated_token", ""))
            collected_records.append(copied_record)
            reconstructed_text += token_text
            normalized_text = reconstructed_text.replace("\n", "")
            if normalized_text.endswith(end_tag):
                return collected_records

        return collected_records

    def parse_turn_response(
        self,
        response_text: str,
        token_records: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer_match = self.ANSWER_PATTERN.search(response_text)
        if answer_match:
            return {
                "action": "answer",
                "answer": answer_match.group(1).strip(),
                "triggered_search": False,
                "think_entropy_ema": [],
                "think_entropy_records": [],
                "invalid_reason": None,
            }

        think_match = self.THINK_PATTERN.search(response_text)
        if not think_match:
            return {
                "action": "invalid",
                "answer": None,
                "triggered_search": False,
                "think_entropy_ema": [],
                "think_entropy_records": [],
                "invalid_reason": "missing_think_block",
            }

        think_records = self._extract_think_span_records(response_text, token_records)
        think_trace = build_think_token_entropy_trace(
            think_records,
            ema_alpha=self.entropy_control.ema_alpha,
        )
        trigger = ema_tail_trigger(
            think_trace["ema_values"],
            threshold=self.entropy_control.trigger_threshold,
            tail_k=self.entropy_control.trigger_tail_k,
        )
        return {
            "action": "search" if trigger["triggered"] else "continue",
            "answer": None,
            "triggered_search": trigger["triggered"],
            "think_entropy_ema": think_trace["ema_values"],
            "think_entropy_records": think_trace["filtered_records"],
            "invalid_reason": None,
        }

    def validate_query_text(self, query_text: str) -> Dict[str, Optional[str]]:
        stripped = query_text.strip()
        if not stripped:
            return {"valid": False, "query_text": None, "invalid_reason": "empty_query"}
        if "<" in stripped or ">" in stripped:
            return {"valid": False, "query_text": None, "invalid_reason": "query_contains_tags"}
        return {"valid": True, "query_text": stripped, "invalid_reason": None}

    def _truncate_to_response_length(self, responses: torch.Tensor, max_new_tokens: Optional[int] = None) -> torch.Tensor:
        target_length = self.config.max_response_length if max_new_tokens is None else max_new_tokens
        if responses.shape[1] <= target_length:
            return responses
        return responses[:, :target_length]

    def _generate_active_batch(
        self,
        active_batch: DataProto,
        return_entropy_trace: bool = False,
        response_length: Optional[int] = None,
    ) -> DataProto:
        active_batch.meta_info["return_entropy_trace"] = return_entropy_trace
        active_batch.meta_info["entropy_top_k"] = self.entropy_control.entropy_top_k
        active_batch.meta_info["response_length"] = response_length or self.config.max_response_length
        return self._generate_with_gpu_padding(active_batch)

    def _batch_decode_responses(self, responses: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(responses, skip_special_tokens=True)

    def _postprocess_turn_response_text(self, response_text: str) -> str:
        if "</answer>" in response_text:
            return response_text.split("</answer>")[0] + "</answer>"
        if "</think>" in response_text:
            return response_text.split("</think>")[0] + "</think>"
        return response_text

    def _postprocess_turn_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        responses_str = self._batch_decode_responses(responses)
        processed_responses_str = [self._postprocess_turn_response_text(resp) for resp in responses_str]
        processed_responses = self._batch_tokenize(processed_responses_str)
        return processed_responses, processed_responses_str

    def _extract_token_records(
        self,
        responses: torch.Tensor,
        token_entropies: Optional[torch.Tensor],
    ) -> List[List[Dict[str, Any]]]:
        if token_entropies is None:
            return [[] for _ in range(responses.shape[0])]

        records_per_example: List[List[Dict[str, Any]]] = []
        for row_ids, row_entropy in zip(responses, token_entropies):
            valid_len = min(row_ids.numel(), row_entropy.numel())
            records = []
            for token_index in range(valid_len):
                token_id = int(row_ids[token_index].item())
                if token_id == self.tokenizer.pad_token_id:
                    break
                records.append({
                    "token_index_in_turn": token_index,
                    "token_id": token_id,
                    "generated_token": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "entropy": float(row_entropy[token_index].item()),
                })
            records_per_example.append(records)
        return records_per_example

    def _build_invalid_feedback(self, invalid_reason: str) -> str:
        if invalid_reason == "missing_think_block":
            return (
                "\nMy previous action is invalid. "
                "Please reason inside <think> and </think>, or provide the final answer inside "
                "<answer> and </answer>. Let me try again.\n"
            )
        return (
            "\nMy previous action is invalid. "
            "Please keep the output structure valid and try again.\n"
        )

    def _build_query_feedback(self, invalid_reason: str) -> str:
        if invalid_reason == "empty_query":
            return "\nMy previous query is empty. Please provide a non-empty search query only.\n"
        if invalid_reason == "query_contains_tags":
            return "\nMy previous query is invalid. Please output only the raw search query text without any tags.\n"
        return "\nMy previous query is invalid. Please output only the raw search query text.\n"

    def _build_query_prompt(self) -> str:
        return "\nBased on the current context, provide only the raw search query text.\n"

    def _run_query_round(self, rollings: DataProto, query_active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        active_batch = DataProto.from_dict({k: v[query_active_mask] for k, v in rollings.batch.items()})
        active_batch.meta_info.update(rollings.meta_info)
        query_output = self._generate_active_batch(
            active_batch,
            return_entropy_trace=False,
            response_length=self.entropy_control.query_max_new_tokens,
        )
        query_ids = self._truncate_to_response_length(
            query_output.batch["responses"],
            max_new_tokens=self.entropy_control.query_max_new_tokens,
        )
        query_texts = self._batch_decode_responses(query_ids)
        return query_ids, query_texts

    def _search_queries(self, query_texts: List[str]) -> List[str]:
        if not query_texts:
            return []
        return self.batch_search(query_texts)

    def _init_turn_logs(self, batch_size: int) -> Dict[str, List[Any]]:
        return {
            "triggered_search": [False] * batch_size,
            "think_entropy_records": [[] for _ in range(batch_size)],
            "think_entropy_ema": [[] for _ in range(batch_size)],
            "query_text": [None] * batch_size,
            "invalid_reason": [None] * batch_size,
        }

    def _finalize_meta_info(self, meta_info: Dict[str, Any], turn_logs: Dict[str, List[Any]], active_mask, think_turn_counts, valid_action_stats, valid_search_stats):
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["valid_action_stats"] = valid_action_stats.tolist()
        meta_info["valid_search_stats"] = valid_search_stats.tolist()
        meta_info["turns_stats"] = think_turn_counts.tolist()
        meta_info.update(turn_logs)
        return meta_info

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor):
        batch_size = gen_batch.batch["input_ids"].shape[0]
        original_left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {"responses": initial_input_ids[:, []], "responses_with_info_mask": initial_input_ids[:, []]}

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        think_turn_counts = torch.zeros(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        turn_logs = self._init_turn_logs(batch_size)
        meta_info: Dict[str, Any] = {}

        while active_mask.sum():
            if not think_turn_counts[active_mask].lt(self.config.max_turns).any():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            rollings_active.meta_info.update(rollings.meta_info)
            gen_output = self._generate_active_batch(rollings_active, return_entropy_trace=True)
            meta_info = dict(gen_output.meta_info)
            active_responses = self._truncate_to_response_length(gen_output.batch["responses"])
            active_responses, active_response_str = self._postprocess_turn_responses(active_responses)
            token_entropies = gen_output.batch["token_entropies"] if "token_entropies" in gen_output.batch.keys() else None
            active_token_records = self._extract_token_records(
                active_responses,
                token_entropies,
            )
            responses_ids, responses_str = self.tensor_fn._example_level_pad(active_responses, active_response_str, active_mask)

            next_obs = [""] * batch_size
            dones = [1] * batch_size
            valid_action = [0] * batch_size
            is_search = [0] * batch_size
            query_round_indices: List[int] = []

            active_example_index = 0
            for batch_index, is_active in enumerate(active_mask.tolist()):
                if not is_active:
                    continue

                parsed = self.parse_turn_response(
                    responses_str[batch_index],
                    active_token_records[active_example_index],
                )
                active_example_index += 1
                think_turn_counts[batch_index] += 1

                turn_logs["triggered_search"][batch_index] = (
                    turn_logs["triggered_search"][batch_index] or parsed["triggered_search"]
                )
                if parsed["think_entropy_ema"]:
                    turn_logs["think_entropy_ema"][batch_index] = parsed["think_entropy_ema"]
                if parsed["think_entropy_records"]:
                    turn_logs["think_entropy_records"][batch_index] = parsed["think_entropy_records"]
                if parsed["invalid_reason"] is not None:
                    turn_logs["invalid_reason"][batch_index] = parsed["invalid_reason"]

                if parsed["action"] == "answer":
                    dones[batch_index] = 1
                    valid_action[batch_index] = 1
                    continue

                if parsed["action"] == "search" and think_turn_counts[batch_index] <= self.config.max_turns:
                    next_obs[batch_index] = self._build_query_prompt()
                    dones[batch_index] = 0
                    valid_action[batch_index] = 1
                    is_search[batch_index] = 1
                    query_round_indices.append(batch_index)
                    continue

                if parsed["action"] == "continue":
                    dones[batch_index] = 0
                    valid_action[batch_index] = 1
                    continue

                next_obs[batch_index] = self._build_invalid_feedback(parsed["invalid_reason"])
                dones[batch_index] = 0

            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

            if query_round_indices:
                query_active_mask = torch.zeros_like(active_mask)
                for idx in query_round_indices:
                    query_active_mask[idx] = True

                query_ids, query_texts = self._run_query_round(rollings, query_active_mask)
                padded_query_ids, padded_query_texts = self.tensor_fn._example_level_pad(query_ids, query_texts, query_active_mask)

                retrieval_obs = [""] * batch_size
                valid_queries: List[str] = []
                valid_query_indices: List[int] = []
                for idx in query_round_indices:
                    validation = self.validate_query_text(padded_query_texts[idx])
                    turn_logs["query_text"][idx] = validation["query_text"] if validation["valid"] else padded_query_texts[idx].strip()
                    if validation["valid"]:
                        valid_queries.append(validation["query_text"])
                        valid_query_indices.append(idx)
                    else:
                        retrieval_obs[idx] = self._build_query_feedback(validation["invalid_reason"])
                        turn_logs["invalid_reason"][idx] = validation["invalid_reason"]

                if valid_queries:
                    search_results = self._search_queries(valid_queries)
                    for idx, search_result in zip(valid_query_indices, search_results):
                        retrieval_obs[idx] = f"\n\n<information>{search_result.strip()}</information>\n\n"

                retrieval_obs_ids = self._process_next_obs(retrieval_obs)
                rollings = self._update_rolling_state(rollings, padded_query_ids, retrieval_obs_ids)
                original_right_side = self._update_right_side(original_right_side, padded_query_ids, retrieval_obs_ids)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            rollings_active.meta_info.update(rollings.meta_info)
            gen_output = self._generate_active_batch(rollings_active, return_entropy_trace=False)
            meta_info = dict(gen_output.meta_info)
            final_ids = self._truncate_to_response_length(gen_output.batch["responses"])
            final_ids, final_str = self._postprocess_turn_responses(final_ids)
            final_valid_action = [0] * batch_size
            final_active_index = 0
            for batch_index, is_active in enumerate(active_mask.tolist()):
                if not is_active:
                    continue
                parsed = self.parse_turn_response(final_str[final_active_index], token_records=[])
                if parsed["action"] == "answer":
                    final_valid_action[batch_index] = 1
                final_active_index += 1
            final_ids, _ = self.tensor_fn._example_level_pad(final_ids, final_str, active_mask)
            original_right_side = self._update_right_side(original_right_side, final_ids)
            valid_action_stats += torch.tensor(final_valid_action, dtype=torch.int)

        meta_info = self._finalize_meta_info(
            meta_info,
            turn_logs,
            active_mask,
            think_turn_counts,
            valid_action_stats,
            valid_search_stats,
        )
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        return self._compose_final_output(original_left_side, original_right_side, meta_info)
