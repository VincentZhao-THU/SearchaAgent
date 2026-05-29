# searchr1-debug HF Baseline

This document records the currently verified working combinations in the
`searchr1-debug` environment.

## Environment snapshot

- Conda env: `searchr1-debug`
- Python: `3.10.20`
- PyTorch: `2.5.1+cu124`
- CUDA toolkit reported by torch: `12.4`
- Transformers: `4.47.1`
- `verl`: editable install from this repo
- `vllm`: not installed
- `flash-attn`: `2.8.3`

## Known-good behavior

The following paths are verified to work in this environment:

- HF minimal script with `attn_implementation=sdpa`
- HF minimal script with `attn_implementation=flash_attention_2`
- HF rollout
- `use_remove_padding=False`
- model dtype:
  - `float32` forward works
  - `float32` generate works
  - `bfloat16` generate works in the minimal HF script

## Final verified combinations

### HF + SDPA baseline

- `attn_implementation=sdpa`
- `use_remove_padding=False`
- minimal `forward` works
- minimal `generate` works
- PPO training reaches `epoch 0, step 1`

### HF + FlashAttention2 baseline

- `flash-attn==2.8.3`
- `attn_implementation=flash_attention_2`
- `use_remove_padding=False`
- minimal `forward` works
- minimal `generate` works
- PPO training reaches `epoch 0, step 1`

## Training status reached

`train_ppo_2gpu_hf.sh` in this environment now gets past the original
forward/generate failure mode and reaches PPO training step 1 under both:

- HF rollout + `sdpa`
- HF rollout + `flash_attention_2`

The original long-standing issue we were validating is considered resolved for
the HF stack in `searchr1-debug`.

## Current caveat

The latest training run later fails in critic update with a Ray
`ActorDiedError`, which is a later-stage stability issue and is distinct from
the original forward/generation stack failure.

## Baseline config knobs

From `train_ppo_2gpu_hf.sh`:

- `actor_rollout_ref.rollout.name=hf`
- `actor_rollout_ref.model.use_remove_padding=False`
- `critic.model.use_remove_padding=False`
- `+actor_rollout_ref.rollout.micro_batch_size=4`

The attention implementation can be switched between:

- `+actor_rollout_ref.model.attn_implementation=sdpa`
- `+critic.model.attn_implementation=sdpa`

or

- `+actor_rollout_ref.model.attn_implementation=flash_attention_2`
- `+critic.model.attn_implementation=flash_attention_2`

## Notes

This file should be updated before and after any direct `vllm` or
`flash-attn` experiments in `searchr1-debug`.

## Dependency experiment notes

These were checked directly in `searchr1-debug` with `pip install --dry-run`.

### vLLM

- `vllm==0.5.4`
  - wants `torch==2.4.0`
  - wants `torchvision==0.19.0`
- `vllm==0.6.3`
  - wants `torch==2.4.0`
- `vllm==0.8.5`
  - wants `torch==2.6.0`
  - wants `transformers>=4.51.1`

Conclusion:

- The repo-supported `vllm` line (`0.5.4` / `0.6.3`) conflicts with the
  current working HF baseline because it downgrades torch from `2.5.1` to
  `2.4.0`.
- A newer `vllm` line such as `0.8.5` moves the environment onto a different
  torch/transformers stack and should not be mixed into this baseline without a
  separate validation pass.

### flash-attn

- `flash-attn==2.8.3 --no-build-isolation`
  - dry-run keeps the current `torch==2.5.1`
  - no torch downgrade was requested

Conclusion:

- `flash-attn 2.8.3` is compatible with the current baseline at both the
  dependency level and runtime level.
- Runtime validation succeeded for:
  - minimal HF `forward`
  - minimal HF `generate`
  - HF PPO training reaching `epoch 0, step 1`
