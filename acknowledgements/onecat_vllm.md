# 1Cat-vLLM (1CatAI) — acknowledgement & lessons

> **Status: Final** — an independent parallel effort; credited here, no code imported.

**Notes packet:** [../upstream_feedback/onecat_vllm/](../upstream_feedback/onecat_vllm/) — the comparison notes (no code).

**Upstream:** **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)** by **1CatAI** — an independent
Tesla-V100 / sm\_70 vLLM fork (1.0.0, mid-2026) carrying AWQ-4bit sm\_70 kernels, a
FlashAttention-V100 backend, FP8 (`e5m2`) KV cache, and MTP speculative decoding for Qwen3.6. See the
upstream repository for its license.

## Why it's here
We weren't the only people refusing to call the V100 dead. 1CatAI shipped a serious, audited sm\_70
vLLM line in the same window we did. We **did not import any of their code** — our kernels and patches
are our own — but studying their fork sharpened our own decisions, and a parallel effort that arrives
at many of the same conclusions independently is worth crediting, not eliding.

## What we learned (with thanks)
Two of their design points map directly onto ours, and the contrast was instructive:
- **A shared speculative-decoding hazard.** Their notes and ours both hit the same class of bug: under
  CUDA-graph capture, padding desynchronises the speculative-token mask from the sequence-length
  metadata, yielding *high draft-acceptance with degenerate output*. Seeing it independently confirmed
  our standing rule — **never trust an acceptance metric without an exactness or output check** — which
  is exactly how our own MTP validation is gated.
- **A different FlashAttention integration tradeoff.** Their backend extracts paged KV into a dense
  per-decode copy and manages that cache at the application layer; the ai-bond path we adopted keeps the
  paged layout intact through the kernel boundary. Two legitimate answers to the same constraint — and
  comparing them is part of *why* we're confident in the one we chose.

Their FP8 MoE rides TurboMind `s884` W8A16 kernels (block-quantised, 128×128); our compressed-tensors
path trades that for per-layer block/channel flexibility. Different bets, same goal.

## Credit
1CatAI's work stands on its own — genuinely parallel, neither derived from ours nor a basis for it.
We're glad the "V100 in 2026" case is being made by more than one team, and we acknowledge their fork
as part of that shared effort.
