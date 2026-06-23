# 1Cat-vLLM — comparison notes

Technical cross-checks between two independent V100/sm_70 vLLM efforts. Stated as observations and
trade-offs, not criticism — both lines arrived at workable answers.

## 1. Speculative decode under CUDA-graph capture (a shared hazard)
**Symptom (both efforts saw this class):** very high draft-acceptance (≈97%) with degenerate output
(repetition loops, e.g. `" the the the…"`), while throughput looks normal.
**Mechanism:** when graph capture pads a verifier batch (e.g. 15 → 20 tokens), the speculative-token
mask computed on the padded tensor desyncs from the `query_start_loc` / sequence-length metadata, so the
verifier compares the wrong KV positions and accepts almost everything.
**Takeaway we both converge on:** *never trust an acceptance metric alone* — gate MTP on token-level
exactness or output inspection. Sync the CPU/GPU spec masks carefully under capture (use the synced GPU
tensor for `query_lens`, and handle non-contiguous non-spec indices explicitly).

## 2. FlashAttention integration shape (a trade, not a verdict)
**Their approach:** the V100 FA backend extracts the paged KV cache into a contiguous per-decode copy
(an app-level cache pool with validity checks), then calls a `*_paged` API. Simpler kernel boundary; pays
an allocation + copy per decode batch and carries cache-management logic in the backend.
**Our approach (ai-bond):** the kernel consumes vLLM's paged layout directly via `block_table`; no dense
copy, no app-level cache. Thinner integration; relies on the kernel honouring the paged layout.
**Why it's worth recording:** both are valid responses to "V100 has no native paged-FA." The contrast is
exactly *why* we're confident in the no-copy path — and a useful data point for anyone choosing.

## 3. FP8 MoE kernel choice (different bets)
Theirs rides TurboMind `s884` W8A16 kernels, block-quantised at 128×128 (no channel-wise). Ours uses a
compressed-tensors path with per-layer block/channel flexibility. Same goal (FP8 MoE on Volta), different
constraints — neither dominates; the flexibility-vs-throughput trade depends on the checkpoint.

*If 1CatAI reads this and any characterisation is off, corrections welcome — these notes are from
studying the public repo, not from their authors.*
