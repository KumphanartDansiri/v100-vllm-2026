# Packet — 1Cat-vLLM (1CatAI)

**Upstream:** `1CatAI/1Cat-vLLM` — an independent Tesla-V100 / sm_70 vLLM fork (see the repo for its
license). **Channel:** GitHub (issue / discussion).
**Narrative:** [../../acknowledgements/onecat_vllm.md](../../acknowledgements/onecat_vllm.md).

> **This is not a code contribution.** 1CatAI shipped a serious, audited V100 line in parallel with
> ours; we imported none of their code and they need none of ours. What's here is *mutual technical
> acknowledgement* — two cross-checks that helped us, offered back in case they're useful to them.

## What we'd share (notes, not patches)
- **A shared speculative-decoding hazard.** Both efforts hit the same class of bug: under CUDA-graph
  capture, padding desynchronises the speculative-token mask from the sequence-length metadata, giving
  high draft-acceptance with degenerate output. Worth flagging as a general gotcha for any sm_70 MTP
  integration. Details: [comparison_notes.md](comparison_notes.md).
- **Two legitimate FlashAttention-integration shapes.** Their backend extracts paged KV into a dense
  per-decode copy; the ai-bond path we adopted keeps the paged layout intact through the kernel. Neither
  is "wrong" — documenting the trade is useful to anyone integrating FA on Volta.

## What we're asking upstream
Nothing to merge — just: if any of the cross-checks in [comparison_notes.md](comparison_notes.md) are
useful, take them; and corrections to our reading of their design are welcome. Credit is mutual.
