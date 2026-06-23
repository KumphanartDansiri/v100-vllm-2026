# Packet — aphrodite-engine

**Upstream:** `dphnAI/aphrodite-engine` (formerly PygmalionAI; maintainer AlpinDale). **License:** AGPL-3.0.
**Channel:** GitHub PR(s).
**Narrative:** [../../acknowledgements/aphrodite.md](../../acknowledgements/aphrodite.md).

> **Status: prepared but parked.** We made vLLM 0.21 the primary frozen engine (it needs no arch patch
> on CUDA 12.6). aphrodite is the "upstream-and-ride" alternative — broad-arch culture makes a clean
> Volta contribution land more naturally here than on vLLM. The materials are kept ready in case the
> project (or its users) want them.

## What we observed / why it matters
Two contributions, both in scope for aphrodite where they aren't for vLLM:
1. **sm_70 is droppable-but-recoverable.** Stock aphrodite 0.21 drops Volta like stock vLLM 0.21; a
   small, clean commit series re-enables it and serves modern models on V100 (gemma-4-31B ≈ 29 tok/s,
   vLLM-0.19 parity). See [sm70_build_rewire.md](sm70_build_rewire.md).
2. **The FP16-MoE `BLOCK_K` fix belongs here as code.** Unlike vLLM, sm_70 is in scope, so the decode
   heuristic + V100 config JSONs can go in as a real PR. See [moe_patch_plan.md](moe_patch_plan.md).

## Environment
8×V100-SXM2-32GB, aphrodite 0.21, CUDA 12.6 / cu126 torch. Build + serve validated June 2026.

## Reproducer / artifacts
The 4-commit rewire + a cu126 Dockerfile diff + build/measure logs are preserved at
`/home/kumphanartd/aphrodite_salvage_archive/v100-sm70-patches/` (implementation side, not copied into
this public repo — see [sm70_build_rewire.md](sm70_build_rewire.md) for the commit list and how to apply).

## Proposed fix / patch status
- Build rewire: **clean, mergeable today** (4 commits).
- MoE heuristic + configs: **ready** (the configs are the same JSONs as the [vLLM packet](../vllm/configs/)).
- compressed-tensors FP8 sm_70: **plan only** — lowering aphrodite's sm_75-gated
  `compressed_tensors_w8a16_fp8` to 70 + a CUDA-core dequant fallback; the *kernel* still needs
  extracting from our plugin into a standalone form before that half is PR-ready.

## What we're asking upstream
If broad V100 support is wanted: take the build rewire and the MoE PR. We'll credit the work as building
on aphrodite's existing sm_70 approach.
