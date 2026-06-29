# vLLM — acknowledgement & feedback

> **Status: Final** — contribution status as of the freeze.

**Ready-to-send packet:** [../upstream_feedback/vllm/](../upstream_feedback/vllm/) — the diagnostic issue + the drop-in V100 config JSONs.

**Upstream:** **[vLLM](https://github.com/vllm-project/vllm)** (vllm-project).
**License: Apache-2.0.** We run **both** the **0.19.x** and
**0.21.x** lines, source-built on a CUDA-12.6 toolchain.

## What it gave us
The entire write-up runs on vLLM. Everything else — the FP8 plugin, the MoE fix, the FlashAttention
and MLA hooks — is a *layer on top of an unmodified vLLM source build*:
- **The sm\_70 unlock is vLLM's own source.** Its CMake `<12.8` branch still lists `7.0`, so building
  from source on CUDA 12.6 re-enables Volta with **zero source patches** (Chapter 1). The prebuilt
  wheels dropped it; the source never did.
- **Two engines, on purpose.** 0.21 carries the newest model architectures; 0.19 is usually faster on
  decode. Both build and serve on V100 from source.
- **The quantization & MoE interfaces** (compressed-tensors W8A16, `fused_moe`, the cudagraph machinery)
  are the seams our plugin attaches to — clean enough that the FP8 path ported from 0.18 → 0.21 with
  essentially no code change.

## What we changed (and send back)
We do **not** patch vLLM's source — the plugin is an out-of-tree layer, and V100 support is (by the
project's stated policy) out of scope for the wheels. So the give-back is a **finding plus data**, not
a source PR:
- **The FP16-MoE `BLOCK_K=128` diagnostic (Chapter 2).** Stock fused-MoE's decode branch picks
  `BLOCK_SIZE_K=128`; on Volta (no `cp.async`, 96 KB shared memory) that register-spills and makes a
  sparse MoE *slower than a dense model of similar class* — a **4–9× e2e cost** from one default, not
  from the hardware. We can contribute this as an issue (framed honestly as "is `BLOCK_K=128` the right
  decode default even on `cp.async` architectures?") plus **two autotuned V100 config JSONs**
  (`E=256,N=128` and `E=128,N=176`, Tesla-V100) — the data-only kind of contribution vLLM already ships
  hundreds of.

The heuristic code itself targets sm\_70, which the project doesn't carry; that part is offered to
[aphrodite-engine](aphrodite.md) instead, where broad-arch support is in scope.

## Credit
vLLM is the foundation of this work. Where a number says `stock-vllm`, it is unmodified vLLM doing the
work; where it says `fp8-plugin` or `+moe_patch`, that's our layer on top — and the separation is kept
explicit precisely so vLLM gets credit for what vLLM does.
