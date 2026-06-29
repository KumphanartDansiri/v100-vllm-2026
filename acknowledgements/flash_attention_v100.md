# flash-attention-v100 (ai-bond) — acknowledgement & feedback

> **Status: Final** — contribution status as of the freeze; see the upstream repo for the live state.

**Ready-to-send packet:** [../upstream_feedback/flash_attention_v100/](../upstream_feedback/flash_attention_v100/) — the patch, reproducers, and a paste-ready issue.

**Upstream:** **[`flash-attention-v100`](https://github.com/ai-bond/flash-attention-v100)** by
**D. Skryabin (ai-bond, @ai_bond007)** — a from-scratch
FlashAttention for Volta that emulates the missing `m16n16k16` tensor-core path with V100's `m8n8k4`
MMA, keeping Tri Dao's public API and `flash_attn_*_cuda` symbols intact. **License: BSD-3-Clause**
(© D. Skryabin).

## What it gave us
This kernel is the engine behind **every prefill / time-to-first-token win in this write-up**, and it
is the reason an **MLA** model runs on Volta at all:
- **MHA/GQA prefill** for the dense and standard-attention MoE models — the headline being **GLM-4.5-Air
  cold first-token 2.66×** faster (≈51.8 s → 19.4 s at 24k), with smaller gains across the fleet.
- **GLM-4.7-Flash MLA prefill** — V100 has no MLA-prefill backend in stock vLLM; the ai-bond dense
  varlen kernel (GLM's `qk=v=256` is an exact tile) is what unblocks the **first MLA model to generate
  on sm\_70** (numerically validated, cos ≈ 1.0 vs an fp32 reference).

We consume it through a thin adapter (decode stays on stock Triton); the kernel's design — *keep the
API, swap only the MMA internally* — is what made that integration a few hundred lines instead of a
fork.

## What we changed (and send back)
Three findings surfaced while wiring it into vLLM on a CUDA-12.6 toolchain, each with a reproducer:
1. **Paged-KV straddle** *(correctness)* — a tile constant let paged-KV tiles cross page boundaries;
   `BLOCK_N_128 160 → 128` fixes it.
2. **CUDA-12.6 build** *(portability)* — `__tanhf` → `tanhf` (two sites) and relaxing the torch CUDA pin
   from 12.9 to 12.6 roughly doubles the Volta toolchain surface the kernel builds on.
3. **Strided-Q read** *(documented, not patched)* — the low-level entry assumes dense query rows; vLLM's
   qkv-split view is strided, so we densify Q in our adapter. Upstream FA2 takes explicit Q strides;
   noted as an option for the kernel rather than changed in the fork.

These are packaged as a written report (3 fixes + reproducers + an 8-model validation matrix) and a
single committed patch on our fork clone, **offered to the maintainer** to take as a PR or re-derive.

## Credit
ai-bond is the original author and is credited as such in the fork's `LICENSE` and `AUTHORS`. This
project is a *consumer* of that work; the FlashAttention-V100 line of every result here is theirs.
