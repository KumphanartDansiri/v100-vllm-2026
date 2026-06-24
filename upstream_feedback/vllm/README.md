# Packet — vLLM

**Upstream:** `vllm-project/vllm`. **License:** Apache-2.0.
**Channel:** a GitHub **issue** (the diagnostic) + a **data/config PR** (the two JSONs).
**Narrative:** [../../acknowledgements/vllm.md](../../acknowledgements/vllm.md).

## Maintainer quick path
*If you maintain vLLM and only have five minutes:*
1. **Read** — [moe_block_k_issue.md](moe_block_k_issue.md): a measured config finding, **not** a V100-support ask.
2. **Artifacts** — [configs/](configs/): two drop-in V100 MoE config JSONs (exact `get_moe_configs` names) + provenance.
3. **Reproducer** — `tools/moe_stages_ab_vllm021.sh` in the fp8-w8a16-sm70 repo (e2e A/B: stock vs fix).
4. **Ask** — the config JSONs as a data PR; the diagnostic as an issue *if* the small-M default question is worth a look.
5. **Status** — prepared, not yet sent.

## What we observed
On V100 (sm_70), stock **fp16/bf16 fused-MoE decode is slower than a same-class dense model** — backwards
for a sparse MoE (Qwen3.6-35B-A3B 15.6 tok/s vs 27B-dense 37–41; gemma-4-26B-A4B 10.9 vs 31B-dense 17.6).

## Why it matters
Root cause is one default: `fused_moe.py::get_default_config` has no sm_70 case, and its decode branch
(M ≤ 64) picks `BLOCK_SIZE_K=128`. On Volta (no `cp.async`, 96 KB smem) that register-spills Triton;
kernel time scales 64→632 µs, 128→1450 µs at M=1. Fixing the tile gives **4–9× e2e**, output
bit-identical. (`num_stages` is *not* the lever — a 4/3/2 sweep was flat; details in the issue draft.)

## Environment
8×V100-SXM2-32GB, vLLM 0.21.0 + CUDA 12.6, Triton 3.6, TP4, cudagraph FULL_DECODE_ONLY, ns=8.

## Reproducer
The implementation repo's `tools/moe_stages_ab_vllm021.sh` (e2e A/B: stock vs heuristic vs tuned JSON);
the tuner is `tools/moe_volta_tune.py`. Result paths and the full A/B table are cited in
[moe_block_k_issue.md](moe_block_k_issue.md).

## Proposed fix / patch status
Two parts, kept separable on purpose:
1. **Diagnostic** — filed as an *issue/question* (not an sm_70-support PR, which is out of policy):
   "is `BLOCK_K=128` the right small-M decode default even on `cp.async` arches?" Full draft in
   [moe_block_k_issue.md](moe_block_k_issue.md).
2. **Data** — two autotuned V100 config JSONs in [configs/](configs/), the kind of device-config
   contribution vLLM already ships hundreds of. The sm_70 *heuristic code* targets an arch vLLM doesn't
   carry, so it's offered to [aphrodite](../aphrodite/) instead.

## What we're asking upstream
Consider the two config JSONs as a data contribution; and consider the diagnostic on its own merit —
even if sm_70 is out of scope, the small-M default question may matter cross-arch. Reference impl
(env-gated monkey-patch) lives in fp8-w8a16-sm70 `vllm_serve.py::_patch_volta_moe_default_config`.
