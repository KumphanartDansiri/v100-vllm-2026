# Packet — flash-attention-v100 (ai-bond)

**Upstream:** `flash-attention-v100` by D. Skryabin (ai-bond). **License:** BSD-3-Clause.
**Channel:** GitHub `github.com/ai-bond/flash-attention-v100` (PR or issue) · Telegram @ai_bond007.
**Copy-paste submission:** [github_issue.md](github_issue.md) (self-contained, diffs inlined).
**Narrative:** [../../acknowledgements/flash_attention_v100.md](../../acknowledgements/flash_attention_v100.md).

## Maintainer quick path
*If you maintain flash-attention-v100 and only have five minutes:*
1. **Read** — [github_issue.md](github_issue.md): paste-ready, the two fixes inlined as diffs.
2. **Artifacts** — [ccb6557.patch](ccb6557.patch) (3 files, +4/−4; `git am` it).
3. **Reproducers** — [reproducers.md](reproducers.md): the paged-straddle smoke + the strided-Q check.
4. **Ask** — take the patch, re-derive it, or tell us the issue/PR format you'd prefer.
5. **Status** — prepared, not yet sent.

## What we observed
Integrating the kernel as the prefill backend for vLLM on 8×V100 (paged KV via the low-level
`varlen_fwd`) surfaced three things — one correctness bug, one build-portability fix, one caller gotcha:
1. **Paged-KV straddle (correctness):** `BLOCK_N_128 = 160` lets a D=128 KV tile cross the 256-row page
   boundary and read another sequence's KV under a non-contiguous block table.
2. **CUDA-12.6 build:** the `setup.py` pin (`cuda >= 12.9`) and `__tanhf` in the softcap path block
   nvcc 12.6 — the *other* Volta-supporting torch island (2.11+cu126).
3. **Strided-Q (documented, not patched):** the low-level entry assumes dense `[T, H_Q*D]` query rows;
   a fused-QKV split view is silently misread at full speed.

## Why it matters
Fix #1 is a silent correctness bug for any paged-KV serving integration (the common case). Fix #2
roughly doubles the Volta toolchain surface the kernel builds on. #3 is a sharp edge every low-level
serving caller will hit — worth a loud doc note or a host-side stride check.

## Environment
8×V100-SXM2-32GB (sm_70), vLLM 0.21, torch 2.11+cu126 / nvcc 12.6, fp16 KV, paged block 256/768.
Validated end-to-end across 8 models (dense / MoE / GDN-hybrid; D=128 & 256). Headline: GLM-4.5-Air
TTFT@24k **51.8 s → 19.45 s (2.66×)**.

## Reproducer
See [reproducers.md](reproducers.md) — the paged straddle smoke (garbage-tail interleaved block tables)
and the strided-Q long-seq check, with commands and expected before/after.

## Proposed fix / patch status
All three are in **[patches.md](patches.md)**; fixes #1 and #2 are committed in our fork clone as
**`ccb6557`** (see [ccb6557.patch](ccb6557.patch) — 3 files, 4 lines). #3 is documented with options
(loud doc note / host stride check / pass strides like upstream FA2), not patched.

## What we're asking upstream
Take `ccb6557` as a PR (or re-derive from the reproducers) for #1 and #2; decide #3's preferred shape.
The full consolidated write-up — context, the validation matrix, and smaller notes (pip `--no-deps`,
the vLLM flash-attn probe gotcha, a head_dim-512 dispatch request for Gemma-4) — is in
**[report.md](report.md)**.
