# flash-attention-v100: integration feedback & fixes (vLLM serving on 8×V100)

*Draft for ai-bond (D. Skryabin) — consolidated after full-fleet verification.
Local fixes are committed as one commit in our fork clone (`ccb6557`); happy to
send as PR(s) or let you re-derive — reproducers included for each.*

## Context

We integrated flash-attention-v100 as the prefill attention kernel for vLLM 0.21
serving on 8×V100-32GB (FP8-quantized LLMs, TP2–TP8), via a thin adapter that
calls the low-level `varlen_fwd` with vLLM's paged KV cache (`block_table` +
`seqused_k`). It is now validated end-to-end on **8 models** (dense, MoE,
GDN-hybrid; head_dim 128 & 256; GQA 2–12 q-heads/kv-head; fp16/fp8/int4 weights;
24k-token prompts; CUDA-graph decode coexistence; multi-user soak). Headline:
GLM-4.5-Air TTFT@24k 51.8s → 19.45s (2.66×); the kernel is ~8× the Triton
attention vLLM otherwise uses on Volta (18.4 vs 2.2 TFLOP/s @ D=128,
17.6 vs 2.2 @ D=256, seqlen 26k).

Three findings are upstream-relevant; the first is a correctness bug.

## 1. Paged KV: BLOCK_N must divide the page (D=128 tile straddles pages)

`include/forward.h` sets `BLOCK_N_128 = 160`. With paged KV
(`page_block_size % 256 == 0` per the host check), the varlen kernel resolves
ONE physical page per KV tile from the tile's start row and loads
`valid_kv_rows` linearly from that pointer (`fused_mha_forward_varlen.cu`,
paged branch). A 160-row tile starting at row 160 spans rows 160–319 — it
crosses the 256-row page boundary, and the linear load walks past the physical
page into whichever block is next in the pool. With a non-contiguous block
table (the normal case in vLLM) this reads another sequence's KV.

The real invariant is `page_block_size % BLOCK_N == 0`, which D=32/64/256
already satisfy (BLOCK_N 256/128/64) but D=128 (160) and D=16 (512) do not.

**Fix:** `BLOCK_N_128 160 → 128`. Correctness restored (cos=1.000000 vs fp32
reference up to 24k tokens, shuffled block tables); measured perf cost ≈ nil
(112.6 → 109.9 ms/layer at 26k in our runs).

**Reproducer:** paged varlen, D=128, two sequences with interleaved physical
blocks (seq0 → blocks [0,2], seq1 → [1,3]), garbage value (100.0) in all
unused slots → output contains the garbage (max_abs ≈ 100.7, cos ≈ 0.004);
after the fix, exact. (Script available: `fa_v100_paged_smoke.py`.)

## 2. CUDA 12.6 support: `__tanhf` is 12.8+/12.9-only; the pin can be 12.6

`setup.py` requires `torch.version.cuda >= 12.9`. The actual blocker is
`__tanhf` in the softcap path (`include/mat_mul.h`, 2 sites) — a host-only
function before CUDA ≈12.8, so device compilation fails on nvcc 12.6.
Replacing it with plain `tanhf` (valid on every CUDA; softcap-only path)
makes the kernels build and pass tests on **torch 2.11+cu126 / nvcc 12.6** —
the *other* Volta-supporting torch island besides 2.10+cu129 (PyTorch dropped
Volta from the cu128/cu129 wheels of 2.11, but kept it in cu126). Perf is
equal across both toolchains in our measurements.

**Fix:** `__tanhf → tanhf` (or guard with `#if CUDART_VERSION >= 12080`) and
relax the pin to 12.6. This doubles the toolchain surface Volta users can
build on.

## 3. Low-level `varlen_fwd` silently misreads strided Q (dense-row assumption)

The low-level kernel assumes query rows are dense `[T, H_Q*D]` (host checks
only `stride(-1)==1`; the kernel indexes `q_base + row * H_Q * D`). Callers
that pass a strided view — e.g. the `q` slice of a fused-QKV projection
(`qkv.split(...)`, row stride = full qkv width) — get silently wrong output at
full speed (cos ≈ 0.5 at GQA 12/1). The public python wrapper masks this via
`.contiguous()`, but direct low-level callers (as any serving integration will
be) hit it. Upstream FA2's varlen kernels take explicit q strides.

**Options:** document the dense-Q requirement loudly at the binding, add a
host-side stride check (`q.stride(0) == H_Q*D` → error or internal copy), or
pass q strides through to the kernel like upstream FA2. We currently densify
in our adapter (~0.2 ms at 24k — negligible vs the 5s attention).

**Reproducer:** allocate `[L, H_Q*D + 2*H_K*D]`, view the first `H_Q*D`
columns as `[L, H_Q, D]`, compare vs `.contiguous()` — cos ≈ 0.25–0.5.
(Script: `fa_v100_longseq_check.py --d 128 --hq 12 --hk 1`, qkv-split case.)

## Validation matrix (all with fixes 1–2 applied; vLLM 0.21, fp16 KV, paged block 256/768)

| Model | Arch | D | GQA/rank | FA TTFT gain | Notes |
|---|---|---|---|---|---|
| GLM-4.5-Air-FP8 (TP8) | dense-attn MoE | 128 | 12/1 | **2.66×** (51.8→19.4s @24k) | + 4-user soak, CUDA-graph decode untouched |
| Qwen2.5-7B (TP2) | dense | 128 | 14/2 | **3.3×** (6.2→1.9s @11k) | |
| DeepSeek-R1-Distill-32B (TP4) | dense | 128 | 10/2 | **2.7×** (10.4→3.8s @11k) | |
| Qwen3-30B-A3B / Coder (TP4) | MoE fp16 | 128 | 8/1 | ~1.04× | prefill dominated by fp16 MoE kernels, not attention |
| Qwen3.6-27B-FP8 (TP4) | GDN-hybrid | 256 | 6/1 | 1.4× | only full-attn layers route |
| Qwen3.6-35B-A3B-FP8 (TP4) | GDN-hybrid MoE | 256 | 4/1 | 1.05× | |
| Qwen3.5-122B-A10B (FP8 & Int4, TP8) | GDN-hybrid MoE | 256 | 4/1 | 1.3× | 12/48 full-attn layers; quant-orthogonal |
| Gemma-4-31B-FP8 (TP4) | full+sliding | 512/256 | — | n/a | full-attn layers are head_dim **512** (> dispatch max 256); sliding layers unvalidated window → we fall back |

Correctness gates passed on V100 silicon: your `test.py`; paged smoke
(interleaved tables, garbage tails, in-place out, LSE shape); long-seq sweep
512→24k (separate AND vLLM-interleaved `kv.unbind` strided cache views);
Sq<Sk bottom-right causal (q=1/64/4096 vs sk=8k/24k — your `seqlen_offset`
path is correct); mixed decode+prefill varlen batches; D=256 at GQA 4/1, 8/4,
4/2. Throughput: 8.0–8.4× vs vLLM's Triton attention at 26k on V100.

## Smaller notes

- `pip install` fights debian-managed `wheel`/`setuptools` (`RECORD file not
  found`) — `--no-deps` works; consider trimming install_requires.
- The `flash_attn` namespace shim is faithful to Tri Dao's API (nice!) but, on
  PYTHONPATH next to vLLM, vLLM's optional flash-attn probe half-succeeds
  (`import flash_attn` ok, `flash_attn.ops` missing) and aborts model init.
  Not your bug — but maybe worth a README note for vLLM users: expose only the
  compiled extension, not the shim.
- Head_dim 512 dispatch (for Gemma-4-class full-attn layers) would extend
  coverage, if smem budget allows on sm70.
