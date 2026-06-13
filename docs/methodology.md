# Methodology & definitions

Every number in this project comes from `data/benchmark_matrix.csv` (the single source of
truth), extracted from raw result files by `scripts/build_matrix_from_results.py`. Each row
carries a `result_path` back to the evidence. Read these definitions before comparing any
two numbers — most "V100 is slow" claims online compare incomparable things.

> **`result_path` values are relative to the companion code repo `fp8-w8a16-sm70/`**, where the
> raw serve logs and summaries live (this repo holds the curated evidence; that repo holds the
> kernels, the plugin, and the full logs). They are textual references, not links within this repo.

## Hardware & stack
- **GPU:** Tesla V100-SXM2-32GB (sm_70, Volta), 8× in one box, NVLink.
- **Engine:** vLLM 0.21.0 **built from source on a CUDA 12.6 toolchain** (image
  `vllm-v100:vllm021-cu126`). The prebuilt pip wheels (CUDA ≥12.8 build) drop sm_70; the
  source's `<12.8` CMake branch still lists `7.0`, so a cu126 build re-enables V100 with no
  source patch. See Chapter 1.
- **Torch:** 2.11.0+cu126. Triton 3.6.

## Metrics
- **decode tok/s (per-user):** steady-state output tokens/sec for one stream, measured *after*
  the first token (excludes prefill/TTFT). The number that matters for "how fast does it type."
  Reported as the **median** of repeated runs; warmup/outlier samples are dropped.
- **aggregate tok/s:** total output tokens/sec summed across *N concurrent* streams. Throughput,
  not latency. `aggregate ≈ per-user × N` only until the GPU saturates.
- **TTFT (s):** time to first token (prefill latency). Steady-state **minimum** is reported (the
  first request of a fresh server includes one-time warmup/cudagraph-capture cost — not TTFT).
- **users:** number of concurrent request streams (1 = single-user latency; 8 = concurrency).

## Execution modes — DO NOT cross-compare
- **eager:** no CUDA graph; every op dispatched from Python per step. Carries large per-step
  launch overhead → **eager decode numbers are 3–5× lower than cudagraph and are NOT a serving
  baseline.** We report eager only to isolate kernel behavior, never as headline serving speed.
- **cudagraph** (`cudagraph_mode=FULL_DECODE_ONLY`): the decode loop is captured into a CUDA
  graph and replayed without Python overhead. **This is the practical serving baseline** and
  what all headline tok/s use unless stated.
- **MTP** (multi-token prediction / speculative decode): a draft head proposes k tokens/step,
  verified in one pass. Changes the tok/s *definition* (tokens/step varies with accept rate), so
  MTP numbers are kept in their own chapter and never mixed into base-decode comparisons.

## Precision / config
- **fp16 / bf16:** full-precision weights (the reference). For MoE models on V100, see the
  `config` column — `stock(pre-moe-patch)` is the un-tuned vLLM default (pathologically slow,
  Chapter 2); `+moe_patch` is with our Volta fused-MoE fix.
- **fp8:** our custom W8A16 sm_70 plugin (FP8 weights resident in HBM, dequantized in-kernel to
  fp16 for the matmul — V100 has no native FP8). `flags=fp8-plugin+coalesced`. Chapter 5.
- **int4:** GPTQ-Int4 (122B only, where FP16 won't fit) — vLLM's Volta-compatible GPTQ path.
- **config column** values: `stock`, `stock(pre-moe-patch)`, `+moe_patch(heuristic)`,
  `+moe_patch(tuned-json)`. The MoE patch affects **fp16/bf16 MoE only**; dense and our fp8 path
  are unaffected (fp8 uses its own kernels).

## "Fits" vs "serves coherently" vs measured speed — three different bars
1. **Fits:** the model loads into VRAM at a given TP without OOM. Sets the **minimum TP** per
   model (e.g. 122B-FP8 needs ≥TP4 on 32GB cards; 35B-FP16 needs ≥TP4).
2. **Serves coherently:** it generates correct, non-repetitive text (exactness label below).
3. **Speed:** the tok/s, only meaningful once 1 and 2 hold.

A **feasible TP set is per-model** (bounded by fit) — we do NOT report a fixed TP=1,2,4,8 for
every model; each model page lists only the TP sizes that fit.

## Exactness labels (for correctness, not speed)
- **Exact:** run-to-run deterministic (identical sha across repeats). The strongest claim.
- **Stable:** coherent + low repetition, but not bit-identical across precision/config (expected
  when comparing fp8-vs-fp16 — different numerics, not a bug).
- **Coherent:** readable, on-task output; not exactness-checked.
- **Fail:** OOM, crash, or degenerate output (repetition/garbage).
Note: FP8-vs-FP16 comparisons top out at **Stable** by construction (different math), never Exact.

## Caveats (stated up front — see README "Known limitations")
- V100 requires a **source build on CUDA 12.6** (or 12.9); pip wheels won't work.
- The **FP8 plugin is a local/custom** sm_70 kernel, not upstream vLLM.
- Some **TP cells are infeasible** (model too big at low TP) — shown as absent, not zero.
- **Vision-encoder** startup profiling can be skipped (`--skip-mm-profiling`) for text-only serving.
- The **MoE fix** is for fp16/bf16 MoE; it does not change fp8 numbers.
