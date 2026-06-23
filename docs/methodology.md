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
- **Engines (dual, on purpose):** we carry **both vLLM 0.21.0 and 0.19.0**, **built from source** —
  the prebuilt pip wheels (CUDA ≥12.8 build) drop sm_70, but a **CUDA 12.6 source build** re-enables
  V100 with no source patch (the `<12.8` CMake branch still lists `7.0`). The base 0.21 and 0.19 images
  are cu126; a transformers-5 variant of the 0.19 image is cu128, used only where a newer model needs it
  (Gemma-4, GLM-4.7-Flash). We keep both engine versions because they win different things: **0.21**
  lands the newest model architectures first and has fewer compatibility gaps; **0.19** is frequently
  *faster* on decode and carries sm_70 more broadly. **Most FP8 rows exist on both** engines; the few
  compatibility exceptions (e.g. Gemma-4-26B FP8 is 0.21-only) are called out in the matrix and model
  pages. See Chapter 1.
- **Images / torch:**
  - **0.21:** `vllm-v100:vllm021-cu126` — torch 2.11.0+cu126, Triton 3.6.
  - **0.19 (base):** `vllm-v100-py312:vllm019-cu126` — torch 2.10+cu126.
  - **0.19 + transformers 5.x** (`vllm019-tf5`, cu128): required **only** by **Gemma-4** and
    **GLM-4.7-Flash** (their `model_type` needs transformers ≥ 5; every other model runs on stock 4.57).
- **Engine + toolchain are recorded per row** in the SSOT (`vllm_version`, `torch_cuda`); the Chapter 1
  matrix and the model pages show both engines side by side, so a "0.19 vs 0.21" gap is always visible.

## Metrics
- **decode tok/s (per-user):** steady-state output tokens/sec for one stream, measured *after*
  the first token (excludes prefill/TTFT). The number that matters for "how fast does it type."
  Reported as the **median** of repeated runs; warmup/outlier samples are dropped.
- **aggregate tok/s:** total output tokens/sec summed across *N concurrent* streams. Throughput,
  not latency. `aggregate ≈ per-user × N` only until the GPU saturates.
- **TTFT (s):** time to first token (prefill latency), single-stream, with chunked prefill **on** (the
  V100-standard serve — disabling it is a known crash-causer). Two numbers are reported: **cold** = a
  fresh, cache-cold request prefilling the full ~22.6k-token prompt (a guaranteed cache miss → the
  worst case); **prefix-cache-hit** = the same prompt when its prefix is already cached (repeated /
  shared context → the best case). Cold TTFT is prefill-bound; some FP8 checkpoints (notably the Qwen
  block-FP8 models) carry a large prefill penalty on V100 even where FP8 wins decode — an unoptimized
  FP8-prefill path, separate from the decode kernels (Chapter 5).
- **users:** number of concurrent request streams (1 = single-user latency; 8 = concurrency).

## Execution modes — DO NOT cross-compare
- **eager:** no CUDA graph; every op dispatched from Python per step. Carries large per-step
  launch overhead → **eager decode runs ~5–10× slower than cudagraph** on the same setup (the measured
  pairs in Chapter 3) and is **NOT a serving baseline.** We report eager only to isolate kernel
  behavior, never as headline serving speed.
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
  fp16 for the matmul — V100 has no native FP8). `config=fp8-plugin+coalesced`. Chapter 5.
- **int4:** GPTQ-Int4 (122B only, where FP16 won't fit) — vLLM's Volta-compatible GPTQ path.
- **config column** values: `stock`, `stock(pre-moe-patch)`, `+moe_patch(heuristic)`,
  `+moe_patch(tuned-json)`. The MoE patch affects **fp16/bf16 MoE only**; dense and our fp8 path
  are unaffected (fp8 uses its own kernels).

## "Fits" vs "serves coherently" vs measured speed — three different bars
1. **Fits:** the model loads into VRAM at a given TP without OOM. Sets the **minimum TP** per
   model (e.g. 122B-FP8 needs **TP8** on 32GB cards; 35B-FP16 needs **≥TP4**).
2. **Serves coherently:** it generates correct, non-repetitive text (exactness label below).
3. **Speed:** the tok/s, only meaningful once 1 and 2 hold.

A **feasible TP set is per-model** (bounded by fit) — we do NOT report a fixed TP=1,2,4,8 for
every model; each model page lists only the TP sizes that fit.

## The correctness suite — the 5 standard tests
Every model × precision × engine cell runs the same fixed five-question probe. **Q1 is the exactness +
decode anchor** — run **×5** at a long window for self-stability and decode mean/variance; **Q2–Q5** are
one-shot faithfulness/coherence probes across four categories. The verbatim prompts and per-question
specs live in the companion repo's `tools/ch1_reliability_bench.sh` (`PROMPT_SPECS` + `prompt_text()`);
the per-model revisit benches reuse the same five.

| # | Category | Reps | max_tokens | Question |
|---|---|---:|---:|---|
| **Q1** | **Exactness / self-stability** | **5** | 4096 | *"Write a detailed, multi-section essay on the history, geography, economy, and culture of France. Use clear subsections with headings and develop each at length."* |
| Q2 | Factual | 1 | 1024 | *"Explain, step by step, how a transformer neural network processes a sentence — from tokenization and embeddings through self-attention and feed-forward layers to the output distribution. Be precise and thorough."* |
| Q3 | Reasoning | 1 | 1024 | *"A train leaves city A at 9:00 traveling 60 km/h toward city B, 280 km away. A second train leaves city B at 9:30 traveling 80 km/h toward city A on the same line. At what clock time do they meet, and how far from city A? Show every step of your reasoning, then give the final answer."* |
| Q4 | Structure | 1 | 1024 | *"Summarize the main causes of World War I as exactly five bullet points. Each bullet must be a single sentence. Do not add any text before or after the five bullets."* |
| Q5 | Code | 1 | 1024 | *"Implement a thread-safe LRU (least-recently-used) cache in Python supporting get(key) and put(key, value) with a fixed capacity. Use only the standard library. Include docstrings and a set of pytest unit tests covering eviction, update, and concurrency."* |

**Shared request parameters:** `temperature=0` (greedy; a reasoning model whose `generation_config`
wants sampling is run at its configured temperature, which makes Q1 self-stability non-deterministic by
design), `seed=1234`, `ignore_eos=true`, streaming. `ignore_eos` forces the full `max_tokens` window so
decode speed is measured over a fixed length — which is *why* a reasoning model can show mild tail
repetition past its natural stop (a measurement choice, not a model failure).

**How Q1 becomes the exactness label:** the five greedy outputs are hashed — **1 distinct hash = Exact,
>1 = Stable** — and the same runs give the reported decode mean/stdev.

## Exactness labels (for correctness, not speed)
- **Exact:** run-to-run deterministic (identical sha across repeats). The strongest claim.
- **Stable:** coherent + low repetition, but not bit-identical across precision/config (expected
  when comparing fp8-vs-fp16 — different numerics, not a bug).
- **Coherent:** readable, on-task output; not exactness-checked.
- **Fail:** OOM, crash, or degenerate output (repetition/garbage).
Note: FP8-vs-FP16 comparisons top out at **Stable** by construction (different math), never Exact.

## Caveats (stated up front — see README "Known limitations")
- V100 requires a **source build on CUDA 12.6**; pip wheels won't work.
- The **FP8 plugin is a local/custom** sm_70 kernel, not upstream vLLM.
- Some **TP cells are infeasible** (model too big at low TP) — shown as absent, not zero.
- **Vision-encoder** startup profiling can be skipped (`--skip-mm-profiling`) for text-only serving.
- The **MoE fix** is for fp16/bf16 MoE; it does not change fp8 numbers.
