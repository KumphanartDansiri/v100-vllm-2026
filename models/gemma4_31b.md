# gemma-4-31B-it

Dense 31B (`Gemma4ForConditionalGeneration`, 60 layers, no MoE — active = total). Measured on both
vLLM 0.21 (`vllm021-cu126`, transformers 5.10) and vLLM 0.19 (`vllm019-tf5`, transformers 5.5),
TP4, cudagraph, `--skip-mm-profiling`. Decode = steady-state 256-tok streaming (TTFT-immune).

- **What fits / feasible TP:** TP4 on 4×V100-32GB. FP16 weights ~62 GB → ~15.5 GB/GPU; FP8 ~31 GB
  → ~7.8 GB/GPU. FP8 halves the weight bytes per card → **~1.55× more KV headroom**.
- **Best TP / flags:** `--tensor-parallel-size 4 --dtype float16 --skip-mm-profiling
  --max-num-batched-tokens 8192 --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
  FP8 adds the plugin (`VLLM_V100_CT_FP8_RESIDENT=1` + `VLLM_V100_FP8_COALESCED_GEMV=1`).
- **Transformers requirement:** Gemma-4 needs **transformers ≥ 5.x on *both* engines** (stock 0.19 +
  tf 4.57 can't parse `model_type=gemma4` — vLLM 0.19's own `gemma4.py` imports
  `transformers.models.gemma4`, added in tf 5.x). Use the `vllm019-tf5` image for 0.19.
- **`--max-num-batched-tokens 8192` is required on vLLM 0.21:** Gemma-4's vision tower has
  multimodal-bidirectional attention, so 0.21 force-disables chunked-MM-input and then requires
  `max_num_batched_tokens ≥ max_tokens_per_mm_item (2496)`; the 2048 default fails at startup. 0.19
  doesn't enforce this. (Text-only decode is insensitive to the value, so it doesn't bias the comparison.)
- **FP16 baseline vs FP8 — FP8 now *wins* single-stream:** **35.3 vs 26.7 tok/s at C1 (FP8 +32%)**, at
  half the weight footprint + more KV headroom. This **inverts** the old "dense FP8 = memory-only" rule
  at low concurrency: a branchless E4M3 dequant (the FP8-dequant breakthrough) lifted dense FP8 above
  FP16 for single-/low-user decode. FP8's value here is now **both** memory *and* low-user speed. Both
  load and generate coherent text.
- **single / multi-user tok/s:** see the matrix table below. **FP8 wins C1–C2** (35.3 / 28.3 vs FP16
  26.7 / 24.1); the curves **cross at ~C4** (FP8 19.7 vs FP16 21.4); **FP16 takes C4–C8** (C8 agg
  **137** vs FP8 **101**).
- **Why FP16 reclaims high concurrency:** dense decode streams the *entire* weight set every token, and
  our FP8 dequant runs on **CUDA cores** while FP16 decode reaches cuBLAS **tensor cores** that scale
  better with batch. So at C4–C8 the per-token compute wall favours FP16; sparse-MoE models sidestep it
  via per-token sparsity (see GLM-4.5-Air / Qwen3.6-35B-A3B, where FP8 wins at *every* concurrency). A
  tensor-core / WMMA FP8 decode kernel would close the dense gap (future work, no timeline). (Kernel:
  `src/fp8_w8a16_sm70/fp8_dequant.cu`.)
- **Lower-TP option (the memory play):** FP8 also fits at **TP2** (½ the cards, short context ≤ 8192) —
  23.1 tok/s C1, 55.7 agg @C8 — so you can serve gemma-4-31B on **2×V100 instead of 4**, freeing cards
  for other models. FP16 needs TP4.
- **0.19 vs 0.21:** within noise on both precisions (FP16 26.73 = 26.73 @C1; FP8 35.28 vs 35.23 @C1;
  C8 agg 136.7 vs 136.8 FP16 / 101.5 vs 101.8 FP8) — a **portability** result. FP8 is our sm_70 kernel,
  so it's engine-invariant by construction.
- **Cold TTFT:** dense gemma-4 prefill is the slow part on Volta — **~185–196 s** cold monolithic at
  TP4 (untuned-prefill state, comparable on both engines and precisions). Decode is the story here;
  prefill tuning is future work.
- **Exactness:** both FP16 and FP8 are **Exact** (deterministic greedy), quality=pass; all 5-test
  categories coherent on both engines.

<!-- render:model:gemma-4-31B-it -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock-vllm | 17.61 | - | 0.16 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 17.53 | - | 0.45 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 35.28 | 35.28 | 196.03 | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 28.25 | 56.5 | - | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 19.65 | 78.6 | - | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 12.68 | 101.47 | - | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock-vllm | 26.73 | 26.73 | 185.82 | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | stock-vllm | 24.07 | 48.14 | - | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | stock-vllm | 21.43 | 85.72 | - | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock-vllm | 17.08 | 136.66 | - | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.19.0/cu128 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 35.23 | 35.23 | 193.65 | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 28.19 | 56.38 | - | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 19.67 | 78.68 | - | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 12.72 | 101.78 | - | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | fp16 | TP4 | 1 | stock-vllm | 26.73 | 26.73 | 184.67 | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.19.0/cu128 | fp16 | TP4 | 2 | stock-vllm | 24.17 | 48.34 | - | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.19.0/cu128 | fp16 | TP4 | 4 | stock-vllm | 21.44 | 85.76 | - | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.19.0/cu128 | fp16 | TP4 | 8 | stock-vllm | 17.1 | 136.79 | - | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 23.07 | 23.07 | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 17.34 | 34.68 | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 11.09 | 44.36 | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 6.97 | 55.74 | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
<!-- endrender -->
