# gemma-4-26B-A4B-it

gemma-4-26B-A4B-it is a **26B-total / 4B-active MoE** (`Gemma4ForConditionalGeneration`). The FP8
checkpoint is `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`. Like all Gemma-4 it needs **transformers 5.x**
(use the `vllm019-tf5` image on 0.19). It is the model that **most clearly justifies carrying vLLM
0.21**: its FP8 MoE runs on 0.21 but hits a stock `gemma4.py` MoE error on 0.19.

- **What fits / best config:** TP4, cudagraph, `--dtype float16`, `--skip-mm-profiling`,
  `--max-num-batched-tokens ≥ 2496` (Gemma-4 vision-tower requirement on 0.21), `max-model-len=32768`.
  FP8 adds the plugin; **FP16 MoE requires the Volta MoE patch** (Chapter 2). FP8 also fits at **TP2**
  (2 cards, short context ≤8192).
- **Engine split — read carefully:** **FP16 runs on both** engines (tf5); **FP8 runs only on 0.21**
  (the `gemma4.py` MoE path errors on stock 0.19). This is the concrete case behind "0.21 earns its
  place on compatibility" in Chapter 1.


<!-- render:model:gemma-4-26B-A4B-it -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 10.2 | - | 0.3 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 46.48 | - | 0.18 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 10.91 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock(pre-moe-patch) | 3.58 | 28.3 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(heuristic) | 43.66 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(heuristic) | 19.1 | 145.15 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(tuned-json) | 43.71 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(tuned-json) | 20.23 | 155.94 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 72.85 | 72.85 | 50.64 | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 59.26 | 118.52 | - | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 48.4 | 193.6 | - | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 33.51 | 268.05 | - | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch | 44.39 | 44.39 | 53.60 | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | +moe_patch | 35.53 | 71.06 | - | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | +moe_patch | 23.16 | 92.64 | - | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch | 21.61 | 172.86 | - | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.19.0/cu128 | fp16 | TP4 | 1 | +moe_patch | 39.47 | 39.47 | 53.34 | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.19.0/cu128 | fp16 | TP4 | 2 | +moe_patch | 32.98 | 65.96 | - | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.19.0/cu128 | fp16 | TP4 | 4 | +moe_patch | 28.35 | 113.4 | - | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.19.0/cu128 | fp16 | TP4 | 8 | +moe_patch | 20.41 | 163.26 | - | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 60.15 | 60.15 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 46.37 | 92.74 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 32.6 | 130.4 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 17.54 | 140.34 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
<!-- endrender -->

## How to read it
- **FP8 ≫ FP16 (MoE arch-fit):** on 0.21, FP8 **72.9 vs 44.4 tok/s C1** and it wins every concurrency
  (C8 agg **268 vs 173**). Sparse activation again sidesteps the dense CUDA-core wall.
- **Gemma-4-26B is the rare model where 0.21 FP16 *beats* 0.19 FP16** (44.4 vs 39.5 C1) — against the
  fleet pattern — though FP8 (0.21-only) is the number you'd actually serve.
- **Cold TTFT ~50–54 s** — higher than the other MoEs because of Gemma-4's vision tower + untuned
  prefill (decode is the story; `--skip-mm-profiling` keeps cold start from being far worse).
- **TP2 ½-GPU option:** FP8 fits on 2 cards at short context — 60 tok/s C1, 140 agg @C8.

## Caveats
- **FP8 is 0.21-only** here (0.19 `gemma4.py` MoE error); FP16 is dual-engine but needs the Volta MoE
  patch + tf5.
- Both FP16 and FP8 are **Exact** (quality=pass); all categories coherent.
- `max-model-len=32768`; the TP2 row is short-context (8192).
