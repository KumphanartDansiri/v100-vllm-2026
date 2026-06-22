# Qwen3.6-35B-A3B

Qwen3.6-35B-A3B is a **35B-total / 3B-active MoE** (active/total ≈ 0.09 — deep in the "sparse-MoE
arch-fit" zone where FP8 W8A16 wins). The FP8 checkpoint is Qwen **block-FP8**. This is the **MoE
showcase**: FP8 beats FP16 at *every* concurrency on V100, the cleanest demonstration of why sparse
activation + half-the-weight-bytes is the V100 sweet spot.

- **What fits / best config:** TP4, cudagraph, `--dtype float16`, `--skip-mm-profiling`,
  `max-model-len=32768`. FP8 adds the plugin (`VLLM_V100_CT_FP8_RESIDENT=1` +
  `VLLM_V100_FP8_COALESCED_GEMV=1`); **FP16 MoE requires the Volta MoE patch** (Chapter 2) or it
  collapses to ~3 tok/s at 8 users. FP8 also fits at **TP2** on 2 cards for short context (≤8192).
- **Runs on both engines** (`Qwen3MoeForCausalLM`, stock 0.19 + 0.21, no transformers upgrade). The
  FP8 routed-expert compute is our own sm_70 kernels, so it's engine-invariant by construction; the
  FP16 path uses vLLM's fused-MoE (+ our Volta tune).


<!-- render:model:Qwen3.6-35B-A3B -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 15.44 | - | 0.74 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 67.6 | - | 1.86 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 15.56 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock(pre-moe-patch) | 3.16 | 24.93 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(heuristic) | 65.91 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(heuristic) | 20.98 | 137.2 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(tuned-json) | 65.85 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(tuned-json) | 22.8 | 173.92 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 74.92 | 74.92 | 14.28 | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 63.35 | 126.7 | - | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 58.76 | 235.04 | - | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 47.7 | 381.59 | - | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch | 55.93 | 55.93 | 12.65 | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | +moe_patch | 39.97 | 79.94 | - | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | +moe_patch | 26.77 | 107.08 | - | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch | 21.4 | 171.17 | - | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.19.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 89.92 | 89.92 | 14.17 | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 74.48 | 148.96 | - | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 68.85 | 275.4 | - | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 51.41 | 411.32 | - | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp16 | TP4 | 1 | +moe_patch | 63.3 | 63.3 | 12.66 | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.19.0/cu126 | fp16 | TP4 | 2 | +moe_patch | 44.01 | 88.02 | - | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.19.0/cu126 | fp16 | TP4 | 4 | +moe_patch | 34.06 | 136.24 | - | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.19.0/cu126 | fp16 | TP4 | 8 | +moe_patch | 28.25 | 225.98 | - | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 71.0 | 71.0 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 56.51 | 113.02 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 48.25 | 193.0 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 36.75 | 294.03 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
<!-- endrender -->

## How to read it
- **FP8 wins at every concurrency** — C1 **89.9 vs 63.3** (0.19) / 74.9 vs 55.9 (0.21), and the gap
  *holds* under load: C8/user **51.4 vs 28.3** (0.19), **47.7 vs 21.4** (0.21). Aggregate @C8 reaches
  **411 (0.19) / 382 (0.21) tok/s** FP8 vs 226 / 171 FP16. This is the headline MoE result: sparse
  activation keeps decode bandwidth-bound, where FP8's half-the-bytes pays off at *all* batch sizes —
  it never hits the dense CUDA-core wall, because each token only touches a few experts.
- **FP16 needs the Volta MoE patch.** Without it, FP16 MoE decode on V100 is BLOCK_K-mis-tuned and
  craters to ~3 tok/s @C8 (the "untuned-Volta-MoE" finding, Chapter 2). The numbers above are the
  *patched* FP16 — the fair comparison.
- **0.19 is faster than 0.21** at every point (FP8 90 vs 75 C1). Cold TTFT ~12–14 s either way.
- **TP2 ½-GPU option:** FP8 fits on 2 cards at short context — 71 tok/s C1, 294 agg @C8 — freeing the
  other 6 cards for another model.

## Caveats
- FP8 MoE is **Stable** (coherent, not run-to-run bit-identical at TP — expert/all-reduce ordering, as
  on stock FP16 MoE); FP16 is **Exact**.
- FP16 numbers are the **+moe_patch** path; stock FP16 MoE is far slower on Volta (Chapter 2).
- `max-model-len=32768`; the TP2 row is short-context (8192).
