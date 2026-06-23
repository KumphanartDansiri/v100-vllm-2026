# Qwen3.6-27B (dense — FP16 + FP8) — V100 model-family page

> **Status: DRAFT** — provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)). Digest tables render from `data/benchmark_matrix.csv` (perf_v2-frozen rows only); the exhaustive raw SSOT table is at the bottom.

A dense 27B. The V100 story is **memory fit vs TP** plus a post-breakthrough twist: FP8 buys a lower
TP floor **and** — since the branchless E4M3 dequant — *faster* low-user decode than FP16 at the same
TP. FP16 reclaims the 8-user aggregate (the dense CUDA-core-vs-tensor-core wall, Chapter 5).

## Family / checkpoints
- `Qwen/Qwen3.6-27B` — FP16 baseline (stock vLLM).
- `Qwen/Qwen3.6-27B-FP8` — FP8 plugin path.
- **Compatibility:** runs on vLLM **0.19 and 0.21 stock** (no transformers-5 upgrade); dense, so the
  MoE patch doesn't apply; the FP8 plugin works on **both engines**. (Chapter 6 matrix.)

## Fit / compatibility
- **FP16 (~52 GB):** needs **TP ≥ 2** (TP1 ≫ 32 GB).
- **FP8 (~29 GB, resident):** feasible **TP{2, 4, 8}**; **TP1 OOMs** (resident weights + KV + CUDA/NCCL
  context don't fit one card). Proof it's resident, not dequantized back to 54 GB: KV headroom scales
  1/TP — **183k tok @TP2 → 745k @TP8**.
- **Best engine:** 0.19 for decode throughput (faster at every same-TP point); 0.21 also works.

## Single-user deployment summary
*What one stream gets at C1, per engine — the precision/TP choice for a solo user or small lab.*

<!-- render:single_user:qwen3_6_27b -->
| vLLM | FP16<br>TP4 | FP8<br>TP4 | FP8<br>TP2 |
|---|---:|---:|---:|
| 0.19 | 40.05 | 54.35 | 35.13 |
| 0.21 | 35.37 | 46.13 | 31.69 |
<!-- endrender -->

Same-card (TP4) **FP8 beats FP16** (54 vs 40 on 0.19); the **half-GPU FP8 TP2** option still serves
~32–35 tok/s, so you can run 27B on 2 cards and free the other two.

## Concurrency shape
*At a comparable serving config (same TP), how precision/engine scales C1→C8. Each config has two rows:
**per-user** = one stream's experience; **aggregate** = total box throughput.*

<!-- render:concurrency:qwen3_6_27b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16 TP4 | per-user | 40.05 | 31.57 | 31.02 | 30.47 |
|  | aggregate | 40.05 | 63.14 | 124.08 | 243.76 |
| 0.19 FP8 TP4 | per-user | 54.35 | 43.0 | 31.68 | 21.91 |
|  | aggregate | 54.35 | 86.0 | 126.72 | 175.28 |
| 0.21 FP16 TP4 | per-user | 35.37 | 28.48 | 27.71 | 27.41 |
|  | aggregate | 35.37 | 56.96 | 110.84 | 219.26 |
| 0.21 FP8 TP4 | per-user | 46.13 | 37.12 | 28.49 | 20.31 |
|  | aggregate | 46.13 | 74.24 | 113.96 | 162.45 |
<!-- endrender -->

FP8 leads per-user through ~C4; **FP16 reclaims the C8 aggregate** (244 vs 175 on 0.19) — the dense
CUDA-core dequant doesn't scale with batch like cuBLAS tensor cores (Chapter 5).

## Caveats
- FP8 is **Exact** (deterministic greedy) on both engines; all categories coherent.
- `max-model-len=32768`; longer contexts shrink KV headroom.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability — it includes the earlier
Ch1 / TP-sweep rows the digests above omit. The digests are the recommended reading; if a digest and
these rows ever disagree, **the SSOT rows win** and the renderer/prose is fixed.*

<!-- render:model:Qwen3.6-27B -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock-vllm | 37.4 | - | 0.26 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 35.05 | - | 1.21 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 22.27 | 22.27 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 9.15 | 73.25 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 36.47 | 36.47 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 14.96 | 119.66 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP8 | 1 | fp8-plugin+coalesced | 44.95 | 44.95 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP8 | 8 | fp8-plugin+coalesced | 20.73 | 165.83 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 31.69 | 31.69 | 62.35 | results/perf_v2_q27b_fp8_021_20260620_181033 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 23.92 | 47.84 | - | results/perf_v2_q27b_fp8_021_20260620_181033 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 18.82 | 75.28 | - | results/perf_v2_q27b_fp8_021_20260620_181033 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 12.11 | 96.91 | - | results/perf_v2_q27b_fp8_021_20260620_181033 |
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock-vllm | 35.37 | 35.37 | 25.97 | results/perf_v2_q27b_fp16_021_20260621_034910 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | stock-vllm | 28.48 | 56.96 | - | results/perf_v2_q27b_fp16_021_20260621_034910 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | stock-vllm | 27.71 | 110.84 | - | results/perf_v2_q27b_fp16_021_20260621_034910 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock-vllm | 27.41 | 219.26 | - | results/perf_v2_q27b_fp16_021_20260621_034910 |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 46.13 | 46.13 | 32.22 | results/perf_v2_q27b4_fp8_021_20260621_182713 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 37.12 | 74.24 | - | results/perf_v2_q27b4_fp8_021_20260621_182713 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 28.49 | 113.96 | - | results/perf_v2_q27b4_fp8_021_20260621_182713 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 20.31 | 162.45 | - | results/perf_v2_q27b4_fp8_021_20260621_182713 |
| 0.19.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 35.13 | 35.13 | 69.07 | results/perf_v2_q27b_fp8_019_20260620_200112 |
| 0.19.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 26.06 | 52.12 | - | results/perf_v2_q27b_fp8_019_20260620_200112 |
| 0.19.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 17.9 | 71.6 | - | results/perf_v2_q27b_fp8_019_20260620_200112 |
| 0.19.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 12.57 | 100.54 | - | results/perf_v2_q27b_fp8_019_20260620_200112 |
| 0.19.0/cu126 | fp16 | TP4 | 1 | stock-vllm | 40.05 | 40.05 | 28.94 | results/perf_v2_q27b_fp16_019_20260621_040702 |
| 0.19.0/cu126 | fp16 | TP4 | 2 | stock-vllm | 31.57 | 63.14 | - | results/perf_v2_q27b_fp16_019_20260621_040702 |
| 0.19.0/cu126 | fp16 | TP4 | 4 | stock-vllm | 31.02 | 124.08 | - | results/perf_v2_q27b_fp16_019_20260621_040702 |
| 0.19.0/cu126 | fp16 | TP4 | 8 | stock-vllm | 30.47 | 243.76 | - | results/perf_v2_q27b_fp16_019_20260621_040702 |
| 0.19.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 54.35 | 54.35 | 34.92 | results/perf_v2_q27b4_fp8_019_20260621_184344 |
| 0.19.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 43.0 | 86.0 | - | results/perf_v2_q27b4_fp8_019_20260621_184344 |
| 0.19.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 31.68 | 126.72 | - | results/perf_v2_q27b4_fp8_019_20260621_184344 |
| 0.19.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 21.91 | 175.28 | - | results/perf_v2_q27b4_fp8_019_20260621_184344 |
<!-- endrender -->
