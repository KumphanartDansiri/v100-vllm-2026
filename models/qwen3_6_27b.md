# Qwen3.6-27B (dense) — V100 model page

> **Status: DRAFT** — numbers provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)).
> Table auto-renders from `data/benchmark_matrix.csv`; perf_v2 includes dual-engine FP16/FP8 same-TP rows plus earlier TP-scaling rows.

A dense 27B. The interesting V100 story here is **memory fit vs TP**, plus a post-breakthrough twist:
FP8 buys you a lower TP floor **and** — since the branchless E4M3 dequant — **faster low-user decode**
than FP16 at the same TP (54 vs 40 tok/s C1 on 0.19). FP16 still reclaims the 8-user aggregate (the
dense CUDA-core-vs-tensor-core wall; see Chapter 5). So dense FP8 is no longer a pure memory play —
it's a low-concurrency *speed* play too.

## What fits (32GB cards)
- **FP16 (~52 GB on disk):** needs **TP≥2** (TP1 is 52 GB ≫ 32 GB).
- **FP8 (~29 GB on disk, resident):** feasible **TP{2,4,8}** — **TP1 OOMs**, see below.

## Why FP8 OOMs at TP1 (a useful V100 lesson)
The FP8 weights are **resident** (~27 GB in HBM, genuinely half of FP16) — *not* dequantized back to
54 GB. TP1 still OOMs because ~27 GB of weights + KV cache + activations + CUDA/NCCL context +
cudagraph buffers don't fit one card's ~29 GB budget (`gpu-memory-utilization 0.90`). Proof it's
resident, not dequantized: KV-cache headroom scales as 1/TP — **183k tokens at TP2 → 745k at TP8
(≈4×)**, exactly what halving weight-bytes-per-card twice predicts. If FP8 were secretly running at
the 54 GB FP16 footprint, TP2 would have a fraction of that KV room.

## Measured (cudagraph, both engines)

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
| 0.21.0/cu126 | fp8 | TP | 1 | +mtp(k=1) | 26.48 | - | - | results/ch2_mtp_20260612/CHAIN_SUMMARY.txt |
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

## How to read it
- **FP8 beats FP16 at low concurrency, same TP4** — C1 **54.4 vs 40.1** (0.19), **46.1 vs 35.4**
  (0.21). FP8 leads **C1–C4**; **FP16 reclaims the 8-user aggregate** (FP16 244 vs FP8 175 agg @C8 on
  0.19). Same dense wall as gemma-4-31B: our FP8 dequant runs on **CUDA cores** while FP16 reaches
  cuBLAS **tensor cores** that scale better at batch — a WMMA FP8 decode kernel would close it.
- **TP is a memory ↔ throughput dial.** TP2 fits FP8 on **2 cards** (~32–35 tok/s C1) — run independent
  replicas for max aggregate; TP4 is the best single-model decode point. Dense decode is
  bandwidth-bound, so more cards = more aggregate HBM bandwidth feeding the same weights.
- **0.19 is faster than 0.21** on decode at every same-TP point (the fleet-wide pattern), FP8 and FP16
  alike.

## Caveats
- FP8 is **Exact** (deterministic greedy) on both engines; all categories coherent.
- Numbers are `max-model-len=32768`; longer contexts shrink KV headroom. The TP8 rows are the earlier
  pre-breakthrough TP-sweep (kept for the TP-scaling shape); the same-TP FP8/FP16 comparison uses the
  perf_v2 TP4 rows.
