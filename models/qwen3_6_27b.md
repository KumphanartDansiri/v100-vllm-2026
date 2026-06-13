# Qwen3.6-27B (dense) — V100 model page

> **Status: DRAFT** — numbers provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)).
> Table auto-renders from `data/benchmark_matrix.csv`. FP16 TP sweep + concurrency still to run.

A dense 27B. The interesting V100 story here is **memory fit vs TP**, and the fact that FP8 buys you
a lower TP floor — not more speed (dense FP8 is a memory play, not a throughput play; see Chapter 5).

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

## Measured (cudagraph, vLLM 0.21+cu126)

<!-- render:model:Qwen3.6-27B -->
| variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|
| fp16 | TP4 | 1 | stock-vllm | 37.4 | - | 0.26 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp8 | TP4 | 1 | fp8-plugin+coalesced | 35.05 | - | 1.21 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp8 | TP2 | 1 | fp8-plugin+coalesced | 22.27 | 22.27 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| fp8 | TP2 | 8 | fp8-plugin+coalesced | 9.15 | 73.25 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| fp8 | TP4 | 1 | fp8-plugin+coalesced | 36.47 | 36.47 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| fp8 | TP4 | 8 | fp8-plugin+coalesced | 14.96 | 119.66 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| fp8 | TP8 | 1 | fp8-plugin+coalesced | 44.95 | 44.95 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| fp8 | TP8 | 8 | fp8-plugin+coalesced | 20.73 | 165.83 | - | results/tp_sweep_q27b_fp8_20260613_110847/SUMMARY.txt |
| fp8 | TP | 1 | +mtp(k=1) | 26.48 | - | - | results/ch2_mtp_20260612/CHAIN_SUMMARY.txt |
<!-- endrender -->

## How to read it
- **TP8 = best latency** (~45 tok/s single-stream, ~21/user at 8 concurrent).
- **TP2 = best GPU efficiency** (~36 aggregate tok/s *per GPU* — run 4 independent TP2 replicas
  across 8 cards for maximum total throughput).
- Decode scales up with TP because dense decode is bandwidth-bound; more cards = more aggregate HBM
  bandwidth feeding the same weights.
- FP8-vs-FP16 at the same TP is roughly a wash on speed (dense) — FP8's value here is the lower TP
  floor and half the memory, freeing cards for other models or higher concurrency.

## Caveats
- FP16 TP-sweep + 8-user numbers pending (only Ch1 TP4 single-user FP16 so far).
- Numbers are `gentok=512, max-model-len=4096`; longer contexts shrink KV headroom.
