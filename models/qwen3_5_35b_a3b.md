# Qwen3.5-35B-A3B (MoE — FP16 + FP8 + GPTQ-Int4) — V100 model-family page

> **Status: Final** — featured Qwen3.5 MoE example. Digest tables render from `data/benchmark_matrix.csv` (06-24/25 exact-triad + perf_v2 rows, engine 0.21); raw SSOT table at the bottom. Precision × tensor-parallelism (incl. GPTQ-Int4 and the TP2 capacity result) is [Chapter 5](../docs/05_fp8_plugin.md).

A sparse MoE (~3B of 35B active/token) — the featured worked example for the *MoE* side. This is where the V100 FP8 path is strongest: **FP8 wins decode at every concurrency** (the reduced weight traffic stays valuable when only ~3B params move per token — the architecture-fit case, [Chapter 5](../docs/05_fp8_plugin.md)), **and** it's the only faithful format that **fits at half TP**. The cost is prefill: the block-FP8 MoE cold-prefill is the slowest cell in the report.

## Family / checkpoints
- `Qwen/Qwen3.5-35B-A3B` — FP16 baseline (stock vLLM + the FP16-MoE config fix, [Chapter 2](../docs/02_fp16_moe_fix.md)).
- `Qwen/Qwen3.5-35B-A3B-FP8` — FP8 W8A16 plugin path (grouped/coalesced MoE decode).
- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` — GPTQ re-quant (stock vLLM, lossy).
- Same config as Qwen3.6-35B-A3B; behavior matches at equal TP ([Chapter 5](../docs/05_fp8_plugin.md)).

## Fit / compatibility
- **FP16 (~66 GB):** runs at **TP4**; **hard OOM at TP2** — ~33 GB of weights per GPU exceeds the 32 GB card *before* any KV cache (Ch.5). FP16 simply cannot make a half-GPU deployment.
- **FP8 (~34 GB, resident):** **TP4 and TP2** both fit (~17 GB/GPU at TP2) — FP8 is what enables the half-GPU MoE.
- **GPTQ-Int4 (~22 GB):** TP4 and TP2 fit; stock vLLM (0.18/cu128 here).

## Single-user deployment summary
*What one stream expects at C1 — decode throughput per engine.*

<!-- render:single_user:qwen3_5_35b_a3b -->
| Choice | 0.21 C1 Decode |
|---|---:|
| FP16 TP4 | 66.20 tok/s |
| FP8 TP4 | 92.96 tok/s |
| FP8 TP2 | 82.20 tok/s |
<!-- endrender -->

FP8 leads FP16 substantially at one user (and the gap *widens* under load — see concurrency). The **half-GPU FP8 TP2** option still serves a strong stream; FP16 is not on the board at TP2.

## First-token latency (TTFT)
*Single-stream cold first-token (worst case) and the FA-V100 variant.*

<!-- render:ttft:qwen3_5_35b_a3b -->
| Choice | Engine | Cold First Token | FA-on Cold | Prefix-cache Hit |
|---|---|---:|---:|---:|
| FP16 TP4 | 0.21 | 14.20 s | 9.09 s | — |
| FP8 TP4 | 0.21 | 55.11 s | 49.90 s | — |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.
<!-- endrender -->

This is the FP8 cost side and it is steep: the **block-FP8 MoE cold prefill is ~55 s vs ~14 s for FP16** — the unoptimized FP8 MoE prefill path (Volta WMMA). Unlike the dense model, **FA-V100 barely helps** (55→50 s) because the bottleneck here is FP8-MoE prefill *compute*, not attention. For long-context / short-output traffic, FP16 or Int4 prefill far cheaper; for decode-heavy serving the cost is paid once and FP8's decode + capacity win dominates.

## Concurrency shape
*At the same TP, how precision scales C1→C8. **Per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:qwen3_5_35b_a3b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.21 FP16 TP4 | Per-user | 66.20 | 45.43 | 29.55 | 22.92 |
|  | Aggregate | 66.20 | 90.86 | 118.20 | 183.34 |
| 0.21 FP8 TP4 | Per-user | 92.96 | 77.59 | 72.36 | 54.93 |
|  | Aggregate | 92.96 | 155.18 | 289.42 | 439.42 |
<!-- endrender -->

**FP8 wins per-user at every concurrency** and the margin grows with load (1.40× at C1 → 2.45× at C4) — FP16-MoE decode collapses under concurrency while FP8 holds. GPTQ-Int4 is fastest raw but lossy; the full FP16/FP8/Int4 triad and the TP2 column are in [Chapter 5](../docs/05_fp8_plugin.md).

## Caveats
- FP8 is **Exact** (bit-deterministic greedy, 1 sha/5) — tighter than the 3.6-35B-A3B-FP8, which was only "Stable". FP8-vs-FP16 greedy output is **Stable** (coherent, token-divergent).
- `max-model-len=4096` for these runs.
- Prefill latency is the current-state limit (see TTFT); decode + capacity are the wins.
- 0.19-engine numbers pending (3.5 runs are 0.21 so far).

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv` for auditability. If a digest and these rows disagree, **the SSOT rows win**.*

<!-- render:model:Qwen3.5-35B-A3B -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(default) | 66.20 | 66.20 | 14.20 | 9.09 | - | results/q35b_exact_triad_20260624_115648/fp16 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | +moe_patch(default) | 45.43 | 90.86 | - | - | - | results/q35b_exact_triad_20260624_115648/fp16 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | +moe_patch(default) | 29.55 | 118.20 | - | - | - | results/q35b_exact_triad_20260624_115648/fp16 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(default) | 22.92 | 183.34 | - | - | - | results/q35b_exact_triad_20260624_115648/fp16 |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 92.96 | 92.96 | 55.11 | 49.90 | - | results/q35b_exact_triad_20260624_115648/fp8 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 77.59 | 155.18 | - | - | - | results/q35b_exact_triad_20260624_115648/fp8 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 72.36 | 289.42 | - | - | - | results/q35b_exact_triad_20260624_115648/fp8 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 54.93 | 439.42 | - | - | - | results/q35b_exact_triad_20260624_115648/fp8 |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 1 | stock-vllm | 126.19 | 126.19 | - | - | - | results/q35b_exact_triad_20260624_115648/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 2 | stock-vllm | 96.12 | 192.23 | - | - | - | results/q35b_exact_triad_20260624_115648/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 4 | stock-vllm | 76.20 | 304.80 | - | - | - | results/q35b_exact_triad_20260624_115648/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 8 | stock-vllm | 75.08 | 600.66 | - | - | - | results/q35b_exact_triad_20260624_115648/gptq |
| 0.21.0/cu126 | fp16 | TP2 | 1 | +moe_patch(default) | OOM | - | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp16 | TP2 | 2 | +moe_patch(default) | OOM | - | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp16 | TP2 | 4 | +moe_patch(default) | OOM | - | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp16 | TP2 | 8 | +moe_patch(default) | OOM | - | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 82.20 | 82.20 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp8 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 64.21 | 128.42 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp8 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 53.82 | 215.29 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp8 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 39.26 | 314.09 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/fp8 |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 1 | stock-vllm | 98.96 | 98.96 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 2 | stock-vllm | 74.19 | 148.37 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 4 | stock-vllm | 50.53 | 202.12 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 8 | stock-vllm | 51.86 | 414.87 | - | - | - | results/q35b_exact_triad_tp2_20260624_174257/gptq |
<!-- endrender -->
