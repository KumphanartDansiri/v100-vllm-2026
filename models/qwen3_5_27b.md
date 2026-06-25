# Qwen3.5-27B (dense — FP16 + FP8 + GPTQ-Int4) — V100 model-family page

> **Status: Final** — featured Qwen3.5 dense example. Digest tables render from `data/benchmark_matrix.csv` (06-24/25 exact-triad + perf_v2 rows, engine 0.21); the exhaustive raw SSOT table is at the bottom. The precision × tensor-parallelism story (incl. GPTQ-Int4 and the TP2 capacity result) is [Chapter 5](../docs/05_fp8_plugin.md).

A dense 27B — the featured worked example for the *dense* side of the architecture-fit line. The V100 story: since the branchless E4M3 dequant, **FP8 decodes *faster* than FP16 at low concurrency** at the same TP, and fits at a lower TP floor; **FP16 reclaims the 8-user aggregate** (the dense CUDA-core-vs-tensor-core wall, [Chapter 5](../docs/05_fp8_plugin.md)). GPTQ-Int4 is fastest raw but lossy.

## Family / checkpoints
- `Qwen/Qwen3.5-27B` — FP16 baseline (stock vLLM).
- `Qwen/Qwen3.5-27B-FP8` — FP8 W8A16 plugin path.
- `Qwen/Qwen3.5-27B-GPTQ-Int4` — GPTQ re-quant (stock vLLM, lossy).
- Architecturally the same config as Qwen3.6-27B; behavior matches once TP is held equal ([Chapter 5](../docs/05_fp8_plugin.md)).

## Fit / compatibility
- **FP16 (~52 GB):** runs at **TP4**; **OOM at TP2** under the standard serve envelope (weights ~25 GB/GPU leave no KV-cache headroom — see Ch.5).
- **FP8 (~29 GB, resident):** **TP4 and TP2** both fit; half-GPU deployment is a real option.
- **GPTQ-Int4 (~29 GB):** TP4 and TP2 fit; stock vLLM (0.18/cu128 here).

## Single-user deployment summary
*What one stream expects at C1 — decode throughput per engine.*

<!-- render:single_user:qwen3_5_27b -->
| Choice | 0.21 C1 Decode |
|---|---:|
| FP16 TP4 | 39.08 tok/s |
| FP8 TP4 | 52.50 tok/s |
| FP8 TP2 | 34.01 tok/s |
<!-- endrender -->

Same-card (TP4) **FP8 beats FP16** at one user; the **half-GPU FP8 TP2** option still serves a usable stream, so you can run 27B on 2 cards and free the other two (FP16 cannot — it OOMs at TP2).

## First-token latency (TTFT)
*Single-stream cold first-token (fresh, cache-cold full prefill — worst case) and the FA-V100 variant.*

<!-- render:ttft:qwen3_5_27b -->
| Choice | Engine | Cold First Token | FA-on Cold | Prefix-cache Hit |
|---|---|---:|---:|---:|
| FP16 TP4 | 0.21 | 26.96 s | 11.23 s | — |
| FP8 TP4 | 0.21 | 32.23 s | 17.36 s | — |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.
<!-- endrender -->

Here the **FA-V100 bridge roughly halves cold TTFT** (FP8 32→17 s, FP16 27→11 s) — the dense prefill bottleneck is attention, so FA helps directly (unlike the MoE, where the cost is FP8 compute and FA barely moves it).

## Concurrency shape
*At the same TP, how precision scales C1→C8. **Per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:qwen3_5_27b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.21 FP16 TP4 | Per-user | 39.08 | 31.11 | 30.30 | 29.30 |
|  | Aggregate | 39.08 | 62.22 | 121.18 | 234.36 |
| 0.21 FP8 TP4 | Per-user | 52.50 | 42.52 | 31.73 | 20.29 |
|  | Aggregate | 52.50 | 85.03 | 126.89 | 162.34 |
<!-- endrender -->

FP8 leads per-user through ~C4; **FP16 reclaims the C8 aggregate** — the dense CUDA-core dequant doesn't scale with batch like cuBLAS tensor cores ([Chapter 5](../docs/05_fp8_plugin.md)). GPTQ-Int4 is fastest raw at every point but is the lossy re-quant (full triad in Ch.5).

## Caveats
- FP8 is **Exact** (bit-deterministic greedy, 1 sha/5); all categories coherent. FP8-vs-FP16 greedy output is **Stable** (coherent, token-divergent — different numerics, not errors).
- `max-model-len=4096` for these runs; longer contexts shrink KV headroom (and were the reason FP16 OOMs at TP2).
- 0.19-engine numbers are pending (the 3.5 runs are 0.21 so far); 3.6-27B has both engines.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv` for auditability. If a digest and these rows disagree, **the SSOT rows win**.*

<!-- render:model:Qwen3.5-27B -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock-vllm | 39.08 | 39.08 | 26.96 | 11.23 | - | results/q27b_exact_triad_20260624_113728/fp16 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | stock-vllm | 31.11 | 62.22 | - | - | - | results/q27b_exact_triad_20260624_113728/fp16 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | stock-vllm | 30.30 | 121.18 | - | - | - | results/q27b_exact_triad_20260624_113728/fp16 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock-vllm | 29.30 | 234.36 | - | - | - | results/q27b_exact_triad_20260624_113728/fp16 |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 52.50 | 52.50 | 32.23 | 17.36 | - | results/q27b_exact_triad_20260624_113728/fp8 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 42.52 | 85.03 | - | - | - | results/q27b_exact_triad_20260624_113728/fp8 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 31.73 | 126.89 | - | - | - | results/q27b_exact_triad_20260624_113728/fp8 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 20.29 | 162.34 | - | - | - | results/q27b_exact_triad_20260624_113728/fp8 |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 1 | stock-vllm | 69.18 | 69.18 | - | - | - | results/q27b_exact_triad_20260624_113728/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 2 | stock-vllm | 55.54 | 111.07 | - | - | - | results/q27b_exact_triad_20260624_113728/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 4 | stock-vllm | 47.40 | 189.58 | - | - | - | results/q27b_exact_triad_20260624_113728/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP4 | 8 | stock-vllm | 44.15 | 353.20 | - | - | - | results/q27b_exact_triad_20260624_113728/gptq |
| 0.21.0/cu126 | fp16 | TP2 | 1 | stock-vllm | OOM | - | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp16 | TP2 | 2 | stock-vllm | OOM | - | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp16 | TP2 | 4 | stock-vllm | OOM | - | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp16 | TP2 | 8 | stock-vllm | OOM | - | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp16 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 34.01 | 34.01 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp8 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 26.18 | 52.34 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp8 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 18.63 | 74.54 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp8 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 12.42 | 99.36 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/fp8 |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 1 | stock-vllm | 43.30 | 43.30 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 2 | stock-vllm | 33.23 | 66.44 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 4 | stock-vllm | 25.46 | 101.81 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/gptq |
| 0.18.0/cu128 | GPTQ-Int4 | TP2 | 8 | stock-vllm | 26.80 | 214.43 | - | - | - | results/q27b_exact_triad_tp2_20260624_174257/gptq |
<!-- endrender -->
