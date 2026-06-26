# Qwen3.5-27B (dense — FP16 + FP8 + GPTQ-Int4) — V100 model-family page

> **Status: Final — fully tested.** Featured Qwen3.5 dense example, the full FP16/FP8/GPTQ-Int4 triad measured at the **fleet condition** (32768 ctx, both engines vLLM 0.19 + 0.21, cu126) — the same setup as every flagship. Digest tables render from `data/benchmark_matrix.csv`; the raw SSOT table is at the bottom. The precision comparison, the TP2 half-GPU capacity result, and the **GPTQ-Int4 correctness caveat** are in [Chapter 5](../docs/05_fp8_plugin.md).

A dense 27B — the featured worked example for the *dense* side of the architecture-fit line. The V100 story: since the branchless E4M3 dequant, **FP8 decodes *faster* than FP16 at low concurrency** at the same TP, and fits at a lower TP floor; **FP16 reclaims the 8-user aggregate** (the dense CUDA-core-vs-tensor-core wall, [Chapter 5](../docs/05_fp8_plugin.md)). GPTQ-Int4 is fastest raw but its **27B checkpoint emits degenerate output on V100** (a known upstream GPTQ-on-Volta issue — speed-only, see Caveats).

## Family / checkpoints
- `Qwen/Qwen3.5-27B` — FP16 baseline (stock vLLM).
- `Qwen/Qwen3.5-27B-FP8` — FP8 W8A16 plugin path.
- `Qwen/Qwen3.5-27B-GPTQ-Int4` — GPTQ re-quant (stock vLLM); **output is degenerate on V100** (upstream GPTQ-on-Volta issue — a speed-only datapoint, see Caveats).
- Architecturally the same config as Qwen3.6-27B; behavior matches once TP is held equal ([Chapter 5](../docs/05_fp8_plugin.md)).

## Fit / compatibility
- **FP16 (~52 GB):** runs at **TP4**; **OOM at TP2** under the standard serve envelope (weights ~25 GB/GPU leave no KV-cache headroom — see Ch.5).
- **FP8 (~29 GB, resident):** **TP4 and TP2** both fit; half-GPU deployment is a real option.
- **GPTQ-Int4 (~29 GB):** TP4 and TP2 fit; stock vLLM (engine-matched 0.19/0.21+cu126).

## Single-user deployment summary
*What one stream expects at C1 — decode throughput per engine.*

<!-- render:single_user:qwen3_5_27b -->
| Choice | Type | 0.19 | 0.21 |
|---|---|:---:|:---:|
| FP16* TP4 | Decode | 40.04 tok/s | 35.43 tok/s |
|  | Exactness | ✓ | ✓ |
|  | Correctness | ✓ | ✓ |
| FP8 TP4 | Decode | 54.05 tok/s | 46.05 tok/s |
|  | Exactness | ✓ | ✓ |
|  | Correctness | ✓ | ✓ |
| GPTQ-Int4 TP4 ⚠ speed-only | Decode | 60.47 tok/s | 50.36 tok/s |
|  | Exactness | ✗ | ✗ |
|  | Correctness | ✗ | ✗ |

_**Decode** = per-user tok/s at C1. **Exactness** ✓ = bit-identical run-to-run (temp 0). **Correctness** ✓ = coherent, usable output. So ✗ exactness / ✓ correctness = not bit-exact but coherent (e.g. FP8/MoE routing drift — expected, not an error); ✗ / ✗ = degenerate output (the GPTQ-Int4 27B case)._

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

Same-card (TP4) **FP8 beats FP16** at one user; the **half-GPU FP8 TP2** option still serves a usable stream, so you can run 27B on 2 cards and free the other two (FP16 cannot — it OOMs at TP2).

## First-token latency (TTFT)
*Single-stream cold first-token (fresh, cache-cold full prefill — worst case) and the FA-V100 variant.*

<!-- render:ttft:qwen3_5_27b -->
| Choice | Engine | Cold First Token | FA-on Cold | Prefix-cache Hit |
|---|---|---:|---:|---:|
| FP16* TP4 | 0.19 | 29.59 s | — | — |
|  | 0.21 | 26.96 s | 11.23 s | — |
| FP8 TP4 | 0.19 | 34.89 s | — | — |
|  | 0.21 | 32.21 s | 16.65 s | — |
| GPTQ-Int4 TP4 ⚠ speed-only | 0.19 | 29.52 s | — | — |
|  | 0.21 | 26.90 s | — | — |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

Here the **FA-V100 bridge roughly halves cold TTFT** (FP8 32→17 s, FP16 27→11 s) — the dense prefill bottleneck is attention, so FA helps directly (unlike the MoE, where the cost is FP8 compute and FA barely moves it).

## Concurrency shape
*At the same TP, how precision scales C1→C8. **Per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:qwen3_5_27b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16* TP4 | Per-user | 40.04 | 31.52 | 30.97 | 30.26 |
|  | Aggregate | 40.04 | 63.03 | 123.88 | 242.07 |
| 0.19 FP8 TP4 | Per-user | 54.05 | 42.83 | 30.83 | 20.43 |
|  | Aggregate | 54.05 | 85.66 | 123.32 | 163.46 |
| 0.19 GPTQ-Int4 TP4 ⚠ speed-only | Per-user | 60.47 | 48.12 | 44.92 | 40.19 |
|  | Aggregate | 60.47 | 96.24 | 179.67 | 321.50 |
| 0.21 FP16* TP4 | Per-user | 35.43 | 28.46 | 27.72 | 27.40 |
|  | Aggregate | 35.43 | 56.93 | 110.87 | 219.19 |
| 0.21 FP8 TP4 | Per-user | 46.05 | 37.06 | 28.41 | 19.44 |
|  | Aggregate | 46.05 | 74.12 | 113.63 | 155.52 |
| 0.21 GPTQ-Int4 TP4 ⚠ speed-only | Per-user | 50.36 | 41.25 | 38.82 | 34.90 |
|  | Aggregate | 50.36 | 82.50 | 155.30 | 279.24 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

FP8 leads per-user through ~C4; **FP16 reclaims the C8 aggregate** — the dense CUDA-core dequant doesn't scale with batch like cuBLAS tensor cores ([Chapter 5](../docs/05_fp8_plugin.md)). GPTQ-Int4 is fastest raw at every point, but its **27B output is garbage on V100** (upstream GPTQ issue — speed-only; see Caveats and the Ch.5 correctness note).

## Caveats
- **Correctness battery (fleet):** FP16 and FP8 are coherent (4/5 tests pass; the 5th is a strict JSON-format nitpick, not an error). FP8-vs-FP16 greedy output is **Stable** (coherent, token-divergent — different numerics, not errors). **GPTQ-Int4 is the exception — degenerate repetition (hard fail) on V100**, a known upstream GPTQ-on-Volta defect, so its tok/s are a speed-only datapoint, not a deployable config.
- `max-model-len=32768` (the fleet condition); the cold-prefill TTFT above is the full ~22.6k-token prompt. **TP2 half-GPU capacity** (FP16 OOMs, FP8/Int4 fit) is a 4096-ctx sub-study in [Chapter 5](../docs/05_fp8_plugin.md).
- **Both engines measured** (0.19 + 0.21): 0.19 runs ~13% faster, but the precision crossover and ordering are identical — see the dual-engine triad in [Chapter 5](../docs/05_fp8_plugin.md).

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv` for auditability. If a digest and these rows disagree, **the SSOT rows win**.*

<!-- render:model:Qwen3.5-27B -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | FP16* | TP4 | 1 | stock-vllm | 35.43 | 35.43 | 26.96 | 11.23 | - | results/perf_v2_q27b35_fp16_021_20260626_042240 |
| 0.21.0/cu126 | FP16* | TP4 | 2 | stock-vllm | 28.46 | 56.93 | - | - | - | results/perf_v2_q27b35_fp16_021_20260626_042240 |
| 0.21.0/cu126 | FP16* | TP4 | 4 | stock-vllm | 27.72 | 110.87 | - | - | - | results/perf_v2_q27b35_fp16_021_20260626_042240 |
| 0.21.0/cu126 | FP16* | TP4 | 8 | stock-vllm | 27.40 | 219.19 | - | - | - | results/perf_v2_q27b35_fp16_021_20260626_042240 |
| 0.19.0/cu126 | FP16* | TP4 | 1 | stock-vllm | 40.04 | 40.04 | 29.59 | - | - | results/perf_v2_q27b35_fp16_019_20260626_054205 |
| 0.19.0/cu126 | FP16* | TP4 | 2 | stock-vllm | 31.52 | 63.03 | - | - | - | results/perf_v2_q27b35_fp16_019_20260626_054205 |
| 0.19.0/cu126 | FP16* | TP4 | 4 | stock-vllm | 30.97 | 123.88 | - | - | - | results/perf_v2_q27b35_fp16_019_20260626_054205 |
| 0.19.0/cu126 | FP16* | TP4 | 8 | stock-vllm | 30.26 | 242.07 | - | - | - | results/perf_v2_q27b35_fp16_019_20260626_054205 |
| 0.21.0/cu126 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 46.05 | 46.05 | 32.21 | 16.65 | - | results/perf_v2_q27b35_fp8_021_20260626_043820 |
| 0.21.0/cu126 | FP8 | TP4 | 2 | fp8-plugin+coalesced | 37.06 | 74.12 | - | - | - | results/perf_v2_q27b35_fp8_021_20260626_043820 |
| 0.21.0/cu126 | FP8 | TP4 | 4 | fp8-plugin+coalesced | 28.41 | 113.63 | - | - | - | results/perf_v2_q27b35_fp8_021_20260626_043820 |
| 0.21.0/cu126 | FP8 | TP4 | 8 | fp8-plugin+coalesced | 19.44 | 155.52 | - | - | - | results/perf_v2_q27b35_fp8_021_20260626_043820 |
| 0.19.0/cu126 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 54.05 | 54.05 | 34.89 | - | - | results/perf_v2_q27b35_fp8_019_20260626_055145 |
| 0.19.0/cu126 | FP8 | TP4 | 2 | fp8-plugin+coalesced | 42.83 | 85.66 | - | - | - | results/perf_v2_q27b35_fp8_019_20260626_055145 |
| 0.19.0/cu126 | FP8 | TP4 | 4 | fp8-plugin+coalesced | 30.83 | 123.32 | - | - | - | results/perf_v2_q27b35_fp8_019_20260626_055145 |
| 0.19.0/cu126 | FP8 | TP4 | 8 | fp8-plugin+coalesced | 20.43 | 163.46 | - | - | - | results/perf_v2_q27b35_fp8_019_20260626_055145 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 1 | stock-vllm | 50.36 | 50.36 | 26.90 | - | - | results/perf_v2_q27b35_int4_021_20260626_045330 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 2 | stock-vllm | 41.25 | 82.50 | - | - | - | results/perf_v2_q27b35_int4_021_20260626_045330 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 4 | stock-vllm | 38.82 | 155.30 | - | - | - | results/perf_v2_q27b35_int4_021_20260626_045330 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 8 | stock-vllm | 34.90 | 279.24 | - | - | - | results/perf_v2_q27b35_int4_021_20260626_045330 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 1 | stock-vllm | 60.47 | 60.47 | 29.52 | - | - | results/perf_v2_q27b35_int4_019_20260626_060105 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 2 | stock-vllm | 48.12 | 96.24 | - | - | - | results/perf_v2_q27b35_int4_019_20260626_060105 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 4 | stock-vllm | 44.92 | 179.67 | - | - | - | results/perf_v2_q27b35_int4_019_20260626_060105 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 8 | stock-vllm | 40.19 | 321.50 | - | - | - | results/perf_v2_q27b35_int4_019_20260626_060105 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->
