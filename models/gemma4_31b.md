# gemma-4-31B-it (dense — FP16 + FP8) — V100 model-family page

> **Status: Final** — numbers frozen from `data/benchmark_matrix.csv` (perf_v2 rows, tag `fp8-v100-2026-matrix`); digest tables auto-render and the exhaustive raw SSOT table is at the bottom. Refresh procedure: [../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md).

A dense 31B (`Gemma4ForConditionalGeneration`, 60 layers, no MoE). After the branchless-dequant
breakthrough, **FP8 is *faster* than FP16 at low concurrency** (35 vs 27 at C1) at half the weight
footprint; FP16 reclaims the 8-user aggregate (the dense CUDA-core-vs-tensor-core wall, Chapter 5).

## Family / checkpoints
- `google/gemma-4-31B-it` — FP16 baseline.
- `RedHatAI/gemma-4-31B-it-FP8-Dynamic` — FP8 plugin path.
- **Compatibility:** needs **transformers 5** on vLLM 0.19 (the `vllm019-tf5` image; stock 0.19's
  tf 4.57 can't parse `model_type=gemma4`); runs on **0.21 stock** (its base ships tf5). Dense → MoE
  patch n/a; FP8 plugin works on both engines. (Chapter 6 matrix.)

## Fit / compatibility
- **TP4** on 4×V100. FP16 ~62 GB → ~15.5 GB/GPU; FP8 ~31 GB → ~7.8 GB/GPU (half the bytes →
  ~1.55× more KV headroom).
- `--max-num-batched-tokens ≥ 2496` is **required on 0.21** (Gemma-4's vision tower forces it; the
  2048 default fails at startup). Text decode is insensitive to the value.
- **FP8 also fits at TP2** (2 cards, short context) — the half-GPU option.
- **Best engine:** within noise on both precisions (FP8 is our engine-invariant kernel; FP16 is a
  portability result).

## Single-user deployment summary
*What one stream expects at C1 — decode throughput on each engine; this is the precision/TP choice for a solo user or small lab.*

<!-- render:single_user:gemma4_31b -->
| Choice | Type | 0.19 | 0.21 |
|---|---|:---:|:---:|
| FP16* TP4 | Decode | 26.73 tok/s | 26.73 tok/s |
|  | Exactness | ✓ | ✓ |
|  | Correctness | ✓ | ✓ |
| FP8 TP4 | Decode | 35.23 tok/s | 35.28 tok/s |
|  | Exactness | ✓ | ✓ |
|  | Correctness | ✓ | ✓ |
| FP8 TP2 | Decode | — | 23.07 tok/s |
|  | Exactness | — | ✓ |
|  | Correctness | — | ✓ |

_**Decode** = per-user tok/s at C1. **Exactness** ✓ = bit-identical run-to-run (temp 0). **Correctness** ✓ = coherent, usable output. So ✗ exactness / ✓ correctness = not bit-exact but coherent (e.g. FP8/MoE routing drift — expected, not an error); ✗ / ✗ = degenerate output (the GPTQ-Int4 27B case)._

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

**FP8 beats FP16 at C1** (35.3 vs 26.7) — the value here is *both* memory and low-user speed. The
half-GPU **FP8 TP2** (~23 tok/s) lets you serve gemma-4-31B on 2 cards. **Decode is the story here** —
cold first-token latency is very high (~190 s, see below) because dense Gemma-4 prefill is untuned on
Volta; it dominates only the very first token of a cold request, not steady decode.

## First-token latency (TTFT)
*Single-stream time to first token: **cold** (a fresh, cache-cold full prefill — worst case) vs **prefix-cache-hit** (repeated / shared context — best case). Decode is the headline; this is the latency side.*

<!-- render:ttft:gemma4_31b -->
| Choice | Engine | Cold First Token | Prefix-cache Hit |
|---|---|---:|---:|
| FP16* TP4 | 0.19 | 187.641 s | 1.750 s |
|  | 0.21 | 186.170 s | 1.756 s |
| FP8 TP4 | 0.19 | 195.310 s | 2.296 s |
|  | 0.21 | 196.400 s | 2.323 s |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

## Concurrency shape
*At a comparable serving config (same TP), how precision/engine scales C1→C8. Each config has two rows:
**per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:gemma4_31b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16* TP4 | Per-user | 26.73 | 24.17 | 21.44 | 17.1 |
|  | Aggregate | 26.73 | 48.34 | 85.76 | 136.79 |
| 0.19 FP8 TP4 | Per-user | 35.23 | 28.19 | 19.67 | 12.72 |
|  | Aggregate | 35.23 | 56.38 | 78.68 | 101.78 |
| 0.21 FP16* TP4 | Per-user | 26.73 | 24.07 | 21.43 | 17.08 |
|  | Aggregate | 26.73 | 48.14 | 85.72 | 136.66 |
| 0.21 FP8 TP4 | Per-user | 35.28 | 28.25 | 19.65 | 12.68 |
|  | Aggregate | 35.28 | 56.5 | 78.6 | 101.47 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

FP8 wins C1–C2, the curves **cross at ~C4**, and **FP16 takes the C8 aggregate** (137 vs 101) — dense
decode streams the whole weight set every token, where our CUDA-core dequant doesn't scale with batch
like cuBLAS tensor cores (Chapter 5).

## Caveats
- Both FP16 and FP8 are **Exact** (deterministic greedy); all categories coherent.
- **Cold TTFT ~185–196 s** — dense gemma-4 prefill is the slow part on Volta (untuned-prefill state).
- `max-model-len=32768`; the TP2 half-GPU option is short-context.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability. The digests are the
recommended reading; if a digest and these rows ever disagree, **the SSOT rows win** and the
renderer/prose is fixed.*

<!-- render:model:gemma-4-31B-it -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | FP16* | TP4 | 1 | stock-vllm | 17.61 | - | 0.16 | - | - | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 17.53 | - | 0.45 | - | - | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 35.28 | 35.28 | 196.400 | - | 2.323 | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | FP8 | TP4 | 2 | fp8-plugin+coalesced | 28.25 | 56.5 | - | - | - | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | FP8 | TP4 | 4 | fp8-plugin+coalesced | 19.65 | 78.6 | - | - | - | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | FP8 | TP4 | 8 | fp8-plugin+coalesced | 12.68 | 101.47 | - | - | - | results/perf_v2_g31b_fp8_021_20260621_042216 |
| 0.21.0/cu126 | FP16* | TP4 | 1 | stock-vllm | 26.73 | 26.73 | 186.170 | - | 1.756 | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.21.0/cu126 | FP16* | TP4 | 2 | stock-vllm | 24.07 | 48.14 | - | - | - | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.21.0/cu126 | FP16* | TP4 | 4 | stock-vllm | 21.43 | 85.72 | - | - | - | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.21.0/cu126 | FP16* | TP4 | 8 | stock-vllm | 17.08 | 136.66 | - | - | - | results/perf_v2_g31b_fp16_021_20260621_043546 |
| 0.19.0/cu128 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 35.23 | 35.23 | 195.310 | - | 2.296 | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | FP8 | TP4 | 2 | fp8-plugin+coalesced | 28.19 | 56.38 | - | - | - | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | FP8 | TP4 | 4 | fp8-plugin+coalesced | 19.67 | 78.68 | - | - | - | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | FP8 | TP4 | 8 | fp8-plugin+coalesced | 12.72 | 101.78 | - | - | - | results/perf_v2_g31b_fp8_019_20260620_221814 |
| 0.19.0/cu128 | FP16* | TP4 | 1 | stock-vllm | 26.73 | 26.73 | 187.641 | - | 1.750 | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.19.0/cu128 | FP16* | TP4 | 2 | stock-vllm | 24.17 | 48.34 | - | - | - | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.19.0/cu128 | FP16* | TP4 | 4 | stock-vllm | 21.44 | 85.76 | - | - | - | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.19.0/cu128 | FP16* | TP4 | 8 | stock-vllm | 17.1 | 136.79 | - | - | - | results/perf_v2_g31b_fp16_019_20260620_223421 |
| 0.21.0/cu126 | FP8 | TP2 | 1 | fp8-plugin+coalesced | 23.07 | 23.07 | - | - | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | FP8 | TP2 | 2 | fp8-plugin+coalesced | 17.34 | 34.68 | - | - | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | FP8 | TP2 | 4 | fp8-plugin+coalesced | 11.09 | 44.36 | - | - | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | FP8 | TP2 | 8 | fp8-plugin+coalesced | 6.97 | 55.74 | - | - | - | results/perf_v2_g31b2_fp8_021_20260622_003941 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->
