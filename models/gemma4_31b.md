# gemma-4-31B-it (dense — FP16 + FP8) — V100 model-family page

> **Status: DRAFT** — provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)). Digest tables render from `data/benchmark_matrix.csv` (perf_v2-frozen rows only); the exhaustive raw SSOT table is at the bottom.

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
*What one stream expects at C1 — decode throughput on each engine, plus representative 0.21 cold first-token latency; this is the precision/TP choice for a solo user or small lab.*

<!-- render:single_user:gemma4_31b -->
| Choice | 0.19 C1 decode | 0.21 C1 decode | 0.21 Cold TTFT | 0.21 Warm TTFT¹ |
|---|---:|---:|---:|---:|
| FP16 TP4 | 26.73 tok/s | 26.73 tok/s | 185.82 s | pending |
| FP8 TP4 | 35.23 tok/s | 35.28 tok/s | 196.03 s | pending |
| FP8 TP2 | — | 23.07 tok/s | — | pending |

¹ **Warm TTFT** = warm / prefix-cache-hit / chunked-prefill serving latency — **pending SSOT refresh**. **Cold TTFT** is cold *monolithic* prefill from the representative SSOT row: a **worst-case** number, *not* warm serving latency — don't read it as steady interactive response.
<!-- endrender -->

**FP8 beats FP16 at C1** (35.3 vs 26.7) — the value here is *both* memory and low-user speed. The
half-GPU **FP8 TP2** (~23 tok/s) lets you serve gemma-4-31B on 2 cards. **Cold first-token latency is
high** (~190 s) — dense Gemma-4 prefill is untuned on Volta and is the slow part; **decode is the story
here**; warm/chunked TTFT is pending and should not be inferred from this cold number.

## Concurrency shape
*At a comparable serving config (same TP), how precision/engine scales C1→C8. Each config has two rows:
**per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:gemma4_31b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16 TP4 | per-user | 26.73 | 24.17 | 21.44 | 17.1 |
|  | aggregate | 26.73 | 48.34 | 85.76 | 136.79 |
| 0.19 FP8 TP4 | per-user | 35.23 | 28.19 | 19.67 | 12.72 |
|  | aggregate | 35.23 | 56.38 | 78.68 | 101.78 |
| 0.21 FP16 TP4 | per-user | 26.73 | 24.07 | 21.43 | 17.08 |
|  | aggregate | 26.73 | 48.14 | 85.72 | 136.66 |
| 0.21 FP8 TP4 | per-user | 35.28 | 28.25 | 19.65 | 12.68 |
|  | aggregate | 35.28 | 56.5 | 78.6 | 101.47 |
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
