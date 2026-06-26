# GLM-4.7-Flash (MLA MoE — BF16) — V100 model-family page

> **Status: Final** — numbers frozen from `data/benchmark_matrix.csv` (perf_v2 rows, tag `fp8-v100-2026-matrix`); digest tables auto-render and the exhaustive raw SSOT table is at the bottom. Refresh procedure: [../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md).

A **~31B-total / ~3B-active lite-MoE** (`Glm4MoeLiteForCausalLM`) with **MLA** attention, BF16 /
unquantized. **The first MLA-attention model to generate on the V100 (`sm_70`) stack** — not an
FP8-plugin model (none of the FP8 kernels engage); it's here to prove the MLA path.

## Family / checkpoints
- `zai-org/GLM-4.7-Flash` — the only checkpoint (BF16); served `--dtype float16` at TP4.
- **Compatibility:** needs **transformers 5** on 0.19 (`vllm019-tf5`) **and the env-gated MLA prefill
  patches on either engine** (stock loads but crashes on the first token); **cudagraph is mandatory**.
  (Chapter 6 matrix.)

## The wall it broke through
MLA models load and decode on Volta but **crash on the first token** — vLLM offers only the sm80+
FlashAttention MLA *prefill* backend. Three env-gated patches unblock it (decode stays stock TritonMLA):
1. **MLA prefill interpose** (`VLLM_V100_MLA_PREFILL=1`) → the ai-bond flash-attention-v100 dense
   varlen kernel (GLM's qk=v=256 is an exact tile). numcheck vs fp32 reference = **cos 1.000000**.
2. **Decode smem clamp** — TritonMLA's grouped decode wants ~100 KB shared memory (> V100's 96 KB);
   clamp the per-iter KV tile (`BLOCK_N 32→16`), correctness-neutral.
3. **Decode cudagraph enable** (`VLLM_V100_MLA_DECODE_CUDAGRAPH=1`) — **6.0 → 37.2 tok/s = 6.2×**
   (Chapter 3).

## Fit / compatibility
- **TP4, cudagraph** (mandatory — eager = 6 tok/s) with `VLLM_V100_MLA_PREFILL=1` +
  `VLLM_V100_MLA_DECODE_CUDAGRAPH=1`. ~15.6 GB/GPU.
- **Best engine — a split:** **0.19 wins decode**, **0.21 wins prefill**. Unlike the FP8-plugin models,
  decode here is **stock TritonMLA** (not our kernels), so it is *not* engine-invariant.

## Single-user deployment summary
*What one stream expects at C1 — decode throughput on each engine.*

<!-- render:single_user:glm4_7_flash -->
| Choice | 0.19 C1 Decode | 0.21 C1 Decode |
|---|---:|---:|
| FP16* TP4 | 35.36 tok/s | 30.97 tok/s |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

~31 (0.21) / ~35 (0.19) tok/s — comfortably usable, on the first MLA model to run on V100 at all. Its
**cold TTFT is high — see the TTFT / prefill section below**; note 0.21's ~144 s is actually *better*
than 0.19's ~212 s (the one model where 0.21 wins prefill).

## Concurrency shape
*How it scales C1→C8 at TP4. Each config has two rows: **per-user** = one stream; **aggregate** = total
box throughput.*

<!-- render:concurrency:glm4_7_flash -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16* TP4 | Per-user | 35.36 | 27.62 | 21.37 | 17.5 |
|  | Aggregate | 35.36 | 55.24 | 85.48 | 140.01 |
| 0.21 FP16* TP4 | Per-user | 30.97 | 21.17 | 13.25 | 10.35 |
|  | Aggregate | 30.97 | 42.34 | 53.0 | 82.78 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

Scales modestly (MLA decode + an untuned MoE shape); **0.19 scales notably better** (aggregate 140 vs
83 @C8). All streams coherent.

## TTFT / prefill — the open item
**Cold first-token ≈ 144 s (0.21) / 212 s (0.19)** — a fresh, cache-cold full prefill at ~22.6k tokens.
MLA prefill is inherently eager and GLM's MoE shape (`E=64, N=384`) has no tuned Volta config yet, so
prefill is the slow part — and uniquely, **0.21 prefills faster than 0.19 here**. The *decode* chapter
is done; the prefill/TTFT chapter (autotune that MoE JSON) is future work.

<!-- render:ttft:glm4_7_flash -->
| Choice | Engine | Cold First Token | Prefix-cache Hit |
|---|---|---:|---:|
| FP16* TP4 | 0.19 | 212.489 s | 0.452 s |
|  | 0.21 | 143.993 s | 0.454 s |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

Prefix-cache-hit (repeated/shared prefix) is sub-second — the cold number is the worst case, paid once
per fresh request.

## Caveats
- **Reasoning model** (`<think>…</think>`): coherent + **7/8** verifiable Q&A correct at realistic
  length; numcheck cos = 1.0, greedy run-to-run **Exact**. The matrix flags `quality=suspect` only
  because its forced-length (`ignore_eos`) protocol trips the reasoning-model repetition caveat, not
  the MLA path.
- **BF16, not FP8** — no FP8 checkpoint exists; the FP8 plugin doesn't engage.
- The three MLA patches are **experimental, env-gated, local** (sm_70). `max-model-len=32768`, TP4.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability. The digests are the
recommended reading; if a digest and these rows ever disagree, **the SSOT rows win** and the
renderer/prose is fixed.*

<!-- render:model:GLM-4.7-Flash -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | FP16* | TP4 | 1 | fp16mla+cudagraph | 30.97 | 30.97 | 143.993 | - | 0.454 | results/perf_v2_glm47_fp16_021_20260621_172718 |
| 0.21.0/cu126 | FP16* | TP4 | 2 | fp16mla+cudagraph | 21.17 | 42.34 | - | - | - | results/perf_v2_glm47_fp16_021_20260621_172718 |
| 0.21.0/cu126 | FP16* | TP4 | 4 | fp16mla+cudagraph | 13.25 | 53.0 | - | - | - | results/perf_v2_glm47_fp16_021_20260621_172718 |
| 0.21.0/cu126 | FP16* | TP4 | 8 | fp16mla+cudagraph | 10.35 | 82.78 | - | - | - | results/perf_v2_glm47_fp16_021_20260621_172718 |
| 0.19.0/cu128 | FP16* | TP4 | 1 | fp16mla+cudagraph | 35.36 | 35.36 | 212.489 | - | 0.452 | results/perf_v2_glm47_fp16_019_20260622_001738 |
| 0.19.0/cu128 | FP16* | TP4 | 2 | fp16mla+cudagraph | 27.62 | 55.24 | - | - | - | results/perf_v2_glm47_fp16_019_20260622_001738 |
| 0.19.0/cu128 | FP16* | TP4 | 4 | fp16mla+cudagraph | 21.37 | 85.48 | - | - | - | results/perf_v2_glm47_fp16_019_20260622_001738 |
| 0.19.0/cu128 | FP16* | TP4 | 8 | fp16mla+cudagraph | 17.5 | 140.01 | - | - | - | results/perf_v2_glm47_fp16_019_20260622_001738 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->
