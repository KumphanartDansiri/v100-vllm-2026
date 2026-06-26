# Qwen3.5-35B-A3B (MoE — FP16 + FP8 + GPTQ-Int4) — V100 model-family page

> **Status: Final — fully tested.** Featured Qwen3.5 MoE example, the full FP16/FP8/GPTQ-Int4 triad measured at the **fleet condition** (32768 ctx, both engines vLLM 0.19 + 0.21, cu126) — the same setup as every flagship. Digest tables render from `data/benchmark_matrix.csv`; raw SSOT table at the bottom. The precision comparison, the TP2 half-GPU capacity result, and faithfulness are in [Chapter 5](../docs/05_fp8_plugin.md). (Unlike the dense 27B, this MoE's **GPTQ-Int4 is coherent** on V100.)

A sparse MoE (~3B of 35B active/token) — the featured worked example for the *MoE* side. This is where the V100 FP8 path is strongest: **FP8 wins decode at every concurrency** (the reduced weight traffic stays valuable when only ~3B params move per token — the architecture-fit case, [Chapter 5](../docs/05_fp8_plugin.md)), **and** it's the only faithful format that **fits at half TP**. The cost is prefill: the block-FP8 MoE cold-prefill is the slowest cell in the report.

## Family / checkpoints
- `Qwen/Qwen3.5-35B-A3B` — FP16 baseline (stock vLLM + the FP16-MoE config fix, [Chapter 2](../docs/02_fp16_moe_fix.md)).
- `Qwen/Qwen3.5-35B-A3B-FP8` — FP8 W8A16 plugin path (grouped/coalesced MoE decode).
- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` — GPTQ re-quant (stock vLLM, lossy).
- Same config as Qwen3.6-35B-A3B; behavior matches at equal TP ([Chapter 5](../docs/05_fp8_plugin.md)).

## Fit / compatibility
- **FP16 (~66 GB):** runs at **TP4**; **hard OOM at TP2** — ~33 GB of weights per GPU exceeds the 32 GB card *before* any KV cache (Ch.5). FP16 simply cannot make a half-GPU deployment.
- **FP8 (~34 GB, resident):** **TP4 and TP2** both fit (~17 GB/GPU at TP2) — FP8 is what enables the half-GPU MoE.
- **GPTQ-Int4 (~22 GB):** TP4 and TP2 fit; stock vLLM (engine-matched 0.19/0.21+cu126).

## Single-user deployment summary
*What one stream expects at C1 — decode throughput per engine.*

<!-- render:single_user:qwen3_5_35b_a3b -->
| Choice | 0.19 C1 Decode | 0.21 C1 Decode |
|---|---:|---:|
| FP16* TP4 | 63.51 tok/s | 56.19 tok/s |
| FP8 TP4 | 90.45 tok/s | 74.86 tok/s |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

FP8 leads FP16 substantially at one user (and the gap *widens* under load — see concurrency). The **half-GPU FP8 TP2** option still serves a strong stream; FP16 is not on the board at TP2.

## First-token latency (TTFT)
*Single-stream cold first-token (worst case) and the FA-V100 variant.*

<!-- render:ttft:qwen3_5_35b_a3b -->
| Choice | Engine | Cold First Token | FA-on Cold | Prefix-cache Hit |
|---|---|---:|---:|---:|
| FP16* TP4 | 0.19 | 14.62 s | — | — |
|  | 0.21 | 14.22 s | 9.08 s | — |
| FP8 TP4 | 0.19 | 55.65 s | — | — |
|  | 0.21 | 56.68 s | 50.01 s | — |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

This is the FP8 cost side and it is steep: the **block-FP8 MoE cold prefill is ~55 s vs ~14 s for FP16** — the unoptimized FP8 MoE prefill path (Volta WMMA). Unlike the dense model, **FA-V100 barely helps** (55→50 s) because the bottleneck here is FP8-MoE prefill *compute*, not attention. For long-context / short-output traffic, FP16 or Int4 prefill far cheaper; for decode-heavy serving the cost is paid once and FP8's decode + capacity win dominates.

## Concurrency shape
*At the same TP, how precision scales C1→C8. **Per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:qwen3_5_35b_a3b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16* TP4 | Per-user | 63.51 | 44.23 | 35.28 | 28.02 |
|  | Aggregate | 63.51 | 88.46 | 141.14 | 224.12 |
| 0.19 FP8 TP4 | Per-user | 90.45 | 74.55 | 66.07 | 51.81 |
|  | Aggregate | 90.45 | 149.11 | 264.30 | 414.49 |
| 0.21 FP16* TP4 | Per-user | 56.19 | 39.98 | 26.94 | 21.39 |
|  | Aggregate | 56.19 | 79.95 | 107.74 | 171.14 |
| 0.21 FP8 TP4 | Per-user | 74.86 | 63.38 | 58.77 | 45.80 |
|  | Aggregate | 74.86 | 126.76 | 235.07 | 366.41 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

**FP8 wins per-user at every concurrency** and the margin grows with load (1.33× at C1 → 2.14× at C8) — FP16-MoE decode collapses under concurrency while FP8 holds. GPTQ-Int4 is fastest raw and — unlike the dense 27B — **coherent** here (the MoE quantizes cleanly); the full FP16/FP8/Int4 triad is in [Chapter 5](../docs/05_fp8_plugin.md).

## Caveats
- **Correctness battery (fleet):** all three precisions are coherent (4/5 tests pass; the 5th is a JSON-format nitpick). FP8-vs-FP16 greedy output is **Stable** (coherent, token-divergent — different numerics, not errors). **GPTQ-Int4 is coherent on this MoE** (the dense 27B-Int4, by contrast, is degenerate — see its page).
- `max-model-len=32768` (the fleet condition). **TP2 half-GPU capacity** (FP16 hard-OOMs, FP8/Int4 fit) is a 4096-ctx sub-study in [Chapter 5](../docs/05_fp8_plugin.md).
- Prefill latency is the current-state limit (see TTFT); decode + capacity are the wins.
- **Both engines measured** (0.19 + 0.21): 0.19 runs ~20% faster, but the precision ordering is identical — see the dual-engine triad in [Chapter 5](../docs/05_fp8_plugin.md).

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv` for auditability. If a digest and these rows disagree, **the SSOT rows win**.*

<!-- render:model:Qwen3.5-35B-A3B -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | FP16* | TP4 | 1 | +moe_patch | 56.19 | 56.19 | 14.22 | 9.08 | - | results/perf_v2_q35b35_fp16_021_20260626_050335 |
| 0.21.0/cu126 | FP16* | TP4 | 2 | +moe_patch | 39.98 | 79.95 | - | - | - | results/perf_v2_q35b35_fp16_021_20260626_050335 |
| 0.21.0/cu126 | FP16* | TP4 | 4 | +moe_patch | 26.94 | 107.74 | - | - | - | results/perf_v2_q35b35_fp16_021_20260626_050335 |
| 0.21.0/cu126 | FP16* | TP4 | 8 | +moe_patch | 21.39 | 171.14 | - | - | - | results/perf_v2_q35b35_fp16_021_20260626_050335 |
| 0.19.0/cu126 | FP16* | TP4 | 1 | +moe_patch | 63.51 | 63.51 | 14.62 | - | - | results/perf_v2_q35b35_fp16_019_20260626_060936 |
| 0.19.0/cu126 | FP16* | TP4 | 2 | +moe_patch | 44.23 | 88.46 | - | - | - | results/perf_v2_q35b35_fp16_019_20260626_060936 |
| 0.19.0/cu126 | FP16* | TP4 | 4 | +moe_patch | 35.28 | 141.14 | - | - | - | results/perf_v2_q35b35_fp16_019_20260626_060936 |
| 0.19.0/cu126 | FP16* | TP4 | 8 | +moe_patch | 28.02 | 224.12 | - | - | - | results/perf_v2_q35b35_fp16_019_20260626_060936 |
| 0.21.0/cu126 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 74.86 | 74.86 | 56.68 | 50.01 | - | results/perf_v2_q35b35_fp8_021_20260626_051726 |
| 0.21.0/cu126 | FP8 | TP4 | 2 | fp8-plugin+coalesced | 63.38 | 126.76 | - | - | - | results/perf_v2_q35b35_fp8_021_20260626_051726 |
| 0.21.0/cu126 | FP8 | TP4 | 4 | fp8-plugin+coalesced | 58.77 | 235.07 | - | - | - | results/perf_v2_q35b35_fp8_021_20260626_051726 |
| 0.21.0/cu126 | FP8 | TP4 | 8 | fp8-plugin+coalesced | 45.80 | 366.41 | - | - | - | results/perf_v2_q35b35_fp8_021_20260626_051726 |
| 0.19.0/cu126 | FP8 | TP4 | 1 | fp8-plugin+coalesced | 90.45 | 90.45 | 55.65 | - | - | results/perf_v2_q35b35_fp8_019_20260626_061725 |
| 0.19.0/cu126 | FP8 | TP4 | 2 | fp8-plugin+coalesced | 74.55 | 149.11 | - | - | - | results/perf_v2_q35b35_fp8_019_20260626_061725 |
| 0.19.0/cu126 | FP8 | TP4 | 4 | fp8-plugin+coalesced | 66.07 | 264.30 | - | - | - | results/perf_v2_q35b35_fp8_019_20260626_061725 |
| 0.19.0/cu126 | FP8 | TP4 | 8 | fp8-plugin+coalesced | 51.81 | 414.49 | - | - | - | results/perf_v2_q35b35_fp8_019_20260626_061725 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 1 | stock-vllm | 78.38 | 78.38 | 16.60 | - | - | results/perf_v2_q35b35_int4_021_20260626_053257 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 2 | stock-vllm | 64.23 | 128.45 | - | - | - | results/perf_v2_q35b35_int4_021_20260626_053257 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 4 | stock-vllm | 60.11 | 240.44 | - | - | - | results/perf_v2_q35b35_int4_021_20260626_053257 |
| 0.21.0/cu126 | GPTQ-Int4 | TP4 | 8 | stock-vllm | 54.47 | 435.75 | - | - | - | results/perf_v2_q35b35_int4_021_20260626_053257 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 1 | stock-vllm | 95.35 | 95.35 | 16.96 | - | - | results/perf_v2_q35b35_int4_019_20260626_062609 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 2 | stock-vllm | 75.58 | 151.17 | - | - | - | results/perf_v2_q35b35_int4_019_20260626_062609 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 4 | stock-vllm | 68.31 | 273.24 | - | - | - | results/perf_v2_q35b35_int4_019_20260626_062609 |
| 0.19.0/cu126 | GPTQ-Int4 | TP4 | 8 | stock-vllm | 62.36 | 498.91 | - | - | - | results/perf_v2_q35b35_int4_019_20260626_062609 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->
