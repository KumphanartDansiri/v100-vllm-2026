# GLM-4.5-Air (MoE — FP8) — V100 model-family page

> **Status: Final** — numbers frozen from `data/benchmark_matrix.csv` (perf_v2 rows, tag `fp8-v100-2026-matrix`); digest tables auto-render and the exhaustive raw SSOT table is at the bottom. Refresh procedure: [../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md).

A **106B-total / 12B-active MoE** (`Glm4MoeForCausalLM`), published as compressed-tensors channel
W8A8-FP8. The first large MoE that fits *and serves at a comfortable rate* on 8×V100 — the headline win
of the FP8 plugin.

## Family / checkpoints
- `zai-org/GLM-4.5-Air-FP8` — the only checkpoint (FP8); served via the plugin at TP8.
- **Compatibility:** runs on vLLM **0.19 and 0.21 stock** — **no transformers-5 upgrade** (stock
  tf 4.57; config, tokenizer, and full generation all verified). FP8 plugin on both engines.
  (Chapter 6 matrix.)

## Fit / compatibility
- **TP8, FP8-resident** (~8.3 GB/GPU after freeing the transient FP16 w13); large KV headroom (tens of
  concurrent 32k-context streams).
- cudagraph (`mode=0` + `FULL_DECODE_ONLY` + `TRITON_ATTN`), coalesced decode GEMV (attention/dense +
  grouped MoE-w13) + the FP16-MoE Volta tune, `--skip-mm-profiling`, `max-model-len=32768`, `ns=8`.
- **Best engine:** GLM-4.5-Air is the **lone near-tie where 0.21 is marginally ahead** (65.45 vs 64.67
  @C1), against the fleet pattern of 0.19 winning decode.

## Single-user deployment summary
*What one stream expects at C1 — decode throughput on each engine.*

<!-- render:single_user:glm4_5_air -->
| Choice | Type | 0.19 | 0.21 |
|---|---|:---:|:---:|
| FP8 TP8 | Decode | 64.67 tok/s | 65.45 tok/s |
|  | Exactness | ✓ | ✗ |
|  | Correctness | ✓ | ✓ |

_**Decode** = per-user tok/s at C1. **Exactness** ✓ = bit-identical run-to-run (temp 0). **Correctness** ✓ = coherent, usable output. So ✗ exactness / ✓ correctness = not bit-exact but coherent (e.g. FP8/MoE routing drift — expected, not an error); ✗ / ✗ = degenerate output (the GPTQ-Int4 27B case)._
<!-- endrender -->

**~65 tok/s single-stream** on a 106B model, on 7-year-old GPUs.

## First-token latency (TTFT)
*Single-stream time to first token: **cold** (a fresh, cache-cold full prefill — worst case) vs **prefix-cache-hit** (repeated / shared context — best case). Decode is the headline; this is the latency side.*

<!-- render:ttft:glm4_5_air -->
| Choice | Engine | Cold First Token | FA-on Cold | Prefix-cache Hit |
|---|---|---:|---:|---:|
| FP8 TP8 | 0.19 | 68.788 s | — | 0.737 s |
|  | 0.21 | 66.942 s | 49.35 s | 0.752 s |

All TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen **block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.
<!-- endrender -->

## Concurrency shape
*How it scales C1→C8 at TP8. Each config has two rows: **per-user** = one stream; **aggregate** = total
box throughput.*

<!-- render:concurrency:glm4_5_air -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP8 TP8 | Per-user | 64.67 | 50.63 | 38.06 | 25.97 |
|  | Aggregate | 64.67 | 101.26 | 152.24 | 207.79 |
| 0.21 FP8 TP8 | Per-user | 65.45 | 51.69 | 38.0 | 26.0 |
|  | Aggregate | 65.45 | 103.38 | 152.0 | 208.0 |
<!-- endrender -->

Aggregate scales 65 → ~103 → ~152 → **~208** at C8 while per-user holds ~26 (above the ~20 floor) —
**8 users is the best overall throughput** point. The 0.19-vs-0.21 spread is run-to-run scheduling
noise, not a systematic gap (the FP8 compute is our engine-invariant kernels).

## Caveats
- Exactness is **Stable** by construction (FP8 vs FP16 = different numerics); MoE@TP8 output isn't
  run-to-run bit-identical. The 5-test suite is all coherent/correct on both engines.
- **Reasoning model** (`<think>…</think>`): use natural stop in production; the `ignore_eos`
  measurement window can show mild tail repetition (a measurement choice, not a failure).
- **Cold first-token ~67 s** (cache-cold full prefill at ~22.6k tokens; FlashAttention cuts it to
  ~49 s); prefix-cache-hit is sub-second. See the TTFT section. `max-model-len=32768`.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability. The digests are the
recommended reading; if a digest and these rows ever disagree, **the SSOT rows win** and the
renderer/prose is fixed.*

<!-- render:model:GLM-4.5-Air -->
| vLLM | Variant | TP | Users | Config | Per-user | Aggregate | Cold TTFT | FA Cold | Prefix Hit | Result path |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | FP8 | TP8 | 1 | fp8-plugin+coalesced | 65.45 | 65.45 | 66.942 | 49.35 | 0.752 | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.21.0/cu126 | FP8 | TP8 | 2 | fp8-plugin+coalesced | 51.69 | 103.38 | - | - | - | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.21.0/cu126 | FP8 | TP8 | 4 | fp8-plugin+coalesced | 38.0 | 152.0 | - | - | - | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.21.0/cu126 | FP8 | TP8 | 8 | fp8-plugin+coalesced | 26.0 | 208.0 | - | - | - | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.19.0/cu126 | FP8 | TP8 | 1 | fp8-plugin+coalesced | 64.67 | 64.67 | 68.788 | - | 0.737 | results/perf_v2_glm_fp8_019_20260620_220710 |
| 0.19.0/cu126 | FP8 | TP8 | 2 | fp8-plugin+coalesced | 50.63 | 101.26 | - | - | - | results/perf_v2_glm_fp8_019_20260620_220710 |
| 0.19.0/cu126 | FP8 | TP8 | 4 | fp8-plugin+coalesced | 38.06 | 152.24 | - | - | - | results/perf_v2_glm_fp8_019_20260620_220710 |
| 0.19.0/cu126 | FP8 | TP8 | 8 | fp8-plugin+coalesced | 25.97 | 207.79 | - | - | - | results/perf_v2_glm_fp8_019_20260620_220710 |
<!-- endrender -->
