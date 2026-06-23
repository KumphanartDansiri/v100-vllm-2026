# GLM-4.5-Air (MoE — FP8) — V100 model-family page

> **Status: DRAFT** — provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)). Digest tables render from `data/benchmark_matrix.csv` (perf_v2-frozen rows only); the exhaustive raw SSOT table is at the bottom.

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
*What one stream expects at C1 — decode throughput on each engine, plus representative 0.21 cold first-token latency.*

<!-- render:single_user:glm4_5_air -->
| Choice | 0.19 C1 decode | 0.21 C1 decode | 0.21 Cold TTFT | 0.21 Warm TTFT¹ |
|---|---:|---:|---:|---:|
| FP8 TP8 | 64.67 tok/s | 65.45 tok/s | 42.06 s | pending |

¹ **Warm TTFT** = warm / prefix-cache-hit / chunked-prefill serving latency — **pending SSOT refresh**. **Cold TTFT** is cold *monolithic* prefill from the representative SSOT row: a **worst-case** number, *not* warm serving latency — don't read it as steady interactive response.
<!-- endrender -->

**~65 tok/s single-stream** on a 106B model, on 7-year-old GPUs.

## Concurrency shape
*How it scales C1→C8 at TP8. Each config has two rows: **per-user** = one stream; **aggregate** = total
box throughput.*

<!-- render:concurrency:glm4_5_air -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP8 TP8 | per-user | 64.67 | 50.63 | 38.06 | 25.97 |
|  | aggregate | 64.67 | 101.26 | 152.24 | 207.79 |
| 0.21 FP8 TP8 | per-user | 65.45 | 51.69 | 38.0 | 26.0 |
|  | aggregate | 65.45 | 103.38 | 152.0 | 208.0 |
<!-- endrender -->

Aggregate scales 65 → ~103 → ~152 → **~208** at C8 while per-user holds ~26 (above the ~20 floor) —
**8 users is the best overall throughput** point. The 0.19-vs-0.21 spread is run-to-run scheduling
noise, not a systematic gap (the FP8 compute is our engine-invariant kernels).

## Caveats
- Exactness is **Stable** by construction (FP8 vs FP16 = different numerics); MoE@TP8 output isn't
  run-to-run bit-identical. The 5-test suite is all coherent/correct on both engines.
- **Reasoning model** (`<think>…</think>`): use natural stop in production; the `ignore_eos`
  measurement window can show mild tail repetition (a measurement choice, not a failure).
- **Cold TTFT ~42 s** (cold monolithic prefill at 32k); warm / prefix-cache-hit is far lower.
  `max-model-len=32768`.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability. The digests are the
recommended reading; if a digest and these rows ever disagree, **the SSOT rows win** and the
renderer/prose is fixed.*

<!-- render:model:GLM-4.5-Air -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp8 | TP8 | 1 | fp8-plugin+coalesced | 65.45 | 65.45 | 42.06 | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.21.0/cu126 | fp8 | TP8 | 2 | fp8-plugin+coalesced | 51.69 | 103.38 | - | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.21.0/cu126 | fp8 | TP8 | 4 | fp8-plugin+coalesced | 38.0 | 152.0 | - | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.21.0/cu126 | fp8 | TP8 | 8 | fp8-plugin+coalesced | 26.0 | 208.0 | - | results/perf_v2_glm_fp8_021_20260620_194734 |
| 0.19.0/cu126 | fp8 | TP8 | 1 | fp8-plugin+coalesced | 64.67 | 64.67 | 42.09 | results/perf_v2_glm_fp8_019_20260620_220710 |
| 0.19.0/cu126 | fp8 | TP8 | 2 | fp8-plugin+coalesced | 50.63 | 101.26 | - | results/perf_v2_glm_fp8_019_20260620_220710 |
| 0.19.0/cu126 | fp8 | TP8 | 4 | fp8-plugin+coalesced | 38.06 | 152.24 | - | results/perf_v2_glm_fp8_019_20260620_220710 |
| 0.19.0/cu126 | fp8 | TP8 | 8 | fp8-plugin+coalesced | 25.97 | 207.79 | - | results/perf_v2_glm_fp8_019_20260620_220710 |
<!-- endrender -->
