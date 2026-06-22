# GLM-4.5-Air (FP8) — V100 model page

> **Status: MEASURED on TWO engines (vLLM 0.21 + vLLM 0.19), perf_v2 matrix 2026-06-20.** All decode
> numbers come from the SSOT (`data/benchmark_matrix.csv`); raw evidence in the companion repo under
> `results/perf_v2_glm_fp8_{021,019}_*`. Both engines run the **same** custom FP8 plugin.

GLM-4.5-Air is a **106B-total / 12B-active MoE** (`Glm4MoeForCausalLM`, 46 layers, 128 routed +
1 shared expert, `first_k_dense_replace=1`). The published checkpoint `zai-org/GLM-4.5-Air-FP8`
is **compressed-tensors channel W8A8-FP8**. It is the first large MoE in this matrix that fits
*and serves at a comfortable decode rate* on 8×V100 — the headline win of the custom FP8 plugin.

## What fits / feasible TP
- **FP8 (resident), TP8.** The plugin keeps FP8 weights resident (~8.3 GB/GPU after freeing the
  transient FP16 w13) and runs the channel Linears + MoE-w13 through coalesced sm_70 GEMV kernels.
  KV headroom is large at TP8 — comfortably tens of concurrent long-context streams (the headline
  numbers below run at `max-model-len=32768`).
- TP8 is the validated envelope (a stock CT-FP8 load otherwise degrades to a ~210 GB FP16-resident
  path on Volta that only fits across all 8 cards). See `docs/05_fp8_plugin.md`.

## Best serving config
- **TP8, cudagraph** (`mode=0` + `FULL_DECODE_ONLY` + `TRITON_ATTN`), FP8 plugin with **coalesced
  decode GEMV** (attention/dense Linears **and** grouped MoE-w13) + the FP16-MoE Volta tune,
  `--skip-mm-profiling`, `max-model-len=32768`, `ns=8`. Full env in the code repo's
  `docs/GLM45_AIR_V100_CONFIG.md`; the decode numbers here come from the perf_v2 matrix harness
  (`tools/perf_v2_matrix.sh` → `perf_v2_cell.sh` + `perf_v2_client.py`, `ENGINE=021|019`).

## Runs on BOTH engines — no special requirements
GLM-4.5-Air's architecture (`Glm4MoeForCausalLM`) is registered in **stock vLLM 0.19 and 0.21**,
and it loads + serves on **stock transformers 4.57.6 — no transformers upgrade is required**
(verified: config, tokenizer, and a full eager generation all succeed on the base 0.19 image; the
transformers-5.x build in our multi-model image is needed by *Gemma-4 / GLM-4.7-Flash*, not this
model). The **same** unified FP8 plugin drives both engines (one `apply`-override integration; the
two engines expose the identical MoE dispatch — 0.21 is a refactor/superset of 0.19's path, not a
different one).

## Correctness — the 5 standard tests (exactness + 4 categories)
Greedy (`temperature=0`), run as a dedicated correctness suite (companion repo). All five are
**coherent and correct** on both engines:

| test | category | result |
|---|---|---|
| Q1 ×5 | exactness / self-stability | coherent, rep≈0.05 (run-to-run **Stable→Exact**: 0.19 came out bit-deterministic, 0.21 Stable — expected for MoE@TP8) |
| Q2 | factual (explain a transformer) | correct, coherent |
| Q3 | reasoning (two-trains math) | correct setup (140t=320 → 11:17, ~137 km from A) |
| Q4 | structure (exactly-5 bullets) | produced exactly 5 correct bullets |
| Q5 | code (thread-safe LRU + tests) | correct design (DLL + dict + lock) |

Exactness tops out at **Stable** by construction (FP8 vs any FP16 gold = different numerics), and for
this MoE at TP8 output is not guaranteed run-to-run bit-identical (expert-reduction / all-reduce
ordering — happens on stock FP16 MoE too; the perf_v2 dual-engine pass logged 0.19=Exact, 0.21=Stable).
GLM-4.5-Air is a **reasoning model** (`<think>…</think>`); under `ignore_eos` forced-length it can show
mild tail repetition — a measurement choice for clean tok/s, not a model failure. Decode speeds are in
the concurrency scan below.

## Measured — concurrency scan (cudagraph), both engines

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

## How to read it
- **1 user = ~65 tok/s** single-stream — comfortably into the usable band, on a 106B model, on
  7-year-old GPUs.
- **8 users = best overall throughput**: ~208 tok/s aggregate while per-user stays ~26 tok/s (above
  the ~20 tok/s floor) — the recommended serving point. Aggregate scales 65 → ~103 → ~152 → ~208;
  per-user degrades gracefully (no cliff), all streams coherent.
- **Cold TTFT ~42 s** (cold monolithic prefill at 32k max-len; warm / prefix-cache-hit is far lower).
- **0.19 vs 0.21 is within ~1% on decode** — and GLM-4.5-Air is the **lone near-tie where 0.21 is
  marginally ahead** (65.45 vs 64.67 @C1; 208.0 vs 207.8 @C8 agg), against the fleet-wide pattern of
  0.19 winning decode. This is a **portability** result, **not a vLLM-version benchmark**: the FP8
  routed-expert compute is our *own* sm_70 kernels (engine-invariant), so the version barely shows. A
  genuine engine-version comparison would have to run on **FP16/BF16 MoE** (vLLM's own fused-MoE
  kernels, the path that changed across versions) — see Chapter 2.
- The mid-concurrency (2–4 user) spread between engines is run-to-run scheduling noise (it flips
  which engine "wins"), not a systematic gap.

## Caveats
- FP8 plugin is custom/local sm_70 kernels (not upstream).
- Reasoning model → use natural stop (not `ignore_eos`) for production; the suite forces a fixed
  window only to measure decode speed cleanly.
- Numbers are `max-model-len=32768`, `ns=8`; longer contexts shrink KV headroom.
- Both engines run the base **cu126** source build (0.19 here needs **no** transformers upgrade); the
  numbers are engine/transformers-invariant because the FP8 compute is the plugin's own sm_70 kernels.
