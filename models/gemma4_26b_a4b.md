# gemma-4-26B-A4B-it (MoE — FP16 + FP8) — V100 model-family page

> **Status: DRAFT** — provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)). Digest tables render from `data/benchmark_matrix.csv` (perf_v2-frozen rows only); the exhaustive raw SSOT table is at the bottom.

A **26B-total / 4B-active MoE** (`Gemma4ForConditionalGeneration`). The model that most clearly
justifies carrying vLLM 0.21: its **FP8 MoE runs only on 0.21** (the `gemma4.py` fused-MoE path errors
on 0.19). Where it runs, FP8 beats FP16 at every concurrency.

## Family / checkpoints
- `google/gemma-4-26B-A4B-it` — FP16 baseline (**requires the Volta MoE patch**, Chapter 2).
- `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` — FP8 plugin path (**0.21 only**).
- **Compatibility:** needs **transformers 5** on 0.19 (`vllm019-tf5`); runs on 0.21 stock. **FP8 is
  0.21-only**; FP16-MoE needs the MoE patch. (Chapter 6 matrix.)

## Fit / compatibility
- **TP4**, cudagraph, `--dtype float16`, `--skip-mm-profiling`, `--max-num-batched-tokens ≥ 2496`
  (0.21), `max-model-len=32768`.
- **FP8 runs only on 0.21** — the `gemma4.py` MoE path errors on 0.19 even with tf5.
  **FP16-MoE requires the Volta MoE patch** (Chapter 2).
- **FP8 also fits at TP2** (2 cards, short context) — 0.21.
- **Best engine for FP8: 0.21 (the only one)**; for FP16, 0.21 is also marginally faster (the rare
  case where 0.21 FP16 > 0.19 FP16).

## Single-user deployment summary
*What one stream expects at C1 — decode throughput on each engine, plus representative 0.21 cold first-token latency. FP8 is 0.21-only (— on 0.19).*

<!-- render:single_user:gemma4_26b_a4b -->
| Choice | 0.19 C1 decode | 0.21 C1 decode | 0.21 Cold TTFT | 0.21 Warm TTFT¹ |
|---|---:|---:|---:|---:|
| FP16 TP4 | 39.47 tok/s | 44.39 tok/s | 53.60 s | pending |
| FP8 TP4 | — | 72.85 tok/s | 50.64 s | pending |
| FP8 TP2 | — | 60.15 tok/s | — | pending |

¹ **Warm TTFT** = warm / prefix-cache-hit / chunked-prefill serving latency — **pending SSOT refresh**. **Cold TTFT** is cold *monolithic* prefill from the representative SSOT row: a **worst-case** number, *not* warm serving latency — don't read it as steady interactive response.
<!-- endrender -->

On 0.21, **FP8 ≫ FP16** (72.9 vs 44.4); the half-GPU **FP8 TP2** (60 tok/s) fits on 2 cards.

## Concurrency shape
*At a comparable serving config (same TP), how precision/engine scales C1→C8. Each config has two rows:
**per-user** = one stream; **aggregate** = total box throughput.*

<!-- render:concurrency:gemma4_26b_a4b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16 TP4 | per-user | 39.47 | 32.98 | 28.35 | 20.41 |
|  | aggregate | 39.47 | 65.96 | 113.4 | 163.26 |
| 0.21 FP16 TP4 | per-user | 44.39 | 35.53 | 23.16 | 21.61 |
|  | aggregate | 44.39 | 71.06 | 92.64 | 172.86 |
| 0.21 FP8 TP4 | per-user | 72.85 | 59.26 | 48.4 | 33.51 |
|  | aggregate | 72.85 | 118.52 | 193.6 | 268.05 |
<!-- endrender -->

**FP8 wins every concurrency** (C8 aggregate 268 vs 173) — the sparse-MoE arch-fit win. (FP8 is shown
on 0.21 only; FP16 on both, where 0.21 edges 0.19 — against the fleet pattern.)

## Caveats
- Both FP16 and FP8 are **Exact** (quality=pass); all categories coherent.
- **FP8 is 0.21-only** (0.19 `gemma4.py` MoE error); FP16 needs the Volta MoE patch + tf5.
- **Cold TTFT ~50–54 s** (Gemma-4 vision tower + untuned prefill). `max-model-len=32768`.

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability. The digests are the
recommended reading; if a digest and these rows ever disagree, **the SSOT rows win** and the
renderer/prose is fixed.*

<!-- render:model:gemma-4-26B-A4B-it -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 10.2 | - | 0.3 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 46.48 | - | 0.18 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 10.91 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock(pre-moe-patch) | 3.58 | 28.3 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(heuristic) | 43.66 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(heuristic) | 19.1 | 145.15 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(tuned-json) | 43.71 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(tuned-json) | 20.23 | 155.94 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 72.85 | 72.85 | 50.64 | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 59.26 | 118.52 | - | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 48.4 | 193.6 | - | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 33.51 | 268.05 | - | results/perf_v2_g26b_fp8_021_20260621_044847 |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch | 44.39 | 44.39 | 53.60 | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | +moe_patch | 35.53 | 71.06 | - | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | +moe_patch | 23.16 | 92.64 | - | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch | 21.61 | 172.86 | - | results/perf_v2_g26b_fp16_021_20260621_045653 |
| 0.19.0/cu128 | fp16 | TP4 | 1 | +moe_patch | 39.47 | 39.47 | 53.34 | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.19.0/cu128 | fp16 | TP4 | 2 | +moe_patch | 32.98 | 65.96 | - | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.19.0/cu128 | fp16 | TP4 | 4 | +moe_patch | 28.35 | 113.4 | - | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.19.0/cu128 | fp16 | TP4 | 8 | +moe_patch | 20.41 | 163.26 | - | results/perf_v2_g26b_fp16_019_20260620_224851 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 60.15 | 60.15 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 46.37 | 92.74 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 32.6 | 130.4 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 17.54 | 140.34 | - | results/perf_v2_g26b2_fp8_021_20260622_005415 |
<!-- endrender -->
