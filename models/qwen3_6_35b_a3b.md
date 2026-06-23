# Qwen3.6-35B-A3B (MoE — FP16 + FP8) — V100 model-family page

> **Status: DRAFT** — provisional until the final freeze ([../docs/FINAL_RERUN.md](../docs/FINAL_RERUN.md)). Digest tables render from `data/benchmark_matrix.csv` (perf_v2-frozen rows only); the exhaustive raw SSOT table is at the bottom.

A **35B-total / 3B-active MoE** (active/total ≈ 0.09 — deep in the "sparse-MoE arch-fit" zone where FP8
W8A16 wins). The FP8 checkpoint is Qwen **block-FP8**. This is the **MoE showcase**: FP8 beats FP16 at
*every* concurrency on V100 — sparse activation keeps decode bandwidth-bound, where FP8's half-the-bytes
pays off and never hits the dense CUDA-core wall (each token touches only a few experts).

## Family / checkpoints
- `Qwen/Qwen3.6-35B-A3B` — FP16 baseline (**requires the Volta MoE patch**, Chapter 2).
- `Qwen/Qwen3.6-35B-A3B-FP8` — FP8 plugin path (block-FP8).
- **Compatibility:** runs on vLLM **0.19 and 0.21 stock** (no transformers-5 upgrade); FP16-MoE needs
  the **MoE patch**; the FP8 plugin works on **both engines**. (Chapter 6 matrix.)

## Fit / compatibility
- **TP4** on 4×V100, cudagraph, `--dtype float16`, `--skip-mm-profiling`, `max-model-len=32768`. FP8
  adds the plugin (`VLLM_V100_CT_FP8_RESIDENT=1` + `VLLM_V100_FP8_COALESCED_GEMV=1`).
- **FP16 MoE requires the Volta MoE patch** (Chapter 2) or it craters to ~3 tok/s @C8.
- **FP8 also fits at TP2** (2 cards, short context ≤ 8192) — the half-GPU option.
- **Best engine:** 0.19 for decode throughput (faster at every point); 0.21 also works.

## Single-user deployment summary
*What one stream gets at C1, per engine — the precision/TP choice for a solo user or small lab.*

<!-- render:single_user:qwen3_6_35b_a3b -->
| vLLM | FP16 TP4 | FP8 TP4 | FP8 TP2 |
|---|---:|---:|---:|
| 0.19 | 63.3 | 89.92 | — |
| 0.21 | 55.93 | 74.92 | 71.0 |
<!-- endrender -->

**FP8 ≫ FP16 at C1** (90 vs 63 on 0.19); the **half-GPU FP8 TP2** option (0.21) still serves 71 tok/s,
so you can run this MoE on 2 cards. (FP16 here is the *patched* path; stock FP16-MoE is far slower.)

## Concurrency shape
*At a comparable serving config (same TP), how precision/engine scales C1→C8. Each config has two rows:
**per-user** = one stream's experience; **aggregate** = total box throughput.*

<!-- render:concurrency:qwen3_6_35b_a3b -->
| Config | Type | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| 0.19 FP16 TP4 | per-user | 63.3 | 44.01 | 34.06 | 28.25 |
|  | aggregate | 63.3 | 88.02 | 136.24 | 225.98 |
| 0.19 FP8 TP4 | per-user | 89.92 | 74.48 | 68.85 | 51.41 |
|  | aggregate | 89.92 | 148.96 | 275.4 | 411.32 |
| 0.21 FP16 TP4 | per-user | 55.93 | 39.97 | 26.77 | 21.4 |
|  | aggregate | 55.93 | 79.94 | 107.08 | 171.17 |
| 0.21 FP8 TP4 | per-user | 74.92 | 63.35 | 58.76 | 47.7 |
|  | aggregate | 74.92 | 126.7 | 235.04 | 381.59 |
<!-- endrender -->

**FP8 wins at every concurrency** — per-user *and* aggregate, C1 through C8 (C8 aggregate 411 vs 226 on
0.19). The sparse MoE never hits the dense CUDA-core wall, so FP8's half-the-bytes pays off at all batch
sizes. This is the headline V100 result.

## Caveats
- FP8 MoE is **Stable** (coherent, not run-to-run bit-identical at TP — expert/all-reduce ordering, as
  on stock FP16 MoE); FP16 is **Exact**.
- FP16 rows are the **+MoE-patch** path; stock FP16-MoE is far slower on Volta (Chapter 2).
- `max-model-len=32768`; the TP2 half-GPU option is short-context (8192).

## Raw SSOT rows
*Rendered directly from `data/benchmark_matrix.csv`, kept for auditability — it includes the earlier
Ch1 / MoE-A/B rows the digests above omit. The digests are the recommended reading; if a digest and
these rows ever disagree, **the SSOT rows win** and the renderer/prose is fixed.*

<!-- render:model:Qwen3.6-35B-A3B -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 15.44 | - | 0.74 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 67.6 | - | 1.86 | results/ch1_20260611/ch1.1_021/manifest.csv |
| 0.21.0/cu126 | fp16 | TP4 | 1 | stock(pre-moe-patch) | 15.56 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | stock(pre-moe-patch) | 3.16 | 24.93 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(heuristic) | 65.91 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(heuristic) | 20.98 | 137.2 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch(tuned-json) | 65.85 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch(tuned-json) | 22.8 | 173.92 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| 0.21.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 74.92 | 74.92 | 14.28 | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 63.35 | 126.7 | - | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 58.76 | 235.04 | - | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 47.7 | 381.59 | - | results/perf_v2_q35b_fp8_021_20260620_183825 |
| 0.21.0/cu126 | fp16 | TP4 | 1 | +moe_patch | 55.93 | 55.93 | 12.65 | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.21.0/cu126 | fp16 | TP4 | 2 | +moe_patch | 39.97 | 79.94 | - | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.21.0/cu126 | fp16 | TP4 | 4 | +moe_patch | 26.77 | 107.08 | - | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.21.0/cu126 | fp16 | TP4 | 8 | +moe_patch | 21.4 | 171.17 | - | results/perf_v2_q35b_fp16_021_20260620_185300 |
| 0.19.0/cu126 | fp8 | TP4 | 1 | fp8-plugin+coalesced | 89.92 | 89.92 | 14.17 | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp8 | TP4 | 2 | fp8-plugin+coalesced | 74.48 | 148.96 | - | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp8 | TP4 | 4 | fp8-plugin+coalesced | 68.85 | 275.4 | - | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp8 | TP4 | 8 | fp8-plugin+coalesced | 51.41 | 411.32 | - | results/perf_v2_q35b_fp8_019_20260620_205633 |
| 0.19.0/cu126 | fp16 | TP4 | 1 | +moe_patch | 63.3 | 63.3 | 12.66 | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.19.0/cu126 | fp16 | TP4 | 2 | +moe_patch | 44.01 | 88.02 | - | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.19.0/cu126 | fp16 | TP4 | 4 | +moe_patch | 34.06 | 136.24 | - | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.19.0/cu126 | fp16 | TP4 | 8 | +moe_patch | 28.25 | 225.98 | - | results/perf_v2_q35b_fp16_019_20260620_210830 |
| 0.21.0/cu126 | fp8 | TP2 | 1 | fp8-plugin+coalesced | 71.0 | 71.0 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 2 | fp8-plugin+coalesced | 56.51 | 113.02 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 4 | fp8-plugin+coalesced | 48.25 | 193.0 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
| 0.21.0/cu126 | fp8 | TP2 | 8 | fp8-plugin+coalesced | 36.75 | 294.03 | - | results/perf_v2_q35b2_fp8_021_20260622_003941 |
<!-- endrender -->
