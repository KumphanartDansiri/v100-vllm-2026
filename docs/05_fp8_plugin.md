# Chapter 5 — The FP8 plugin: fitting bigger models on V100, not magic speed

> **Status: DRAFT** — numbers provisional until the final freeze ([FINAL_RERUN.md](FINAL_RERUN.md)). Tables auto-render from `data/benchmark_matrix.csv`.

V100 has no native FP8. So our plugin is **W8A16**: FP8 weights stay **resident in HBM** (half the
bytes) and are dequantized to FP16 *inside the kernel*, on the fly, for the matmul. The win is
**memory and bandwidth**, not new compute — and that framing predicts exactly where it helps.

## What it is
- Custom sm_70 CUDA kernels (a coalesced FP8→FP16 GEMV for decode, grouped variants for MoE experts).
- Weights resident as FP8 → **~half the VRAM** of FP16 (verified: Qwen3.6-27B-FP8 ≈ 27 GB vs 52 GB).
- Drop-in under vLLM 0.21 via the `fp8_w8a16_sm70` package; FP16 activations, so quality is "Stable"
  vs FP16 (different numerics, coherent output — see methodology).

## Where it wins, where it doesn't (the arch-fit rule)
FP8-W8A16 wins **iff the model is bandwidth-bound and the weights barely fit (or don't) in FP16**:
- **Large MoE** (low active/total ratio): decode reads only a few experts' worth of bytes, halved →
  faster *and* the model fits. This is the sweet spot — and the only way the 122B-A10B fits at all.
- **Dense decode:** roughly a *wash* on speed vs FP16 (it's compute-bound at batch 1; dequant adds
  CUDA-core work). The value is the **lower TP floor + half memory**, not tok/s. Never serve a dense
  model in FP8 *for speed*; serve it in FP8 to *fit more on the box*.

## Numbers
Single-user baselines are in the Chapter 1 overview (the `fp8-plugin+coalesced` rows). Per-model
feasible-TP × concurrency curves are on the model pages (Chapter 6) — e.g. Qwen3.6-27B-FP8 fits at
TP≥2 and scales 22→36→45 tok/s across TP2/4/8. The flagship **Qwen3.5-122B-A10B-FP8** only exists on
V100 *because* of resident FP8 (FP16 would need ~244 GB).

## Caveats
- "Stable, not Exact" vs FP16 by construction (W8A16 changes numerics; output stays coherent).
- The plugin is **custom/local**, not upstream vLLM.
- The FP16 MoE fix (Chapter 2) does not apply here — FP8 MoE uses these kernels, not Triton fused_moe.

*Evidence: `results/ch1_20260611/` (fp8 rows), `results/tp_sweep_*` (per-model). Kernel details +
the dense-vs-MoE profile in the code repo's `docs/V100_OPTIMIZATION_FINDINGS.md`.*
