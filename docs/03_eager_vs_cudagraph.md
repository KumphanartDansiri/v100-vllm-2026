# Chapter 3 — CUDAGraph & baselines: what number is even fair to quote?

> **Status: DRAFT** — numbers provisional until the final freeze ([FINAL_RERUN.md](FINAL_RERUN.md)). Table auto-renders from `data/eager_vs_cudagraph.csv`.

This is the comparison contract for the whole series. One rule:

> **Eager is for diagnosis. CUDAGraph is the serving baseline.** Never compare across modes.

## The two modes
- **Eager:** every op dispatched from Python, per step. On a fast GPU that overhead hides; on V100 it
  dominates — so eager *understates* real serving speed by several×.
- **CUDAGraph** (`cudagraph_mode=FULL_DECODE_ONLY`): the decode step is captured once and replayed on
  the GPU with no Python in the loop. This is how you actually serve, and it's what every headline
  number in this write-up uses. (It works on Volta — which surprises people who assume otherwise.)

## The measured pair (same model, TP, prompt, harness — only the mode changes)

<!-- render:eager_cudagraph -->
| model | eager tok/s | cudagraph tok/s | improvement |
|---|---|---|---|
| Qwen/Qwen3.6-27B | 7.13 | 39.09 | 5.48x |
| Qwen/Qwen3.6-35B-A3B-FP8 | 7.23 | 70.57 | 9.76x |
| zai-org/GLM-4.7-Flash | 6.0 | 37.2 | 6.20x |
<!-- endrender -->

This is **not a fleet census** — it's the clean paired A/Bs we have: same model, same harness, same
prompt/window, only eager vs cudagraph changes. Two things to read off them:
1. **CUDAGraph is 5–10× faster than eager** on the exact same setup. If you benchmark V100 in eager
   and conclude it's too slow, that's the mistake — you measured Python, not the GPU.
2. **The gain is model-dependent, and that's the point** — 5.5× for dense Qwen3.6-27B, 9.8× for the
   Qwen3.6-35B-A3B MoE, 6.2× for the GLM-4.7-Flash MLA-MoE. There's no single multiplier; the common
   lesson is that **eager decode measures launch overhead as much as model speed.** The tell: all three
   land at ~6–7 tok/s *in eager* (within ~1 tok/s of each other despite wildly different architectures)
   — per-step launch overhead dominates — and only once cudagraph replays it away do their real speeds
   (39 / 71 / 37) appear. The MoE gains most (more per-step kernel launches to erase); for
   **GLM-4.7-Flash, cudagraph is not optional but mandatory** — 6 tok/s eager is unusable, and only
   cudagraph brings it into the usable band.

## Why this governs every other chapter
All Chapter 1 / FP8 / model-page numbers are cudagraph; the MoE fix (Ch2) and MTP (Ch4) A/Bs are
cudagraph-vs-cudagraph. We never compare an eager arm to a cudagraph arm. When you see *any* V100
number — ours or anyone's — the first question is "eager or cudagraph?"

*Evidence: `results/eager_vs_cudagraph_20260613_121255/` (Qwen pairs) and
`results/glm47_mla_v100_20260615/` (GLM-4.7-Flash eager/cudagraph A/B) — SUMMARY files + per-mode
serve logs. Paired by construction: identical model/TP/prompt/harness, only `cudagraph_mode` differs.*
