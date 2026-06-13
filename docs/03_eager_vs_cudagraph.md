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
| model | mode | tok/s | relative | result_path |
|---|---|---|---|---|
| q27b fp16 | eager | 7.13 | 1.00x | results/eager_vs_cudagraph_20260613_121255/q27b_eager.log |
| q27b fp16 | cudagraph | 39.09 | 5.48x | results/eager_vs_cudagraph_20260613_121255/q27b_cudagraph.log |
| q35b fp8 | eager | 7.23 | 1.00x | results/eager_vs_cudagraph_20260613_121255/q35b_eager.log |
| q35b fp8 | cudagraph | 70.57 | 9.76x | results/eager_vs_cudagraph_20260613_121255/q35b_cudagraph.log |
<!-- endrender -->

Two things to read off it:
1. **CUDAGraph is 5–10× faster than eager** on the exact same setup. If you benchmark V100 in eager
   and conclude it's too slow, that's the mistake — you measured Python, not the GPU.
2. **Both models run at ~7 tok/s in eager** — a 27B dense and a 35B-A3B MoE, wildly different models,
   land within 0.1 tok/s of each other. That's the tell: eager decode is bottlenecked on per-step
   *launch overhead*, not on the model. CUDAGraph removes that overhead, and only then do the models'
   real speeds (39 vs 71) appear. (The MoE gains more — 9.8× vs 5.5× — because it has more per-step
   kernel launches for eager to waste and cudagraph to erase.)

## Why this governs every other chapter
All Chapter 1 / FP8 / model-page numbers are cudagraph; the MoE fix (Ch2) and MTP (Ch4) A/Bs are
cudagraph-vs-cudagraph. We never compare an eager arm to a cudagraph arm. When you see *any* V100
number — ours or anyone's — the first question is "eager or cudagraph?"

*Evidence: `results/eager_vs_cudagraph_20260613_121255/` (SUMMARY.csv + per-mode serve logs). Paired
by construction: identical model/TP/prompt/harness, only `cudagraph_mode` differs.*
