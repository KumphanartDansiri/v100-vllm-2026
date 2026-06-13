# Chapter 3 — CUDAGraph & baselines: what number is even fair to quote?

> **Status: DRAFT (prose)** — the illustrative measurement table is pending one small eager-vs-cudagraph
> run (see note); definitions below are final. ([FINAL_RERUN.md](FINAL_RERUN.md))

This is the methodological guardrail for the whole series. If you take one thing from it: **eager-mode
tok/s is not a serving baseline, and most "V100 is too slow" posts are quoting eager.**

## The two modes
- **Eager:** every operation is dispatched from Python, one at a time, every decode step. On a fast
  GPU that per-step launch overhead is hidden; on the relatively slower V100 it dominates, so eager
  *understates* real serving speed by a large factor (typically **3–5×** on these models).
- **CUDAGraph** (`cudagraph_mode=FULL_DECODE_ONLY`): the decode step is captured once and replayed on
  the GPU with no Python in the loop. **This is the practical serving baseline** — it's how you'd
  actually run a server, and it's what every headline number in this write-up uses.

CUDAGraph works on Volta (sm_70). That alone surprises people who assume "no modern features on V100."

## Why this matters for everything else here
- All Chapter 1 / model-page / FP8 numbers are **cudagraph**. If you compare them to an eager number
  you measured, you'll wrongly conclude V100 is ~4× slower than it is.
- The FP16-MoE fix (Chapter 2) and MTP (Chapter 4) are *also* measured under cudagraph — the
  comparisons are mode-consistent (we never compare an eager arm to a cudagraph arm).
- Dense vs MoE behave differently under capture (MoE routing is data-dependent and needs the fast
  route-prep path to capture at all) — but both end up captured for the headline numbers.

## Illustration (pending one clean measurement)

<!-- render:eager_cudagraph -->
_(pending: a paired eager-vs-cudagraph run on 1–2 representative models, to anchor the 3–5× claim with
a citable `result_path`. The bring-up smoke showed ~5× on dense 27B and ~10× on 122B-Int4, but those
numbers aren't in a durable result file yet, so we re-measure cleanly before publishing this table.)_
<!-- endrender -->

## Takeaway
When you see a V100 inference number anywhere (including ours), the first question is **"eager or
cudagraph?"** Quote cudagraph for serving; use eager only to reason about kernels in isolation, and
say which one you mean.

*Definitions are final; the illustration table is the one cell in this series still awaiting a clean
measured pair (intentionally not back-filled from memory).*
