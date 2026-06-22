# Final freeze & future refresh

The numbers are SSOT-driven: prose carries claims/ranges, tables carry exact figures auto-rendered
from `data/benchmark_matrix.csv`. That separation is what lets a freeze — or a later refresh — be a
couple of commands plus a read, not a rewrite.

## What froze, and on what
The main matrix is **frozen from the dual-engine `perf_v2` campaign** (2026-06-20/21): the Chapter 1
matrix and every per-model feasible-TP × concurrency row come from one consistent run set on **both**
engines — vLLM **0.21** (`vllm-v100:vllm021-cu126`) and **0.19** (`vllm-v100-py312:vllm019-cu126`, plus
the `vllm019-tf5`/cu128 variant for Gemma-4 / GLM-4.7-Flash) — on a clean idle box, same flags. This
replaced the earlier single-engine plan (one `vllm021-cu126` campaign) and the date-mixed bring-up
numbers.

The **A/B chapters keep their own dedicated runs by design** — *not* folded into perf_v2. The MoE fix
(Ch2), eager-vs-cudagraph (Ch3), and MTP (Ch4) are each a same-harness toggle where only one variable
changes, so both arms must come from one paired run; mixing them with the perf_v2 serving matrix would
break the comparison. Each cites its own `result_path`.

## Still open before going public
- **Flip the DRAFT banners** on the chapters once the prose is settled (the *numbers* are already final).
- **Publish** on the agreed cadence.

## Future refresh — the procedure (if you re-freeze)
1. **Run the bench** on a clean idle box, both engines: the perf_v2 matrix harness (per-model TP ×
   concurrency), plus the Ch2 MoE A/B, Ch3 eager/cudagraph, and Ch4 MTP harnesses if those change.
   (Harnesses live in the companion `fp8-w8a16-sm70` repo.)
2. **Rebuild the SSOT:** `python3 scripts/build_matrix_from_results.py` → regenerates
   `data/benchmark_matrix.csv` and `data/eager_vs_cudagraph.csv` from the fresh results.
3. **Re-render every table:** `python3 scripts/render_tables.py --inject` → all chapter/model tables
   update from the CSV in one shot.
4. **Reconcile the prose:** skim each chapter/model page for figures or framing the new numbers
   changed. The tables auto-sync; the **prose does not** — when the matrix last moved to perf_v2 the
   tables re-rendered cleanly but the prose needed a hand pass. Budget for that step; don't skip it.

## The discipline that keeps the refresh cheap
- **Prose** = claims + ranges robust to ±small drift ("4–9×", "beats dense", "FP8 wins on MoE").
- **Tables** = exact figures, auto-rendered from the CSV — never hand-typed in prose.
