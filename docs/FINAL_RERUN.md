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

**TTFT** is frozen from the same campaign's `ttftboth` phase (the 2026-06-21 sweep, single
unique-prompt cold sends): three single-stream fields per C1 row — `ttft_s` (cold first-token, a full
~22.6k-token cache-cold prefill = worst case), `ttft_prefix_hit_s` (prefix-cache hit = best case), and
`ttft_fa_cold_s` (FlashAttention cold, FA-eligible 0.21 cells). All TTFT is chunked-prefill **on** (the
project-standard serve; disabling it is a V100 crash-causer, so there is no "monolithic chunked-OFF"
number). The earlier scattered cold-mono numbers — which under-reported MoE cold prefill ~5× — are
**not** used.

The **A/B chapters keep their own dedicated runs by design** — *not* folded into perf_v2. The MoE fix
(Ch2), eager-vs-cudagraph (Ch3), and MTP (Ch4) are each a same-harness toggle where only one variable
changes, so both arms must come from one paired run; mixing them with the perf_v2 serving matrix would
break the comparison. Each cites its own `result_path`.

## Going public
- **DRAFT banners flipped to "Status: Final"** (done 2026-06-23) — numbers frozen at tag
  `fp8-v100-2026-matrix`; the chapters no longer carry provisional notices.
- **Push and publish:** `git push origin main`, then publish on the agreed cadence.

## Future refresh — the procedure (if you re-freeze)
1. **Run the bench** on a clean idle box, both engines: the perf_v2 matrix harness (per-model TP ×
   concurrency **and** its `ttftboth` TTFT phase), plus the Ch2 MoE A/B, Ch3 eager/cudagraph, and Ch4
   MTP harnesses if those change. (Harnesses live in the companion `fp8-w8a16-sm70` repo.)
2. **Consolidate, then rebuild the SSOT:** in the `fp8-w8a16-sm70` repo run
   `python3 tools/perf_v2_consolidate.py` → reconciles the per-cell results into
   `results/perf_v2_COMBINED.csv` (decode + TTFT source columns, with per-metric provenance); then
   here run `python3 scripts/build_matrix_from_results.py` → regenerates `data/benchmark_matrix.csv`
   (incl. `ttft_s` / `ttft_prefix_hit_s` / `ttft_fa_cold_s`) and `data/eager_vs_cudagraph.csv`.
3. **Re-render every table:** `python3 scripts/render_tables.py --inject` → all chapter/model tables
   update from the CSV in one shot.
4. **Reconcile the prose:** skim each chapter/model page for figures or framing the new numbers
   changed. The tables auto-sync; the **prose does not** — when the matrix last moved to perf_v2 the
   tables re-rendered cleanly but the prose needed a hand pass. Budget for that step; don't skip it.

## The discipline that keeps the refresh cheap
- **Prose** = claims + ranges robust to ±small drift ("4–9×", "beats dense", "FP8 wins on MoE").
- **Tables** = exact figures, auto-rendered from the CSV — never hand-typed in prose.
