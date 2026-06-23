# Chapter 4 — MTP (speculative decode): kept separate so it can't contaminate the baselines

> **Status: Final** — numbers frozen at the SSOT; tables auto-render from `data/benchmark_matrix.csv`. Refresh procedure: [FINAL_RERUN.md](FINAL_RERUN.md).

Multi-Token Prediction runs a small draft head to propose *k* tokens per step, verified in one pass.
It can speed up decode — but it changes what "tok/s" *means* (output per step now depends on an
acceptance rate), so we measure it in its own chapter and never fold MTP numbers into the base-decode
comparisons of Chapters 1–3 or the model pages.

**Correctness and exactness are policy choices, not throughput facts.** Every base serving number in
this write-up is reported *without* MTP, and we do **not** headline MTP-boosted tok/s — because MTP can
change output exactness. So this chapter is an **operator guide, not a leaderboard**: if your
application accepts the resulting correctness/exactness trade-off, the sweep below shows where MTP is
likely to help and which *k* are worth trying. (It's also why the 122B's 1.67× lives here, not as a
Chapter 1 serving number.)

## The headline finding: MTP pays only when there's fixed cost to amortize

MTP is **not a universal speed button.** It pays when the decode steps it avoids are expensive enough
to amortize the draft+verify work — so the win tracks **per-step fixed cost**, and the **optimal *k*
depends on the model and precision.** Two reads off the sweep:
- **Inverse to baseline speed (same model, two baselines):** Qwen3.6-35B-A3B at stock FP16 (slow,
  15.86 tok/s) → MTP **helps** at k=1 (1.08×); the *same model* on our fast FP8 path (70 tok/s) → MTP
  **hurts** (0.95×). Once decode is already fast, k=1 spec-decode is net-negative.
- **The best k is model-dependent:** on the TP8 **122B MoE** (comm-bound — an 8-way all-reduce every
  step) the win **climbs to 1.67× at k=4**; on the already-fast **35B-A3B FP8** it **peaks at k=2
  (1.24×) then declines**; on **dense 27B** it does not beat base decode in the measured k=1–2 sweep.

## The k-sweep (cudagraph, single-stream)

<!-- render:mtp -->
| Model | k | Base tok/s | MTP tok/s | Speedup | Accept | Exactness |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.6-27B (FP8, dense) | 1 | 36.28 | 26.48 | 0.73x | 88.4% | Exact |
|  | 2 | 36.25 | 35.51 | 0.98x | 77.1% | Exact |
| Qwen3.6-35B-A3B (FP8, MoE) | 1 | 70.06 | 66.38 | 0.95x | 83.9% | Diff |
|  | 2 | 70.08 | 86.77 | 1.24x | 71.3% | Diff |
|  | 3 | 70.11 | 79.98 | 1.14x | 58.1% | Diff |
|  | 4 | 70.11 | 77.48 | 1.11x | 47.0% | Diff |
| Qwen3.6-35B-A3B (FP16, MoE) | 1 | 15.86 | 17.2 | 1.08x | 84.9% | Diff |
| Qwen3.5-122B-A10B (FP8, MoE) | 1 | 45.86 | 48.2 | 1.05x | 87.4% | Diff |
|  | 2 | 45.70 | 66.09 | 1.45x | 77.8% | Diff |
|  | 3 | 45.82 | 74.53 | 1.63x | 67.2% | Diff |
|  | 4 | 45.85 | 76.78 | 1.67x | 56.9% | Diff |
<!-- endrender -->

This is the **measured** sweep, not an exhaustive grid: k≥2 was run where it mattered (the FP8 MoE and
dense models); the FP16 comparison is k=1 only, and gemma is reported as unsupported/n/a rather than
assigned a number. Read it with the acceptance/exactness columns, not tok/s alone:
- **Acceptance% is a perf signal, not a usability one.** High acceptance with garbage output is a
  known trap — pair it with exactness/coherence. (Here: 27B-FP8 is token-for-token **Exact** with
  acceptance 88% but *slower* at 0.73× — acceptance was high, speed still lost.)
- **Exact vs Diff:** dense 27B is bit-exact with MTP (Exact at k=1 and k=2); the MoE models show Diff
  (benign FP nondeterminism in the MoE reduction under the spec-batch shape — happens on stock FP16
  too, so it exonerates our plugin), output stays coherent. **Diff means *different from base decode*,
  not *wrong*** — whether that's acceptable is the application's call, not a universal verdict.
- **gemma:** MTP is not supported (`NotImplementedError: Unsupported speculative method 'mtp'`) — an
  honest n/a, not a number.

## Why the 122B keeps winning as k rises

The 122B-A10B at TP8 is the standout: its per-step fixed cost (the 8-way all-reduce) is large enough
that amortizing it over more speculated tokens keeps paying — the sweep climbs **1.05× → 1.45× → 1.63×
→ 1.67×** from k=1 to k=4, *even as acceptance falls* (87% → 57%), because each accepted token still
skips a full all-reduce-bound decode step. Capturing k≥2 under cudagraph on Volta took one fix — raising
the grouped-MoE route-slot cap (`MAX_ROUTE_SLOTS=512`); without it the k≥2 capture aborts. By contrast
the fast 35B-A3B FP8 tops out at **k=2** and dense 27B does not break even in the measured k=1–2 sweep
— same "amortize the fixed cost" rule, different amounts of fixed cost to amortize.

## Takeaway for readers
**MTP is an optional operating mode, not the default benchmark mode.** Exactness-sensitive deployments
should stay on base decode; throughput-sensitive ones can use this sweep as a starting point. Concretely:
turn MTP **on** when the model is communication-bound at high TP (the 122B case) — and then **tune k**
(122B keeps gaining through k=4). On an already-fast FP8 single-GPU-class decode, leave it off or cap at
k=2; past the peak it costs more than it returns. And never quote an MTP tok/s next to a base-decode
tok/s without saying so.

*Evidence: `results/ch2_mtp_20260612/CHAIN_SUMMARY.txt` (k=1) and `results/ch2_mtp*slots512*` (the
k=2–4 sweep). Acceptance ≠ proof — see methodology.*
