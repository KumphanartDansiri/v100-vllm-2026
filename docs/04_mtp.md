# Chapter 4 — MTP (speculative decode): kept separate so it can't contaminate the baselines

> **Status: DRAFT** — numbers provisional until the final freeze ([FINAL_RERUN.md](FINAL_RERUN.md)). Tables auto-render from `data/benchmark_matrix.csv`.

Multi-Token Prediction runs a small draft head to propose *k* tokens per step, verified in one pass.
It can speed up decode — but it changes what "tok/s" *means* (output per step now depends on an
acceptance rate), so we measure it in its own chapter and never fold MTP numbers into the base-decode
comparisons of Chapters 1–3 or the model pages.

## The headline finding: MTP's payoff is inversely proportional to your baseline speed

On V100, MTP helps *least* exactly where decode is already fast, because the draft+verify overhead
eats the gain. The clean proof is the **same model at two baselines**: Qwen3.6-35B-A3B at stock FP16
(slow, 15.86 tok/s) → MTP **helps** (1.08×); the same model with our fast FP8 path (70 tok/s) → MTP
**hurts** (0.95×). Once the baseline is fast, k=1 MTP is net-negative.

## k=1 results (cudagraph, single-stream)

<!-- render:mtp -->
<!-- endrender -->

Read it with the acceptance/exactness columns, not tok/s alone:
- **Acceptance% is a perf signal, not a usability one.** High acceptance with garbage output is a
  known trap — pair it with exactness/coherence. (Here: 27B-FP8 is token-for-token **EXACT** with
  acceptance 88% but *slower* at 0.73× — acceptance was high, speed still lost.)
- **EXACT vs DIFF:** dense 27B is bit-exact with MTP; the MoE models show DIFF (benign FP
  nondeterminism in the MoE reduction under the spec-batch shape — happens on stock FP16 too, so it
  exonerates our plugin), output stays coherent.
- **gemma:** MTP is not supported (`NotImplementedError: Unsupported speculative method 'mtp'`) — an
  honest n/a, not a number.

## Where MTP actually wins: the comm-bound flagship at k≥2

The one place MTP pays on V100 is the **122B-A10B at TP8**, where per-step fixed overhead (8-way
all-reduce) is large enough that amortizing it over multiple tokens wins. At k=1 it's already +1.05×;
with the route-slot cap fix that unblocks k=2 capture (`MAX_ROUTE_SLOTS=512`), 122B k=2 reaches
**~1.45×** (≈66 tok/s, coherent, ~78% accept). That's the flagship MTP story — comm-bound model,
multi-token amortization. (Evidence: `results/ch2_mtp2_slots512_122b`, `results/ch2_mtp_k23_slots512_122b`.)

## Takeaway for readers
Turn MTP **on** only when your baseline decode is slow *and* the model is communication-bound at high
TP (the 122B case). On an already-fast FP8 single-GPU-class decode, leave it off — it costs more than
it returns. And never quote an MTP tok/s next to a base-decode tok/s without saying so.

*Evidence: `results/ch2_mtp_20260612/CHAIN_SUMMARY.txt` (k=1 matrix), `results/ch2_mtp*slots512*`
(the k=2 flagship). Acceptance ≠ proof — see methodology.*
