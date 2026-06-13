# Chapter 2 — The FP16 MoE bug: vLLM leaves 4–9× on the floor on V100

> **Status: DRAFT** — numbers provisional until the final freeze rerun ([docs/FINAL_RERUN.md](FINAL_RERUN.md)). Tables auto-render from `data/benchmark_matrix.csv`.

If you run a Mixture-of-Experts model in FP16 on a V100 with stock vLLM, it is **slower than a
dense model of similar size** — which is backwards. A sparse MoE only activates a few experts per
token, so it should move *less* weight than a dense model and run *faster*. On V100 it runs slower.
That's not the hardware giving up. It's a single config default that's wrong for Volta, and a
one-file change recovers **4–9×**.

## The symptom

Same engine, same box (8×V100-32GB, vLLM 0.21 built for CUDA 12.6, cudagraph), just MoE vs dense:

| model | stock FP16 | a *dense* model of similar class |
|---|---|---|
| Qwen3.6-35B-A3B (only 3B active) | ~15.5 tok/s | Qwen3.6-27B dense ≈ 37–41 tok/s |
| gemma-4-26B-A4B (4B active) | ~10.9 tok/s | gemma-4-31B dense ≈ 17.6 tok/s |

A 35B model with 3B active parameters should be *trouncing* a 27B dense model, not running at
0.4× its speed. Something is wrong with how the MoE kernel runs, not with the math.

## The detective story (and the wrong turn worth admitting)

vLLM's fused-MoE Triton kernel picks its launch tiling in `get_default_config`. There is no V100
(sm_70) case — every CUDA device gets the same defaults, tuned for Ampere and newer.

The *obvious* suspect was `num_stages`. Ampere defaults to `num_stages=4`: a deep software pipeline
that prefetches the next weight tile while computing the current one — cheap on Ampere because of
the `cp.async` instruction (async global→shared copy). V100 has no `cp.async`, so the theory was:
deep pipelining without hardware support craters the kernel. Plausible, with a tidy mechanism.

**We measured it. It was wrong.** Sweeping `num_stages` 4→3→2 end-to-end was *flat* — 15.57 tok/s
every time, byte-identical output. `num_stages` was not the lever.

The real culprit was one tile dimension over: **`BLOCK_SIZE_K`**. The decode branch (small batch,
M≤64) picks `BLOCK_SIZE_K=128`. On V100 that register-spills Triton's codegen — the spill traffic
contends on memory bandwidth and the cost grows roughly linearly with batch. Drop it to 64 and the
kernel is ~2.3× faster at batch 1 and up to ~9× at batch 16. (The prefill path already uses 64;
only decode picked the bad value.) Kernel time vs `BLOCK_K` at batch 1: **64 → 632 µs, 128 → 1450
µs, 256 → 2300 µs** — almost monotonic in that one knob.

The lesson we keep relearning: a mechanism that *sounds* right (no `cp.async` → hates deep pipelines)
is not evidence. The flat `num_stages` sweep saved us from "fixing" the wrong thing.

## The fix and the result

<!-- render:moe_fix -->
| model | config | users | per-user tok/s | aggregate tok/s |
|---|---|---|---|---|
| gemma-4-26B-A4B-it | stock(pre-moe-patch) | 1u | 10.2 | - |
| Qwen3.6-35B-A3B | stock(pre-moe-patch) | 1u | 15.44 | - |
| Qwen3.6-35B-A3B | stock(pre-moe-patch) | 1u | 15.56 | - |
| Qwen3.6-35B-A3B | stock(pre-moe-patch) | 8u | 3.16 | 24.93 |
| Qwen3.6-35B-A3B | +moe_patch(heuristic) | 1u | 65.91 | - |
| Qwen3.6-35B-A3B | +moe_patch(heuristic) | 8u | 20.98 | 137.2 |
| Qwen3.6-35B-A3B | +moe_patch(tuned-json) | 1u | 65.85 | - |
| Qwen3.6-35B-A3B | +moe_patch(tuned-json) | 8u | 22.8 | 173.92 |
| gemma-4-26B-A4B-it | stock(pre-moe-patch) | 1u | 10.91 | - |
| gemma-4-26B-A4B-it | stock(pre-moe-patch) | 8u | 3.58 | 28.3 |
| gemma-4-26B-A4B-it | +moe_patch(heuristic) | 1u | 43.66 | - |
| gemma-4-26B-A4B-it | +moe_patch(heuristic) | 8u | 19.1 | 145.15 |
| gemma-4-26B-A4B-it | +moe_patch(tuned-json) | 1u | 43.71 | - |
| gemma-4-26B-A4B-it | +moe_patch(tuned-json) | 8u | 20.23 | 155.94 |
<!-- endrender -->

Single-stream, the inversion is gone: 35B goes from ~15.6 to **~66 tok/s (≈4×)** and gemma-26B from
~10.9 to **~44 (≈4×)** — and the output is **bit-identical** to stock (pure speed, not an
approximation). The win *grows with concurrency* because the stock kernel degrades with batch: at
8 concurrent users the 35B aggregate goes from ~25 to **~174 tok/s**. Sparse-beats-dense is restored.

Two forms ship: a **default-on heuristic** (`BLOCK_K=64` for small-M on sm<80 — works for any MoE
model and any TP with no per-model tuning) and, for the two models we tuned exhaustively, **per-M
autotuned config files** that add another ~5–10% at concurrency over the heuristic.

## Why this is a *default* bug, not a "V100 is old" story

vLLM ships ~317 tuned config files for A100/H100/MI300X and **zero for any V100** — so V100 always
falls back to `get_default_config`, and that fallback was written for hardware with `cp.async`.
There's even precedent in vLLM's own tree: a neighboring MoE kernel (`moe_fused_mul_sum`) *does*
special-case pre-Ampere with `num_stages=2`. Someone fixed one kernel's Volta path and left the main
GEMM's default Volta-blind. This is a gap, not a hardware ceiling.

## Upstreaming

Reported to **both** engines, framed for what each will take:
- **vLLM** (our main engine — we run 0.21 on V100): filed as a finding, not an sm_70-support PR.
  The general-interest part: the decode-branch `BLOCK_K=128` default may be worth re-checking for
  small-M even on `cp.async` hardware — we only have V100 evidence, so we pose it as a question,
  plus the V100 config files as a data contribution.
- **aphrodite-engine** (broad-arch support; where we learned the sm_70 build approach): the full
  fix including the sm<80 heuristic, as a PR.

## Caveats (so nobody over-reads this)

- **FP16/BF16 MoE only.** It does not touch dense models or our FP8 path (FP8 uses its own kernels —
  Chapter 5).
- "4–9×" is **over the stock untuned default** — recovering a regression, not out-running good
  hardware. The fixed FP16 number (~66) is the honest comparator used elsewhere in this write-up.
- Measured on V100-32GB, vLLM 0.21+cu126, cudagraph. Build details in Chapter 1.

*Evidence: `results/moe_stages_ab_*` (A/B runs), `results/moe_decode_tile_sweep_*` (the BLOCK_K
sweep), `results/moe_decode_msweep_*` (batch-scaling). Full root-cause trace lives in the code
repo's `docs/V100_OPTIMIZATION_FINDINGS.md`.*
