# Chapter 5 — The FP8 plugin: residency, and now decode speed

> **Status: DRAFT** — numbers provisional until the final freeze ([FINAL_RERUN.md](FINAL_RERUN.md)). Single-user matrix rows auto-render from `data/benchmark_matrix.csv`; the A/B sweep tables below cite their own result dirs.

V100 has no native FP8. The plugin is **W8A16**: FP8 weights stay **resident in HBM** (half the
bytes) and are dequantized to FP16 *inside the kernel* for the matmul. The first telling of this
chapter was "fit bigger models, don't expect speed." The revisit overturns half of that — and
sharpens the rest.

## The limiter was our software, not Volta
The old E4M3→FP16 converter was branchy, with a subnormal `while`-loop → warp-divergent and
ALU-bound. *That*, not an sm_70 hardware floor, was the dense-FP8 decode limiter. A **branchless**
converter — value-identical to the old one (bit-exact across all 256 byte patterns, numtest-green)
— removes it, and because it is the *canonical* converter, **every FP8 kernel inherited the
speedup**. The lift is visible even with the coalesced-layout kernels *off* (both columns below are
the **matched-engine A/B on the shipping vLLM 0.21 stack**, single stream):

| Model (TP=8, vLLM 0.21) | Converter on, coalesced **off** | + Coalesced **on** |
|---|---:|---:|
| Qwen3.5-122B-A10B-FP8 | 51.8 | 57.0 |
| GLM-4.5-Air-FP8 | 57.8 | 65.2 |

Against the **first-published** baselines (122B 34.6, GLM-Air 30.8), the model-level progress is
**1.65×** and **2.12×** — but those baselines were measured on the older engine, so that span bundles
the 0.18→0.21 engine move in with the kernel work; read it as *product progress*, not a single-kernel
attribution. The clean kernel attribution, all on 0.21: the branchless converter alone is most of the
lift (e.g. 122B's first 0.21 coalesced reading was 44.3 with the slow converter → 57.0 with the fast
one, ~1.29×), and coalesced layout adds the rest (51.8→57.0, ~1.10×).

## Coalesced kernels add a second, shape-dependent win
On top of the converter, the coalesced decode kernels (a warp-per-output GEMV for attention/dense
Linears, grouped variants for MoE experts) add more — and how much depends on model shape and load.
On the **FP8 MoE flagships** the gain *grows* with concurrency:

| C | 122B off→on | GLM-Air off→on |
|---|---|---|
| 1 | 51.8→57.0 (1.10×) | 57.8→65.2 (1.13×) |
| 4 | 35.8→40.9 (1.14×) | 26.1→35.3 (1.35×) |
| 8 | 23.1→31.1 (1.35×) | 17.6→23.4 (1.33×) |

The grouped-expert path benefits directly as batches fill, so **coalesced wins across the measured
concurrency range on large MoE** — and these models fit on 8×V100 *only* resident in FP8 (the 122B
would need ~244 GB in FP16).

## Dense vs MoE: a split case
- **Large MoE — settled.** FP8 wins across the measured C1–C8 range (above), engine-matched and
  engagement-proven. This is the headline V100 result.
- **Dense — split, and partly open.** FP8 is now *faster* than FP16 at low-user decode after
  branchless dequant (Qwen-27B C1 39→52, 1.34×), but the **current** dense C8 kernel still trails
  FP16. We have closed half2 and split-K as measured dead ends; **vectorized FP8 dequant
  (`prmt`/byte-perm) remains the next NCU-gated hypothesis**. So today's dense rule: FP8 for
  low-user decode and residency; FP16 still wins dense high-concurrency, pending that kernel
  question. (Stated as a *current result*, not a structural wall.)

## FP8 vs Int4 (122B, engine-matched on vLLM 0.21)
Where both formats exist they are **peers**: per-user 58/57 (C1), 47/46 (C2), 41/41 (C4), Int4
+~15% (C8). A naive comparison that ran Int4 on vLLM 0.18 and FP8 on 0.21 inflated Int4 by ~85% of
the apparent gap — engine *version*, not format; on the matched 0.21 stack the lead all but
disappears below C8. FP8 is the **broader fleet** path: GLM-Air and the Gemma FP8 checkpoints have
**no GPTQ-Int4 equivalent**, e4m3 is numerically richer than int4, and MTP runs on FP8 but crashes
on gptq-int4 122B. Int4 is the right pick where a GPTQ checkpoint exists *and* C8 aggregate
throughput is the sole objective.

## Caveats
- "Stable, not Exact" vs FP16 by construction (W8A16 changes numerics; output stays coherent).
- The plugin is **custom/local**, not upstream vLLM.
- Methodology: each A/B holds model/TP/**engine** fixed and toggles one flag; coalesced engagement
  is proven from server logs (zero coalesced-w13 hits with the flag off; w13 + attention-coalescing
  banners with it on), so the deltas are the kernel, not config drift.

*Evidence: `results/coal_ab_q122b_*`, `results/coal_ab_glm_*` (flagship A/B), `results/q122b_int4_v021_*`
(engine-matched Int4), `results/ch1_20260611/` (single-user matrix). Kernel details + the dense-vs-MoE
profile in the code repo's `docs/COALESCED_FP8_GEMV.md` and `docs/SESSION_LOG.md`.*
