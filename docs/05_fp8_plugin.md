# Chapter 5 — The FP8 plugin: residency, decode speed, and prefill cost

> **Status: Final** — the canonical FP8 evidence chapter; the former Chapters 9 and 10 (precision×TP and the Qwen3.5 featured-pair profile) are folded in here. Numbers trace to the SSOT `data/benchmark_matrix.csv`; A/B sweeps cite their own result dirs. Refresh: [FINAL_RERUN.md](FINAL_RERUN.md).

V100 has no native FP8. The plugin is **W8A16**: FP8 weights stay **resident in HBM** (half the
bytes) and are dequantized to FP16 *inside the kernel* for the matmul. The first telling of this
chapter was "fit bigger models, don't expect speed." The revisit overturns half of that — and
sharpens the rest. This is the full FP8 case in one place: the kernel story (converter, coalesced),
the dense-vs-MoE split, the **Qwen3.5 exact-pair profile** (precision×TP, prefill/TTFT, faithfulness),
and the deployment takeaway.

## The limiter was our software, not Volta
The old E4M3→FP16 converter was branchy, with a subnormal `while`-loop → warp-divergent and
ALU-bound. *That*, not an sm_70 hardware floor, was the dense-FP8 decode limiter. A **branchless**
converter — value-identical to the old one (bit-exact across all 256 byte patterns, numtest-green)
— removes it, and because it is the *canonical* converter, **every FP8 kernel inherited the
speedup**. The lift is visible even with the coalesced-layout kernels *off* (both columns below are
the **matched-engine A/B on the shipping vLLM 0.21 stack**, single stream — the kernels are ours, so
these deltas are **engine-invariant** and hold on 0.19 too):

| Model (TP=8, vLLM 0.21) | Converter on<br>Coalesced **off** | Converter on<br>Coalesced **on** |
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
concurrency range on large MoE** — and these models are practical on 8×V100 only as **FP8-resident**
(the 122B would need ~244 GB in FP16; GLM-Air's FP16-resident path loads but is extremely tight).

## Dense vs MoE: a split case
- **Large MoE — settled.** On the sparse-MoE flagships FP8 beats FP16 at *every* measured concurrency
  (model pages / Chapter 1), and coalesced adds to that across C1–C8 (the table above) — engine-matched
  and engagement-proven. This is the headline V100 result.
- **Dense — split, with a known kernel wall at concurrency.** After the branchless dequant, dense FP8
  is *faster* than FP16 at low-user decode — the converter moved Qwen-27B C1 from **39 (just below
  FP16's ~40) to 52** (1.34×, now above it). But the coalesced decode kernel is a **GEMV that
  accumulates each of the M concurrent rows with scalar CUDA-core FMAs** (no tensor cores), so its cost
  scales ~linearly with M, while cuBLAS FP16 rides **tensor cores** that stay nearly flat. On a
  5120×5120 attention Linear (ms/call, the **shipping FP8 path** — the dedicated GEMV at M=1, the
  batched kernel above) ours runs **0.048 → 0.072 → 0.101 → 0.164** at M=1→8 while cuBLAS is
  **0.078 → 0.098 → 0.109 → 0.095** — FP8 wins to ~M=4, FP16 takes M=8. Dense models feel it
  because they stream the *whole* weight set every token; MoE sidesteps it via per-token expert
  sparsity. half2, split-K, **and vectorized dequant are all measured *not* to close it** (they shave
  the dequant, but the wall is per-M MAC throughput on CUDA cores). The structural fix is a
  **tensor-core / WMMA FP8 decode kernel** (the prefill path already runs WMMA) — a current result, not
  a permanent ceiling.

## The Qwen3.5 exact pair: precision × TP (the controlled comparison)

The flagship evidence above is large-MoE. The cleanest *controlled* test is two exact Qwen3.5
checkpoints — one dense (**27B**), one MoE (**35B-A3B**) — each as official **FP16**, our **FP8
W8A16**, and **GPTQ-Int4**, at full **TP4** and half **TP2**. Per-user decode tok/s (vLLM 0.21,
4096 ctx, 512 tok, temp 0):

**Dense — Qwen3.5-27B** (TP4, per-user tok/s)
| Users | FP16 | FP8 W8A16 | GPTQ-Int4 |
|---:|---:|---:|---:|
| 1 | 39.1 | **52.5** | 69.2 |
| 2 | 31.1 | **42.5** | 55.5 |
| 4 | 30.3 | 31.7 | 47.4 |
| 8 | 29.3 | 20.3 | 44.2 |

**MoE — Qwen3.5-35B-A3B** (TP4, per-user tok/s)
| Users | FP16 | FP8 W8A16 | GPTQ-Int4 |
|---:|---:|---:|---:|
| 1 | 66.2 | **92.9** | 126.2 |
| 2 | 45.4 | **77.6** | 96.1 |
| 4 | 29.5 | **72.4** | 76.2 |
| 8 | 22.9 | **54.9** | 75.1 |

Dense reproduces the crossover (FP8 **1.34×** FP16 at C1, parity at C4, **0.69×** at C8 — the
CUDA-core wall above); MoE-FP8 wins at *every* concurrency and the margin **grows with load
(1.40× → 2.45×)**. Int4 is fastest raw but lossy. (FP8 cells in **bold** where they beat FP16.)

**Capacity — the half-TP result.** At TP2 (2×32 GB) **both FP16 checkpoints fall out of the serving
envelope** while FP8 and Int4 fit (C1 tok/s):

| Model | FP16 @TP2 | FP8 @TP2 | Int4 @TP2 |
|---|---|---:|---:|
| 27B dense | **OOM** (no KV room) | 34.0 | 43.3 |
| 35B-A3B MoE | **OOM** (weights >32 GB) | 82.2 | 99.0 |

The two OOMs differ: 35B-A3B FP16 is a **hard weight-OOM** (~33 GB/GPU before any KV); 27B FP16 is a
**KV-room OOM** (weights load, the standard envelope leaves no KV headroom). Either way, **FP8 is the
faithful format that puts a modern MoE on half the GPUs** — FP16 cannot. Full per-concurrency TP2
tables: the [27B](../models/qwen3_5_27b.md) and [35B-A3B](../models/qwen3_5_35b_a3b.md) model pages.

## Prefill cost (TTFT) — the honest tradeoff

Decode and capacity favor FP8; prefill does not. Cold first-token, C1, TP4, long (~24k) input:

| Model | FP16 | FP8 | FP8 + FA-V100 |
|---|---:|---:|---:|
| 27B dense | 27.0 s | 32.2 s | 17.4 s |
| 35B-A3B MoE | 14.2 s | 55.1 s | 49.9 s |

FP8 prefill is slower than FP16 — modest on dense (~1.2×), **steep on the MoE** (55 vs 14 s) —
because the block-FP8 MoE prefill runs an unoptimized Volta WMMA dequant path. The **FA-V100 bridge**
roughly halves the *dense* prefill (attention-bound) but barely moves the **MoE-FP8** (the bottleneck
is FP8 compute, not attention). So FP8 is for decode-heavy / capacity-bound serving; long-context,
short-output traffic (RAG, summarize) prefills cheaper on FP16 or Int4.

> **Config note.** TTFT is measured **chunked-prefill on** (`enable_chunked_prefill=True`, the
> deployment standard). The decode tok/s tables above come from a steady-state sweep harness run with
> chunked-prefill *off* — which does not change steady-state decode throughput (it only affects prefill
> scheduling). Both are otherwise the standard V100 serve (`mode=0 + FULL_DECODE_ONLY` cudagraph,
> TRITON_ATTN, ns=8). Disabling chunked prefill is only a crash risk on large hybrid configs at long
> context (e.g. 122B @ 28k); these 4096-ctx runs are safe.

## Faithfulness

Temperature 0, the reliability harness:

- **Self-stability:** all four exact-pair cells are **Exact** — bit-deterministic across 5 runs,
  *including the MoE-FP8* (the 3.6-35B-A3B-FP8 was only "Stable"; this 3.5 checkpoint is tighter).
- **FP8 vs FP16:** **Stable, not Exact** — both coherent, but greedy tokens diverge (different
  numerics, not errors). That is the correct grade for any FP8-vs-FP16 pair: byte-identity is the
  wrong bar, coherent-equivalent is, and FP8 clears it.

## 3.5 vs 3.6 — the deep-dive generalizes

Qwen3.6 is the same architecture/config as 3.5, so the exact-pair findings should carry to the broad
3.6 baseline matrix. They do — FP8 long-input TTFT, matched at TP4:

| FP8 long-TTFT @ TP4 | 3.5 | 3.6 |
|---|---:|---:|
| 27B | 32.2 s | 32.2 s |
| 35B-A3B | 55.1 s | 53.3 s |

Essentially identical — supporting the "3.5 deep-dive, 3.6 wide baseline" split. (An earlier 62 s for
3.6-27B was a TP2 artifact; a 14-vs-72 s contamination on 3.6-35B resolved to a clean 53 s.)

## FP8 vs Int4 (122B, engine-matched on vLLM 0.21)
Where both formats exist they are **peers** — on the matched 0.21 stack they trade the lead through
mid-concurrency: per-user **58/57** (C1, Int4 +3%), **46/46** (C2, a tie), **42/41** (C4, FP8 nudges
ahead), and Int4 pulls clear only at **C8 (+~20%, 37 vs 31)**. A naive comparison that ran Int4 on
vLLM 0.18 and FP8 on 0.21 inflated Int4 by ~85% of the apparent gap — engine *version*, not format; on
the matched stack the lead all but disappears below C8. FP8 is the **broader fleet** path: GLM-Air and the Gemma FP8 checkpoints have
**no GPTQ-Int4 equivalent**, e4m3 is numerically richer than int4, and MTP runs on FP8 but crashes
on gptq-int4 122B. Int4 is the right pick where a GPTQ checkpoint exists *and* C8 aggregate
throughput is the sole objective.

## Takeaway

**FP8 W8A16 on V100 is a decode + capacity win with an honest prefill cost.** Reach for it when:
- **decode-heavy serving** — low-concurrency dense (FP8 > FP16 to ~C4) and MoE at *every* concurrency;
- **capacity-bound** — fitting a big MoE on fewer cards (half-TP, where FP16 OOMs), or more replicas per box.

Reach for **FP16 or Int4** instead when long-context / short-output **prefill latency** dominates, or
for the **C8-dense aggregate** (the CUDA-core wall). Output is bit-deterministic and coherent
(Stable-vs-FP16, not byte-identical). The featured worked examples are the
[Qwen3.5-27B](../models/qwen3_5_27b.md) and [Qwen3.5-35B-A3B](../models/qwen3_5_35b_a3b.md) pages.

## Caveats
- "Stable, not Exact" vs FP16 by construction (W8A16 changes numerics; output stays coherent).
- The plugin is **custom/local**, not upstream vLLM.
- Methodology: each A/B holds model/TP/**engine** fixed and toggles one flag; coalesced engagement
  is proven from server logs (zero coalesced-w13 hits with the flag off; w13 + attention-coalescing
  banners with it on), so the deltas are the kernel, not config drift.

*Evidence: `results/coal_ab_q122b_*`, `results/coal_ab_glm_*` (flagship A/B), the SSOT's matched 122B
FP8-vs-Int4 rows (`data/benchmark_matrix.csv`), `results/fp8_vecdq_microbench_20260620/` (the dense
GEMV-vs-cuBLAS M-scaling + half2/vecdq dead ends), `results/ch1_20260611/` (single-user matrix).
**Qwen3.5 exact pair:** decode `results/q27b_exact_triad_*` + `results/q35b_exact_triad_*` (TP4 and
`*_tp2_*` half-GPU); TTFT `results/perf_v2_q27b35_*` + `results/perf_v2_q35b35_*` (and the matched 3.6
TP4 fill `results/perf_v2_q27b_fp8_021_20260625_*` / `…q35b_fp8…`); faithfulness from `tools/ch1_report.py`
on `/tmp/v100_ch1/manifest.csv` (Axis-1 self-stability + Axis-2 FP8-vs-FP16 agreement). Kernel
details + the dense-vs-MoE profile in the code repo's `docs/COALESCED_FP8_GEMV.md` and `docs/SESSION_LOG.md`.*
