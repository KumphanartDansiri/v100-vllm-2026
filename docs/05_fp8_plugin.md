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
**1.65×** and **2.12×** — but those baselines were measured on an older engine, so that span bundles
the engine move in with the kernel work; read it as *product progress*, not a single-kernel
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

## The Qwen3.5 pair: precision comparison at the fleet condition

The flagship evidence above is large-MoE. The cleanest *controlled* test is two exact Qwen3.5
checkpoints — one dense (**27B**), one MoE (**35B-A3B**) — each as official **FP16**, our **FP8
W8A16**, and **GPTQ-Int4**, measured at the **fleet condition** (32768 ctx, 256-tok median-of-5,
chunked-prefill on, TP4) — the same setup as every flagship. Per-user decode tok/s on vLLM 0.21 (the
dual-engine 0.19+0.21 view is the triad table below):

**Dense — Qwen3.5-27B** (fleet, per-user tok/s)
| Users | FP16 | FP8 W8A16 | GPTQ-Int4 |
|---:|---:|---:|---:|
| 1 | 35.4 | **46.0** | 50.4 |
| 2 | 28.5 | **37.1** | 41.3 |
| 4 | 27.7 | **28.4** | 38.8 |
| 8 | 27.4 | 19.4 | 34.9 |

**MoE — Qwen3.5-35B-A3B** (fleet, per-user tok/s)
| Users | FP16 | FP8 W8A16 | GPTQ-Int4 |
|---:|---:|---:|---:|
| 1 | 56.2 | **74.9** | 78.4 |
| 2 | 40.0 | **63.4** | 64.2 |
| 4 | 26.9 | **58.8** | 60.1 |
| 8 | 21.4 | **45.8** | 54.5 |

Dense reproduces the crossover (FP8 **1.30×** FP16 at C1, ~parity at C4, **0.71×** at C8 — the
CUDA-core wall above); MoE-FP8 wins at *every* concurrency and the margin **grows with load
(1.33× → 2.14×)**. (FP8 cells in **bold** where they beat FP16.)

> **GPTQ-Int4 correctness — read before using the Int4 columns.** Int4 is fastest raw, but on V100 its
> *output quality splits by model*: **MoE** Int4 (35B here, and the 122B) is coherent, but the **dense
> 27B** Int4 emits **degenerate repetition** — the correctness battery hard-**fails** it on both engines.
> That is a known, **upstream** GPTQ-on-Volta issue (the hybrid model's layers quantize badly), **not**
> the serving stack. So the **27B-Int4 numbers are a speed-only datapoint — do not deploy it**; its
> tok/s are reported only for completeness.

**Capacity — the half-TP result (a 4096-ctx capacity sub-study; the fleet runs TP4).** At TP2 (2×32 GB)
**both FP16 checkpoints fall out of the serving
envelope** while FP8 and Int4 fit (C1 tok/s):

| Model | FP16 @TP2 | FP8 @TP2 | Int4 @TP2 |
|---|---|---:|---:|
| 27B dense | **OOM** (no KV room) | 34.0 | 38.9 |
| 35B-A3B MoE | **OOM** (weights >32 GB) | 82.2 | 82.4 |

The two OOMs differ: 35B-A3B FP16 is a **hard weight-OOM** (~33 GB/GPU before any KV); 27B FP16 is a
**KV-room OOM** (weights load, the standard envelope leaves no KV headroom). Either way, **FP8 is the
faithful format that puts a modern MoE on half the GPUs** — FP16 cannot. The full **fleet TP4** triad
(FP16/FP8/Int4 × both engines, C1–C8) is on the [27B](../models/qwen3_5_27b.md) and
[35B-A3B](../models/qwen3_5_35b_a3b.md) model pages; the TP2 half-GPU result above is the 4096-ctx
capacity sub-study (full per-concurrency C1–C8 in the engineering archive).

## Triad performance comparison — both engines (the dual-engine promotion)

Both featured checkpoints are now **fully tested across both freeze-high engines**: every precision —
FP16, FP8 W8A16, GPTQ-Int4 — measured on **vLLM 0.19+cu126 *and* 0.21+cu126** at the same fleet serve
(TP4 / 32768-ctx / 256-tok median-5 / chunked-on / cudagraph), so the engine is the only variable. These
tables render straight from the SSOT (per-user decode tok/s, C1→C8; **27B-Int4 is the degenerate-output
GPTQ case noted above — speed-only**):

**Dense — Qwen3.5-27B**
<!-- render:triad:qwen3_5_27b -->
| Precision | Engine | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| FP16* | 0.19 | 40.04 | 31.52 | 30.97 | 30.26 |
|  | 0.21 | 35.43 | 28.46 | 27.72 | 27.40 |
| FP8 | 0.19 | 54.05 | 42.83 | 30.83 | 20.43 |
|  | 0.21 | 46.05 | 37.06 | 28.41 | 19.44 |
| GPTQ-Int4 | 0.19 | 60.47 | 48.12 | 44.92 | 40.19 |
|  | 0.21 | 50.36 | 41.25 | 38.82 | 34.90 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

**MoE — Qwen3.5-35B-A3B**
<!-- render:triad:qwen3_5_35b_a3b -->
| Precision | Engine | C1 | C2 | C4 | C8 |
|---|---|---:|---:|---:|---:|
| FP16* | 0.19 | 63.51 | 44.23 | 35.28 | 28.02 |
|  | 0.21 | 56.19 | 39.98 | 26.94 | 21.39 |
| FP8 | 0.19 | 90.45 | 74.55 | 66.07 | 51.81 |
|  | 0.21 | 74.86 | 63.38 | 58.77 | 45.80 |
| GPTQ-Int4 | 0.19 | 95.35 | 75.58 | 68.31 | 62.36 |
|  | 0.21 | 78.38 | 64.23 | 60.11 | 54.47 |

_\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; `--dtype float16`) — the decode/latency numbers are FP16 runtime._
<!-- endrender -->

Two things hold across the whole triad. **(1) The precision ordering is engine-invariant** — Int4 (raw
speed) > FP8 > FP16, FP8's dense crossover, MoE-FP8 winning at every concurrency, the TP2 capacity
result: none of it depends on the engine, so the *precision* guidance is the same whichever engine you
run (the 27B-Int4 *correctness* failure is also engine-invariant — garbage on both). **(2) 0.19 is
uniformly faster than 0.21** — ~13–22% across every cell, the documented engine regression — so 0.19 is
the throughput pick and 0.21 the keep-it-patchable-foundation pick (see
[Chapter 1](01_v100_in_2026.md)).

## Prefill cost (TTFT) — the honest tradeoff

Decode and capacity favor FP8; prefill does not. Cold first-token, C1, TP4, long (~24k) input:

| Model | FP16 | FP8 | FP8 + FA-V100 |
|---|---:|---:|---:|
| 27B dense | 27.0 s | 32.2 s | 16.7 s |
| 35B-A3B MoE | 14.2 s | 56.7 s | 50.0 s |

FP8 prefill is slower than FP16 — modest on dense (~1.2×), **steep on the MoE** (55 vs 14 s) —
because the block-FP8 MoE prefill runs an unoptimized Volta WMMA dequant path. The **FA-V100 bridge**
roughly halves the *dense* prefill (attention-bound) but barely moves the **MoE-FP8** (the bottleneck
is FP8 compute, not attention). So FP8 is for decode-heavy / capacity-bound serving; long-context,
short-output traffic (RAG, summarize) prefills cheaper on FP16 or Int4.

> **Config note.** Decode tok/s, TTFT, and the correctness battery all come from the **same fleet
> serve**: 32768 ctx, **chunked-prefill on** (`enable_chunked_prefill=True`, the deployment standard),
> `mode=0 + FULL_DECODE_ONLY` cudagraph, TRITON_ATTN, ns=8 — decode is 256-tok median-of-5; cold TTFT is
> a fresh ~22.6k-token prefill. Chunked prefill is the standard serve; disabling it is a known V100
> crash-causer on large hybrid configs at long context (e.g. 122B @ 28k).

## Faithfulness

From the fleet correctness battery (temperature 0):
- **FP16 and FP8 are coherent** on both checkpoints — the battery passes 4/5, the lone "suspect" being a
  strict JSON-format gate, not an error. FP16 decode is bit-deterministic run-to-run; **FP8 is Stable** —
  coherent and token-divergent vs FP16 (different numerics, not errors). Byte-identity is the wrong bar
  for an FP8-vs-FP16 pair; coherent-equivalent is, and FP8 clears it.
- **GPTQ-Int4 splits by model:** coherent on the 35B MoE (and the 122B), but **degenerate on the dense
  27B** (pure repetition, a hard battery fail on both engines) — a known upstream GPTQ-on-Volta defect,
  not the serving stack. Treat 27B-Int4 as speed-only.

## 3.5 ≈ 3.6 — same architecture, same condition

Qwen3.5 and 3.6 share architecture/config, and now that both are measured at the identical fleet
condition their numbers track closely (e.g. FP8 cold TTFT @ TP4: 27B ≈ 32 s on both; 35B-A3B ≈ 56 vs
53 s). The 3.5 pair is the featured worked example; the 3.6 pair is the broad baseline — the findings
are interchangeable.

## FP8 vs Int4 (122B, engine-matched on vLLM 0.21)
Where both formats exist they are **peers** — on the matched 0.21 stack they trade the lead through
mid-concurrency: per-user **58/57** (C1, Int4 +3%), **46/46** (C2, a tie), **42/41** (C4, FP8 nudges
ahead), and Int4 pulls clear only at **C8 (+~20%, 37 vs 31)** — the lead all but disappears below C8
once both formats are measured on the same engine. FP8 is the **broader fleet** path: GLM-Air and the Gemma FP8 checkpoints have
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
**Qwen3.5 pair:** fleet TP4 decode + TTFT + correctness battery `results/perf_v2_q27b35_*` +
`results/perf_v2_q35b35_*` (the canonical fleet rows); the **TP2 half-GPU capacity sub-study** from
`results/q27b_exact_triad_*_tp2_*` + `results/q35b_exact_triad_*_tp2_*` (4096-ctx); faithfulness /
exactness from the perf_v2 battery (`quality_status` + run-to-run sha)
on `/tmp/v100_ch1/manifest.csv` (Axis-1 self-stability + Axis-2 FP8-vs-FP16 agreement). Kernel
details + the dense-vs-MoE profile in the code repo's `docs/COALESCED_FP8_GEMV.md` and `docs/SESSION_LOG.md`.*
