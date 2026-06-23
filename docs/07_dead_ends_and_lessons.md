# Chapter 7 — Dead ends & lessons: the optimizations we *didn't* ship, and why

> **Status: Final** — verdicts final; any figures are claims/ranges, each cited to a result path.

Most write-ups show you the wins. Here are the things that looked obviously worth doing and **didn't
survive measurement.** This chapter exists because it's the honest backbone of the rest: the one real
win (the MoE fix, Chapter 2) is trustworthy precisely because the same measure-first discipline killed
four plausible siblings. If we'd shipped on intuition, half of these would be in the "wins" column,
wrong.

A recurring shape: a mechanism that *sounds* right is not evidence. Every entry below had a tidy
mechanism. The profiler disagreed each time.

## 1. `num_stages` — the plausible cause of the MoE slowdown that wasn't
**Hypothesis:** stock MoE picks `num_stages=4` (deep software pipeline), which needs Ampere's
`cp.async`; V100 lacks it, so deep pipelining should crater the kernel.
**Measured:** sweeping `num_stages` 4→3→2 end-to-end was **flat — 15.57 tok/s, byte-identical output.**
Not the lever. The real cause was one tile dimension over (`BLOCK_SIZE_K`, Chapter 2).
**Lesson:** the most mechanistically satisfying story was simply wrong; the flat sweep saved us from
"fixing" the wrong knob. *(results/moe_stages_ab_q35b_20260612_132040)*

## 2. FP8 "the headroom is in route/scatter" — plausible, measured minority
**Hypothesis:** our FP8 MoE GEMV is mature, so the remaining decode time must be in the glue —
routing, scatter, data movement.
**Measured:** profiling one decode step, the **GEMVs are 73% of GPU time; route/scatter glue is 27%**
— and the launch/dispatch overhead that *looked* huge in eager is replayed away by cudagraph in
production. The glue was a minority, not the target.
**Lesson:** "the kernel is mature so the cost must be elsewhere" is an assumption, not a finding.
*(results/moe_fp8_profile_20260613_072405)*

## 3. The `w2` GEMV — suspicious at M=1, self-healed under load
**Hypothesis:** `w2_gemm` was 44% of MoE GPU and **2.3× slower than `w13` despite half the weight
bytes** — surely *the* FP8 optimization target.
**Measured:** at 8 concurrent users (M=8) the anomaly **vanished** — `w13 ≈ w2`, and per-slot `w2`
got 5.2× more efficient. The M=1 penalty was fixed under-occupancy (its wide/short shape can't fill
the GPU at 8 routed slots), which **amortizes away with batch.** Not a kernel flaw; a low-load artifact.
**Lesson:** profile at the load you actually serve. A single-stream anomaly nearly sent us into a
kernel rewrite for a problem that disappears under concurrency. *(results/moe_fp8_profile_20260613_072405,
M=1 vs M=8 buckets)*

## 4. FlashAttention for the vision encoder — right API, wrong regime
**Hypothesis:** the V100 FlashAttention fork has exactly the varlen API a ViT needs; wire it in and
the vision tower gets faster than Torch SDPA.
**Measured:** the kernel only compiles head dims {16,32,64,128,256}; the SigLIP tower is **D=72**
(hard-errors). Padding 72→128 is numerically correct but **0.37–0.47× SDPA** (2–2.7× *slower*) at
every ViT sequence length; extrapolating past the pad tax, even a native D=72 kernel would land
~0.64–0.71× — still a loss. The reason: V100's SDPA already auto-selects the CUTLASS mem-efficient
kernel, and at ViT's *short* sequences FlashAttention's long-sequence advantage doesn't exist.
**Lesson:** "we have the kernel" ≠ "it wins here." The baseline was already optimal for that regime.
*(results/vit_fa_v100_d72_20260613_080508)*

## 5. Imitating `cp.async` in software — can't beat what the baseline already does
**Hypothesis:** V100 lacks Ampere's async global→shared copy (`cp.async`), so a hand-rolled software
pipeline could recover the missing overlap and beat the stock kernels.
**Measured / reasoned:** the `num_stages` sweep (#1) is the direct evidence — deeper software
pipelining plateaus at a simple double-buffer on V100, because each prefetch stage spends the very
registers/occupancy V100 uses to hide latency. And the baselines (CUTLASS mem-efficient SDPA, Triton)
*already* software-pipeline on Volta. "Imitate cp.async" isn't an untapped lever; it's table stakes
the baseline has. Beating it means out-engineering CUTLASS, not adding a missing optimization.
**Lesson:** before reinventing a hardware feature in software, check whether the library you're trying
to beat already did it. It had.

## The unifying lesson (why these failed, and the wins didn't)
There's a single principle under all five. V100 has fast tensor-core compute but none of the async
memory-feed hardware (`cp.async`, then Hopper's TMA, then Blackwell's TMEM) that later generations
added *specifically* to keep those cores fed. So V100 is competitive in the **memory-bound / resident
regime** (decode GEMV, FP8-resident weights — where our wins live) and at-ceiling in the
**compute/tile-staging regime** (prefill, big GEMM, ViT). Every dead end above was a bet on winning the
staging regime that three later silicon generations were built to fix; every win was *moving work into
V100's favorable regime.* The hardware drew the line; measurement found it.

**The discipline, in one sentence:** a plausible mechanism is a hypothesis, not a result — run the
microbench (or the profiler) before you write the kernel. It cost us a few hours of measurement and
saved us four wrong "optimizations."

*Full traces for every claim here: the code repo's `docs/V100_OPTIMIZATION_FINDINGS.md`.*
