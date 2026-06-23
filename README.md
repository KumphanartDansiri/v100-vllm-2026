# V100 vLLM in 2026 — running modern LLMs on Tesla V100 (sm_70)

Tesla V100s are everywhere on the secondhand/datacenter-surplus market and "unsupported" by
modern inference stacks — but they are far from dead. This is a measured, reproducible write-up of
running **7 modern model families on 8×V100-32GB with vLLM 0.19 + 0.21 in 2026**: what loads, what
flags you need, a vLLM MoE bug that costs 4–9× and its fix, and a custom FP8 plugin that makes large
MoE models both fit *and* decode faster than FP16.

> Every number here comes from one source of truth — [`data/benchmark_matrix.csv`](data/benchmark_matrix.csv) —
> extracted from raw logs by [`scripts/build_matrix_from_results.py`](scripts/build_matrix_from_results.py)
> and rendered by [`scripts/render_tables.py`](scripts/render_tables.py). Read
> [`docs/methodology.md`](docs/methodology.md) before comparing any two numbers.

## The one thing to know

vLLM's **prebuilt pip wheels dropped practical V100 support at 0.21** (built on CUDA ≥12.8, whose
arch list starts at sm_75). But vLLM's **source still has an sm_70 path** — its CMake `<12.8` branch
still lists `7.0`. So **building from source on a CUDA 12.6 toolchain re-enables V100 with zero source
patches** — and we do it for **both engines: 0.21 for the newest models, 0.19 because it's usually
faster on decode.** That's the unlock; everything else builds on it.

## Tested stack

| Component | Version |
|---|---|
| GPU | 8× NVIDIA V100-32GB SXM2 (sm_70) |
| Engines | vLLM 0.19.x and 0.21.x, **source-built** on CUDA 12.6 (Gemma-4 + GLM-4.7 use a transformers-5 / cu128 image on 0.19) |
| Torch / CUDA | 2.11 / cu126 (0.21) · 2.10 / cu128 (0.19-tf5) |
| FP8 + MoE plugin | companion **fp8-w8a16-sm70** @ tag `fp8-v100-2026-matrix` |
| FlashAttention-V100 | ai-bond fork, engine-specific ABI builds (0.21 + a 0.19 rebuild for MLA) |

FP8 and patched-MoE numbers come from the plugin, **not stock vLLM**; stock behaviour is reported
separately where it applies.

## Chapters (the write-up)

1. [V100 in 2026: dual-engine source builds + modern model support](docs/01_v100_in_2026.md) — what
   loads, the flags, and the 7-model single-user baseline + full per-engine (0.19 / 0.21) matrices.
2. [The FP16 MoE bug](docs/02_fp16_moe_fix.md) — stock FP16 MoE is 4–9× slower than it should be on
   V100 from one bad default (`BLOCK_K=128`, not the hardware). Root cause, fix, upstream story.
3. [CUDAGraph & baselines](docs/03_eager_vs_cudagraph.md) — why eager numbers mislead; what a fair
   serving baseline is.
4. [MTP / speculative decode](docs/04_mtp.md) — where it helps, kept separate from base decode.
5. [The FP8 plugin](docs/05_fp8_plugin.md) — custom W8A16 sm_70 kernels; FP8 as a decode-*speed* path
   (beats FP16 at low concurrency, and across *all* concurrency for MoE), not just a residency trick.
6. [Model pages](docs/06_model_results_template.md) — one per family (7):
   [Qwen3.6-27B](models/qwen3_6_27b.md) ·
   [Qwen3.6-35B-A3B](models/qwen3_6_35b_a3b.md) ·
   [Qwen3.5-122B-A10B](models/qwen3_5_122b_a10b.md) ·
   [gemma-4-31B](models/gemma4_31b.md) ·
   [gemma-4-26B-A4B](models/gemma4_26b_a4b.md) ·
   [GLM-4.5-Air](models/glm4_5_air.md) ·
   [GLM-4.7-Flash](models/glm4_7_flash.md)
7. [Dead ends & lessons](docs/07_dead_ends_and_lessons.md) — the optimizations we measured and
   *didn't* ship (num_stages, FP8 route/scatter, w2, FA-for-ViT, cp.async-imitation), and why that
   makes the wins trustworthy.
8. [Acknowledgements & upstream feedback](docs/08_acknowledgements.md) — what we built on and what we
   send back, one page per project:
   [flash-attention-v100 (ai-bond)](acknowledgements/flash_attention_v100.md) ·
   [vLLM](acknowledgements/vllm.md) ·
   [aphrodite-engine](acknowledgements/aphrodite.md) ·
   [1Cat-vLLM (1CatAI)](acknowledgements/onecat_vllm.md)

## Status

This repo reaches a *minimum coherent state* (shared CSV, shared definitions) before the chapter
posts go live. **Chapter 1 and the SSOT matrix are current** (dual-engine, 7 models, frozen at tag
`fp8-v100-2026-matrix`); **later chapters and model pages are being reconciled to the same
dual-engine framing.** Cells we have not measured are shown **absent, never zero**.

## Code

The FP8 plugin + the MoE fix live in the companion repo **fp8-w8a16-sm70** (link TBD on publish).
This repo is the *evidence/results*; that repo is the *software*.

## Known limitations (read before quoting numbers)

- V100 needs a **source build on CUDA 12.6/12.9**; pip wheels won't run.
- The **FP8 plugin is custom/local** sm_70 kernels, not upstream.
- **TP is fit-bounded per model** — not every model runs at every TP; infeasible cells are omitted.
- **eager ≠ serving baseline** (3–5× slower than cudagraph); headline numbers are cudagraph.
- FP16-MoE numbers labeled `stock(pre-moe-patch)` are the un-fixed default — see Chapter 2.
- **FP8 runs on both engines** with two exceptions noted in the tables: gemma-4-26B-A4B FP8 on 0.19
  (a stock `gemma4.py` MoE error) and GLM-4.7-Flash, which needs the MLA / FlashAttention-V100 path.
