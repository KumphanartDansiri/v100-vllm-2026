# V100 in 2026 — running modern LLMs on Tesla V100 (sm_70)

Tesla V100s are everywhere on the secondhand/datacenter-surplus market and "unsupported" by
modern inference stacks — but they are far from dead. This is a measured, reproducible write-up of
running **6 modern model families on 8×V100-32GB with vLLM 0.21 in 2026**: what loads, what flags
you need, a vLLM MoE bug that costs 4–9× and its fix, and a custom FP8 plugin that lets large MoE
models fit and serve.

> Every number here comes from one source of truth — [`data/benchmark_matrix.csv`](data/benchmark_matrix.csv) —
> extracted from raw logs by [`scripts/build_matrix_from_results.py`](scripts/build_matrix_from_results.py)
> and rendered by [`scripts/render_tables.py`](scripts/render_tables.py). Read
> [`docs/methodology.md`](docs/methodology.md) before comparing any two numbers.

## The one thing to know

vLLM's **prebuilt pip wheels dropped practical V100 support at 0.21** (built on CUDA ≥12.8, whose
arch list starts at sm_75). But vLLM's **source still has an sm_70 path** — its CMake `<12.8` branch
still lists `7.0`. So **building vLLM 0.21 from source on a CUDA 12.6 toolchain re-enables V100 with
zero source patches.** That's the unlock; everything else builds on it.

## Chapters (the write-up)

1. [V100 in 2026: vLLM 0.21 source build + modern model support](docs/01_v100_in_2026.md) — what
   loads, the flags, the 0.19/0.21 context, the 6-model overview table.
2. [The FP16 MoE bug](docs/02_fp16_moe_fix.md) — stock FP16 MoE is 4–9× slower than it should be on
   V100 from one bad default (`BLOCK_K=128`, not the hardware). Root cause, fix, upstream story.
3. [CUDAGraph & baselines](docs/03_eager_vs_cudagraph.md) — why eager numbers mislead; what a fair
   serving baseline is.
4. [MTP / speculative decode](docs/04_mtp.md) — where it helps, kept separate from base decode.
5. [The FP8 plugin](docs/05_fp8_plugin.md) — custom W8A16 sm_70 kernels; where resident FP8 wins.
6. Model pages — one per family, [template](docs/06_model_results_template.md):
   [Qwen3.6-27B](models/qwen3_6_27b.md) ·
   [Qwen3.6-35B-A3B](models/qwen3_6_35b_a3b.md) ·
   [Qwen3.5-122B-A10B](models/qwen3_5_122b_a10b.md) ·
   [gemma-4-31B](models/gemma4_31b.md) ·
   [gemma-4-26B-A4B](models/gemma4_26b_a4b.md) ·
   [GLM-4.5-Air](models/glm4_5_air.md)

## Status

This repo reaches a *minimum coherent state* (shared CSV, shared definitions) before the chapter
posts go live. Chapters 1–4 are backed by measured data; Chapter 5 (FP8 plugin) is consolidating;
the model pages (Ch6) fill in as per-model feasible-TP × concurrency sweeps land. Cells we have not
measured are shown **absent, never zero**.

## Code

The FP8 plugin + the MoE fix live in the companion repo **fp8-w8a16-sm70** (link TBD on publish).
This repo is the *evidence/results*; that repo is the *software*.

## Known limitations (read before quoting numbers)

- V100 needs a **source build on CUDA 12.6/12.9**; pip wheels won't run.
- The **FP8 plugin is custom/local** sm_70 kernels, not upstream.
- **TP is fit-bounded per model** — not every model runs at every TP; infeasible cells are omitted.
- **eager ≠ serving baseline** (3–5× slower than cudagraph); headline numbers are cudagraph.
- FP16-MoE numbers labeled `stock(pre-moe-patch)` are the un-fixed default — see Chapter 2.
