# Chapter 9 — Precision × tensor-parallelism: FP16 vs FP8 vs GPTQ-Int4, full TP vs half TP

Two exact Qwen3.5 checkpoints make the cleanest precision comparison on V100:

- **Qwen3.5-27B** — dense; every weight is active each token.
- **Qwen3.5-35B-A3B** — sparse MoE; about 3B of 35B parameters are active each token.

For each model we compare the three formats a V100 operator actually has: the official **FP16** checkpoint, the **FP8 W8A16** plugin path, and the **GPTQ-Int4** re-quant. The important axis is not only precision; it is tensor parallelism. TP4 is the normal four-GPU baseline. TP2 is the half-GPU deployment question: can the same model run on two V100-32GB cards, freeing the other two cards for another replica or another job?

All numbers are steady-state per-user decode tok/s, `max_model_len=4096`, 512 generated tokens, temperature 0, users C1/C2/C4/C8. FP16 and FP8 are served on vLLM 0.21/cu126 through the plugin stack; GPTQ-Int4 uses the existing stock vLLM 0.18/cu128 harness. Raw run directories live in the FP8 repo under `results/q27b_exact_triad_*` and `results/q35b_exact_triad_*`; consolidated rows are in `data/benchmark_matrix.csv` and `vllm-fp8-w8a16-sm70/results/qwen35_triad_matrix_20260624.csv`.

## Capacity First

Raw weights alone understate the problem because vLLM also needs KV cache, CUDA workspaces, graph capture room, and allocator slack. The TP2 run is therefore treated as the source of truth for fit.

| Model | Format | Total weights | Per-GPU @ TP4 | Per-GPU @ TP2 | TP2 result |
|---|---|---:|---:|---:|:---:|
| 27B dense | FP16 | 51 GB | ~13 GB | ~25 GB | **OOM** |
| 27B dense | FP8 W8A16 | 28 GB | ~7 GB | ~14 GB | Fits |
| 27B dense | GPTQ-Int4 | 28 GB | ~7 GB | ~14 GB | Fits |
| 35B-A3B MoE | FP16 | 66 GB | ~17 GB | ~33 GB | **OOM** |
| 35B-A3B MoE | FP8 W8A16 | 34 GB | ~9 GB | ~17 GB | Fits |
| 35B-A3B MoE | GPTQ-Int4 | 22 GB | ~6 GB | ~11 GB | Fits |

The headline is sharper after measurement: **both FP16 checkpoints are off the board at TP2 under the real serve envelope.** FP8 and GPTQ fit. For the MoE, FP8 is the faithful-to-source format that lets the model run on half the GPUs; FP16 simply cannot make that deployment.

## Dense — Qwen3.5-27B

### TP4, Full Baseline

| Users | FP16 | FP8 W8A16 | FP8/FP16 | GPTQ-Int4 |
|---:|---:|---:|---:|---:|
| 1 | 39.08 | **52.50** | 1.34x | 69.18 |
| 2 | 31.11 | **42.52** | 1.37x | 55.54 |
| 4 | 30.30 | **31.73** | 1.05x | 47.40 |
| 8 | 29.30 | 20.29 | 0.69x | 44.15 |

### TP2, Half-GPU Deployment

| Users | FP16 | FP8 W8A16 | GPTQ-Int4 |
|---:|---:|---:|---:|
| 1 | OOM | 34.01 | 43.30 |
| 2 | OOM | 26.18 | 33.23 |
| 4 | OOM | 18.63 | 25.46 |
| 8 | OOM | 12.42 | 26.80 |

Dense read: at TP4, FP8 beats FP16 at 1-2 users, ties at 4, and loses at 8 where the FP16 tensor-core path benefits from batch. At TP2, the more important result is capacity: the FP16 checkpoint does not fit in the measured serving envelope, while FP8 and GPTQ both run.

## MoE — Qwen3.5-35B-A3B

### TP4, Full Baseline

| Users | FP16 | FP8 W8A16 | FP8/FP16 | GPTQ-Int4 |
|---:|---:|---:|---:|---:|
| 1 | 66.20 | **92.96** | 1.40x | 126.19 |
| 2 | 45.43 | **77.59** | 1.71x | 96.12 |
| 4 | 29.55 | **72.36** | 2.45x | 76.20 |
| 8 | 22.92 | **54.93** | 2.40x | 75.08 |

### TP2, Half-GPU Deployment

| Users | FP16 | FP8 W8A16 | GPTQ-Int4 |
|---:|---:|---:|---:|
| 1 | OOM | 82.20 | 98.96 |
| 2 | OOM | 64.21 | 74.19 |
| 4 | OOM | 53.82 | 50.53 |
| 8 | OOM | 39.26 | 51.86 |

MoE read: at TP4, FP8 wins over FP16 at every concurrency and the margin grows under load. At TP2, FP16 is not a candidate. GPTQ is often faster raw, but it is a lossy re-quant; FP8 is the source-format path that still fits and keeps strong decode speed on two cards.

## Deployment Takeaways

- **Same TP, dense:** FP8 is a real speed path at low concurrency, not just a memory trick; GPTQ is faster but lossy.
- **Same TP, MoE:** FP8 is the faithful format and faster than FP16 across the whole concurrency sweep.
- **Half TP:** FP16 fails for both exact Qwen3.5 examples in the measured vLLM serving envelope. FP8 and GPTQ are the formats that turn TP2 into a viable deployment.
- **Fleet planning:** for the MoE, FP8 turns a four-GPU faithful deployment into a two-GPU faithful deployment. On an 8x V100 box, that is the difference between two replicas and four replicas.

## Methodology Notes

- TP4 source: `results/q27b_exact_triad_20260624_113728/` and `results/q35b_exact_triad_20260624_115648/`.
- TP2 source: `results/q27b_exact_triad_tp2_20260624_174257/` and `results/q35b_exact_triad_tp2_20260624_174257/`.
- FP16 OOM means the fit-aware sweep marked the config infeasible at load (not counted as zero throughput). The two OOMs differ in kind, confirmed in the serve logs: the **35B-A3B is a hard weight-OOM** — ~33 GB/GPU exceeds the 32 GB card *before any KV cache* (`Failed to load model — not enough GPU memory`), so it fundamentally cannot fit at TP2. The **27B is a KV-cache-room OOM** — weights (~25 GB/GPU) load, but the standard `gpu_memory_utilization=0.85` / 4096-ctx envelope leaves no room for KV blocks (`No available memory for the cache blocks`). The 27B could be coaxed onto TP2 with a tighter envelope at a degraded operating point; the 35B cannot. Both are reported "OOM" = infeasible under the *standard* serve envelope.
- GPTQ C4 cells were noisy in both TP4 and TP2; tables use the median of two runs, matching the rest of the triad harness.
