# Chapter 2 — The FP16 MoE bug: 4–9× left on the floor by one default

> **Status: DRAFT** — numbers provisional until the final freeze rerun ([docs/FINAL_RERUN.md](FINAL_RERUN.md)). Tables auto-render from `data/benchmark_matrix.csv`.


**Claim:** stock vLLM runs FP16 Mixture-of-Experts models *slower than a same-class dense model*
on V100 — which is backwards, since a sparse MoE touches less weight per token. It's not the
hardware. It's a single config default that's wrong for Volta, and a one-file fix recovers 4–9×.

## Symptom

| model | stock FP16 MoE | same-class dense FP16 |
|---|---|---|
| Qwen3.6-35B-A3B (3B active) | ~15.5 tok/s | Qwen3.6-27B = 37–41 |
| gemma-4-26B-A4B (4B active) | ~10.9 tok/s | gemma-4-31B = 17.6 |

## Root cause (measured, not guessed)

`fused_moe.py::get_default_config` has no sm_70 case. Its **decode branch (M≤64) picks
`BLOCK_SIZE_K=128`**, which register-spills Triton's Volta codegen (no `cp.async`, 96 KB smem) —
spill traffic contends on HBM and the cost grows ~linearly with batch. **`num_stages` is NOT the
cause** (an e2e sweep of 4/3/2 was flat); the lever is `BLOCK_K`: 64→632 µs, 128→1450, 256→2300
at M=1. Prefill already gets `BLOCK_K=64` — only decode is hit.

## Fix & result (table rendered from the SSOT)

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

Single-stream: 35B **15.6→65.9 (4.2×)**, gemma **10.9→43.6 (4.0×)**, output bit-identical. The win
*grows with concurrency* (stock degrades ~linearly with batch): at 8 users, 35B per-user
**3.2→22.8**, aggregate **24.9→174**. Two fix forms — a default-ON `BLOCK_K=64` heuristic (any
model/TP) and per-shape autotuned config JSONs (the `+tuned-json` rows, ~5–10% over the heuristic
at concurrency).

## Upstream

Reported to **both** engines: vLLM (our main engine) as a finding + V100 config data (the decode
`BLOCK_K` default may be worth checking on cp.async arches too); aphrodite as a full PR including
the sm<80 heuristic. The fix is fp16/bf16-MoE only — it does not touch the FP8 path.

*Evidence: `results/moe_stages_ab_q35b_*`, `results/moe_decode_tile_sweep_*`. Full root-cause trace
in the code repo's `docs/V100_OPTIMIZATION_FINDINGS.md`.*