# V100 fused-MoE config JSONs — provenance & scope

Two autotuned Triton fused-MoE configs for Tesla V100. The filenames are vLLM's exact
`get_moe_configs` lookup convention (`E={experts},N={per-rank N},device_name={dev}.json`), so each is a
**drop-in** for `vllm/model_executor/layers/fused_moe/configs/`. The JSONs are kept pure (no metadata
keys, which the loader wouldn't expect); provenance lives here.

Canonical source: fp8-w8a16-sm70 `src/fp8_w8a16_sm70/moe_configs/`, produced by `tools/moe_volta_tune.py`
(a feasibility-pruned, shell-ordered tuner — an a-priori `num_stages*(BM*BK+BK*BN)*2 ≤ 96 KB` smem
filter drops infeasible tiles *without compiling* them, which is what makes a Volta tune minutes not days).

## `E=256,N=128,device_name=Tesla_V100-SXM2-32GB.json`
- **Motivated by:** Qwen3.6-35B-A3B (MoE), TP4, fp16.
- **GPU:** Tesla V100-SXM2-32GB (sm_70).
- **Tested on:** vLLM 0.21.0 + CUDA 12.6, Triton 3.6, cudagraph `FULL_DECODE_ONLY`, ns=8.
- **Result:** single-stream decode 15.56 → 65.85 tok/s (4.2×); 8-user aggregate 24.9 → 180.8. Output
  bit-identical to stock.

## `E=128,N=176,device_name=Tesla_V100-SXM2-32GB.json`
- **Motivated by:** gemma-4-26B-A4B (MoE), TP4, fp16.
- **GPU:** Tesla V100-SXM2-32GB (sm_70).
- **Tested on:** vLLM 0.21.0 + CUDA 12.6, Triton 3.6, cudagraph `FULL_DECODE_ONLY`, ns=8.
- **Result:** single-stream decode 10.91 → 43.71 tok/s (4.0×); 8-user per-user 3.56 → 20.27.

## Scope (please read before treating as a default)
- **fp16/bf16 unquantized MoE only**; quantized paths (fp8/int4) untouched.
- These cover the **decode** range (M = 1–64) — where the `BLOCK_K=128` pathology bites. Prefill (M > 64)
  already gets `BLOCK_K=64` from the stock default. For a complete upstream file, extend the tune to the
  full M ladder (1–4096).
- **TP-specific:** `N` is the per-rank shard, so re-tune per (model, TP).
- Intended as **Volta decode/small-batch MoE data**, *not* a universal default change.
