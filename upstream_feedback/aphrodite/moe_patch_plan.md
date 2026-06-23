# aphrodite — FP16-MoE fix as a real PR

The same Volta fused-MoE finding offered to vLLM as a diagnostic ([../vllm/moe_block_k_issue.md](../vllm/moe_block_k_issue.md))
can land in aphrodite as **code**, because sm_70 is in scope here.

## The PR
1. **sm<80 small-M heuristic** in `get_default_config` (unquantized path). Same shape as our reference
   impl; mirrors aphrodite's own existing `moe_fused_mul_sum.py::_heuristic_config` (which already
   special-cases `num_stages=2` for `is_sm80_before`):

   ```python
   if has_device_capability(70) and not has_device_capability(80) and M <= 64:
       if M <= 4:
           return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                   "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
       return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
               "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2}
   ```
2. **The two V100 config JSONs** ([../vllm/configs/](../vllm/configs/)) — `get_moe_configs` runs before
   `get_default_config`, so tuned shapes use the exact JSON and everything else falls to the heuristic.

## Credit
Frame the PR as building on aphrodite's existing sm_70 approach (the rewire above, and their
`_heuristic_config` precedent) — this is filling in the main MoE GEMM default they hadn't gotten to,
not a new direction.

## Caveat / scope
fp16/bf16 only; covers decode (M ≤ 64). For a complete file, extend the tune to the full M ladder. The
structural ceiling remains (Triton's sm_70 MoE GEMM has no tensor-core `tl.dot` path — ~40× off the
bandwidth floor); this fix removes the *pathology*, not the architectural gap.

## A compressed-tensors FP8 follow-up (plan, not ready)
Separately, aphrodite's `compressed_tensors_w8a16_fp8` is sm_75-gated; lowering the capability gate to
70 + wiring a CUDA-core dequant fallback would bring FP8 checkpoints to Volta. The gate-lowering and
fallback shape are contributable; the **kernel** itself still needs extracting from our plugin into a
standalone form first, so this is flagged as future work, not part of the MoE PR above.
