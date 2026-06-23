# Draft issue — fused-MoE small-M decode default (`BLOCK_SIZE_K=128`) is pathological on sm_70

*Draft text for a `vllm-project/vllm` issue. Framed as a finding + a question, not an sm_70-support
request. Numbers are V100-only; the cross-arch question is raised honestly, not asserted.*

---

**Summary.** On Tesla V100 (sm_70), stock fp16/bf16 fused-MoE **decode** runs slower than a same-class
dense model — backwards for a sparse MoE. The cause is the decode-branch tile default in
`get_default_config`, specifically `BLOCK_SIZE_K=128`.

**Observed (TP4, fp16, cudagraph `FULL_DECODE_ONLY`, ns=8, vLLM 0.21.0 + CUDA 12.6, Triton 3.6):**

| model | stock decode | same-class dense |
|---|---|---|
| Qwen3.6-35B-A3B (MoE) | 15.6 tok/s | 27B dense ≈ 37–41 |
| gemma-4-26B-A4B (MoE) | 10.9 tok/s | 31B dense ≈ 17.6 |

**Root cause (code-traced + measured).** `fused_moe.py::get_default_config` has no sm_70 case (only
ROCm). Its decode branch (M ≤ 64) returns `BLOCK_SIZE_K=128` (+ `num_stages=4`). On Volta — no
`cp.async`, 96 KB shared memory — `BLOCK_K=128` register-spills Triton's codegen; spill traffic
contends on HBM and per-call cost grows ~linearly with batch. Isolated kernel time at M=1 scales
`BLOCK_K` 64→632 µs, 128→1450 µs, 256→2300 µs.

**What is *not* the cause:** `num_stages`. A 4/3/2 end-to-end sweep with the tile held fixed was flat
(15.57 tok/s on all arms, bit-identical output) — disproving the intuitive "no cp.async → fewer stages."
The dominant lever is `BLOCK_SIZE_K`.

**Fix that works (V100):** an sm<80, small-M branch returning `BLOCK_SIZE_K=64` (fat-N at the M≈8 knee).
End-to-end this restores sparse > dense: q35b 15.6→65.9 tok/s (4.2×), g26b 10.9→43.7 (4.0×); at 8
concurrent users, q35b aggregate 24.9→180.8. Output bit-identical to stock (pure speed). There's
in-tree precedent for a Volta case: `moe_fused_mul_sum.py::_heuristic_config` already special-cases
`num_stages=2` for `is_sm80_before`; the main GEMM default just never got the same treatment.

**The question for maintainers.** We only have V100 data, so we're not claiming the small-M default is
wrong elsewhere — but: *is `BLOCK_SIZE_K=128` the best small-M (M ≤ 64) decode default even on
`cp.async` architectures, or is it tuned for prefill-sized M?* That's worth a look independent of sm_70.

**What we can contribute.** The two measured V100 config JSONs below (a data contribution like the
~hundreds already in-tree). The sm_70 heuristic code targets an arch vLLM doesn't ship, so we're not
proposing it here; it's offered to aphrodite-engine, where broad-arch support is in scope.

## Attached configs
- `configs/E=256,N=128,device_name=Tesla_V100-SXM2-32GB.json` — Qwen3.6-35B-A3B, expert shape
  E=256 / per-rank N=128 (TP4).
- `configs/E=128,N=176,device_name=Tesla_V100-SXM2-32GB.json` — gemma-4-26B-A4B, expert shape
  E=128 / per-rank N=176 (TP4).

Filenames are vLLM's exact `get_moe_configs` lookup convention, so they drop into
`vllm/model_executor/layers/fused_moe/configs/` as-is. Provenance and scope: [configs/README.md](configs/README.md).
