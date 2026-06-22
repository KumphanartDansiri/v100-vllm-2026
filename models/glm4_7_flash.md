# GLM-4.7-Flash (BF16, MLA) — V100 model page

> **Status: pending SSOT refresh.** This page was drafted from earlier GLM-4.7 validation runs
> (vLLM 0.21 2026-06-15, vLLM 0.19 2026-06-17) and has **not yet been reconciled with the current
> dual-engine `perf_v2` matrix — the numbers below are superseded.** For current numbers, use
> Chapter 1 and `data/benchmark_matrix.csv`. **This is the first MLA-attention model to generate
> on the V100 (`sm_70`) stack.**

GLM-4.7-Flash is a **~31B-total / ~3B-active lite-MoE** (`Glm4MoeLiteForCausalLM`, 47 layers,
64 routed + 1 shared expert, 4 active). Its attention is **MLA** (multi-head latent attention:
`q_lora_rank=768`, `kv_lora_rank=512`, `qk_head_dim=256` = 192 nope + 64 rope, `v_head_dim=256`).
The published checkpoint is **BF16 / unquantized** — verified: 9656 BF16 + 47 F32 tensors, **zero
FP8 tensors, no `quantization_config`**. It is served `--dtype float16` on V100, TP=4. (So this is
*not* an FP8-plugin model; none of the FP8 kernels engage. It is here to prove the MLA path.)

## The wall it broke through
MLA models **load and decode** on Volta but **crash on the first token**: vLLM offers only the
sm80+ FlashAttention MLA *prefill* backend for Volta, so prefill raises
`FlashAttention only supports Ampere GPUs or newer`. Unblocking it took **three stacked,
env-gated patches** (decode stays the stock TritonMLA path):
1. **MLA prefill interpose** (`VLLM_V100_MLA_PREFILL=1`) — route the prefill choke point through
   the ai-bond flash-attention-v100 **dense varlen** kernel (GLM's qk=v=256 is an exact tile → no
   padding). numcheck vs fp32 reference = **cos 1.000000** on all shapes (incl. the 2-chunk LSE merge).
2. **Decode smem clamp** — TritonMLA's grouped decode kernel requests ~100 KB shared memory for
   GLM's 512+64 latent (over V100's 96 KB); clamp the per-iter KV tile (`BLOCK_N 32→16`), correctness-neutral.
3. **Decode cudagraph enable** (`VLLM_V100_MLA_DECODE_CUDAGRAPH=1`) — let TritonMLA decode be
   cudagraph-captured (it was gated off by an unset support flag). **6.0 → ~38 tok/s = 6.2×.**

All three work on **both** engines. The A/B's OFF arm (patch disabled) reproduces the blocker
(`ampere-crash=9`, no output); the ON arm generates cleanly (`ampere-crash=0`).

## Best serving config
- **TP4, cudagraph** (`mode=0` + `FULL_DECODE_ONLY`) with `VLLM_V100_MLA_PREFILL=1` +
  `VLLM_V100_MLA_DECODE_CUDAGRAPH=1`. **cudagraph is MANDATORY** for a usable decode rate
  (eager = 6 tok/s). ~15.6 GB/GPU at TP4.
- One launcher reproduces every number here, on either engine:
  `tools/glm47_flash_mla_v100_ab.sh` (`ENGINE=021|019`, streams TTFT + steady-state decode for a
  short and a long/chunked prompt; auto-builds the per-engine ai-bond `.so`).

## Runs on BOTH engines
The arch `Glm4MoeLiteForCausalLM` is registered in **stock vLLM 0.19 and 0.21** (each with its own
`glm4_moe_lite.py`, `trust_remote_code=False`). The same MLA-unblock patch drives both — it resolves
the prefill choke point per engine (0.21's dedicated prefill backend vs 0.19's inline MLA impl;
identical method signature). The ai-bond `.so` is torch-ABI-specific, so it is rebuilt per engine
(torch 2.11 for 0.21, torch 2.10 for 0.19) — the launcher does this automatically.

## Measured — single-stream (cudagraph), both engines

<!-- render:model:GLM-4.7-Flash -->
| vLLM | variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|---|
| 0.21.0/cu126 | bf16 | TP4 | 1 | fp16mla+cudagraph | 37.2 | 37.2 | - | results/glm47_mla_v100_20260615/cudagraph_ON_SUMMARY.txt |
| 0.19.0/cu128 | bf16 | TP4 | 1 | fp16mla+cudagraph | 38.80 | 38.80 | 2.06 | results/glm47_mla_019_20260617_144003/SUMMARY.txt |
<!-- endrender -->

## How to read it
- **~37–39 tok/s single-stream decode on both engines** — comfortably in the usable band, on the
  first MLA model to run on V100 at all. 0.19 (38.80) ≈ slightly ahead of 0.21 (37.2); engine-invariant.
- **TTFT (short prompt) = 2.06 s** (cleanly isolated on the 0.19 streaming run; the 2026-06-15 0.21
  harness didn't separate TTFT, hence "-").
- **Long-context TTFT is the open item**: a ~3k-token prompt takes **~31.6 s to first token**. MLA
  prefill is inherently **eager**, and GLM's MoE shape (`E=64, N=384`) has no tuned Volta config yet,
  so prefill is the slow part. The *decode* chapter is done; the prefill/TTFT chapter (autotune that
  MoE JSON) is future work.
- **Correctness**: numcheck cos = 1.0 (exact attention math), greedy run-to-run **exact**, **7/8**
  verifiable chat Q&A correct (capital, 17×23=391, exact-token instruction-following, code) — the one
  miss is a small-reasoning-model miscount, not numerical corruption.

## Caveats
- **Reasoning model** (`<think>…</think>`, gen-config `temp=1.0`). Coherent + correct at realistic
  length (≤1024, rep ≈ 0.05–0.09). Under the standard suite's `ignore_eos@4096` forced-length stress
  it degenerates into repetition loops (rep 0.67 greedy) — a **reasoning-model × forced-length protocol
  mismatch**, not the MLA implementation (it's clean at 1024 and numcheck is exact). For production use
  natural stop, not `ignore_eos`.
- **BF16, not FP8** — no FP8 download exists; the FP8 plugin does not engage for this model.
- The three MLA patches are **experimental, env-gated, local** (sm_70 only); decode rides stock TritonMLA.
- Numbers are `max-model-len=8192`, TP4. The long-context TTFT above is the untuned-prefill state.
