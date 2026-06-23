# Reproducers — flash-attention-v100

Both scripts live in the implementation repo (**fp8-w8a16-sm70**) under `tools/`; they run against the
ai-bond kernel on V100 silicon. Listed here so a maintainer can re-derive each finding independently.

## Fix 1 — paged-KV straddle
**Script:** `fa_v100_paged_smoke.py` (paged varlen, D=128).
**Setup:** two sequences with interleaved physical blocks (seq0 → blocks [0,2], seq1 → [1,3]); write a
sentinel (`100.0`) into all unused slots.
**Before fix (`BLOCK_N_128=160`):** output contains the sentinel — `max_abs ≈ 100.7`, `cos ≈ 0.004`
(the 160-row tile walks past the 256-row page into the neighbour's KV).
**After fix (`=128`):** exact, `cos = 1.000000`.

## Fix 3 — strided-Q
**Script:** `fa_v100_longseq_check.py --d 128 --hq 12 --hk 1` (qkv-split case).
**Setup:** allocate `[L, H_Q*D + 2*H_K*D]`, view the first `H_Q*D` columns as `[L, H_Q, D]`, compare
the kernel output on that strided view vs on `.contiguous()`.
**Result:** strided view `cos ≈ 0.25–0.5` (silently wrong, full speed); `.contiguous()` is exact —
demonstrating the dense-row assumption.

## Correctness gates already passed (V100 silicon)
ai-bond's own `test.py`; the paged smoke above (interleaved tables, garbage tails, in-place out, LSE
shape); long-seq sweep 512→24k (standalone and vLLM-interleaved `kv.unbind` strided cache views);
`Sq<Sk` bottom-right causal (q=1/64/4096 vs sk=8k/24k — the `seqlen_offset` path is correct); mixed
decode+prefill varlen batches; D=256 at GQA 4/1, 8/4, 4/2. Throughput 8.0–8.4× vs vLLM's Triton
attention at 26k. Full matrix in [report.md](report.md).
