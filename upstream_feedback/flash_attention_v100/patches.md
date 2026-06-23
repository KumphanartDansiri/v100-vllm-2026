# Patches — flash-attention-v100

Canonical: commit **`ccb6557`** ("V100 paged-KV + CUDA 12.6 fixes") in our fork clone of
`flash-attention-v100`. Applyable copy: [ccb6557.patch](ccb6557.patch) (3 files, +4/−4). Apply with
`git am < ccb6557.patch` or `patch -p1 < ccb6557.patch`.

## Fix 1 — paged-KV tile must divide the page *(correctness)*
`include/forward.h`: `BLOCK_N_128 160 → 128`. The real invariant is `page_block_size % BLOCK_N == 0`
(D=32/64/256 already satisfy it; D=128 at 160 and D=16 at 512 do not). After the fix: cos = 1.000000 vs
fp32 reference up to 24k tokens with shuffled block tables; measured perf cost ≈ nil (112.6 → 109.9
ms/layer at 26k).

## Fix 2 — CUDA-12.6 build *(portability)*
`include/mat_mul.h`: `__tanhf → tanhf` (2 sites; `__tanhf` is 12.8+/12.9-only, softcap-only path).
`setup.py`: relax the torch CUDA pin `12.9 → 12.6`. Builds + passes tests on torch 2.11+cu126 / nvcc
12.6; perf equal across toolchains. (Optionally guard with `#if CUDART_VERSION >= 12080` instead of the
bare swap.)

## Fix 3 — strided-Q misread *(documented, not patched)*
The low-level `varlen_fwd` assumes dense `[T, H_Q*D]` query rows (host checks only `stride(-1)==1`); a
fused-QKV `.split()` view (row stride = full qkv width) is silently misread (cos ≈ 0.5 at GQA 12/1). We
densify Q in our adapter (~0.2 ms at 24k). **Options for upstream:** (a) loud doc note at the binding,
(b) host-side stride check `q.stride(0) == H_Q*D` → error/internal copy, or (c) thread q strides through
to the kernel like upstream FA2. Not changed in the fork — maintainer's call on the preferred shape.
