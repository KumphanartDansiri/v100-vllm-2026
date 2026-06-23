# aphrodite — sm_70 build re-enable

A four-commit series that re-adds Tesla V100 (sm_70) to an aphrodite 0.21 source build. Validated:
builds on a CUDA-12.6 toolchain and serves modern models on 8×V100 (gemma-4-31B ≈ 29.22 tok/s,
cudagraph — parity with the vLLM-0.19-tf5 path).

**Canonical location (implementation side):** `aphrodite_salvage_archive/v100-sm70-patches/` (kept out
of this publication repo per the "code lives in the implementation repo" convention). Listed here so the
shape is reviewable and a maintainer can re-derive or request the patch files.

## The commits
1. **Re-add `7.0` to the CUDA ≥12.8 arch list.** The ≥12.8 CMake branch starts at sm_75; add 7.0 back.
   (On CUDA 12.6 the `<12.8` branch already lists 7.0 — this commit is what lets the *newer* toolchains
   build Volta too.)
2. **Exclude EXL3 kernels** — they're sm_80+ only; compiling them on sm_70 fails.
3. **Guard the Marlin op registrations** — otherwise an undefined-symbol trap at load on Volta.
4. **V100 smoke tooling** — a cu126 Dockerfile + build/smoke scripts (the same shape as our vLLM-0.21
   build).

A companion **cu126 Dockerfile diff** (use CUDA 12.6 base + cu126 torch instead of 12.8) is preserved
alongside; it is the same toolchain fix that makes the build pick up sm_70 cleanly.

## Apply
The series applies on top of the aphrodite v0.21.0 tag. Request the `.patch` files (or a branch) and
`git am` them; build with the cu126 Dockerfile. Smoke with the bundled `smoke_gemma4.sh`.

## Note
This is **not** aphrodite-maintained today — it's our rewire on top of their tag. The point of
upstreaming is to move it from "our local patch" to "a supported path," which only the maintainer can do.
