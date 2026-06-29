# aphrodite-engine — acknowledgement & feedback

> **Status: Final** — contribution status as of the freeze; this is the *parked* alternative line.

**Ready-to-send packet:** [../upstream_feedback/aphrodite/](../upstream_feedback/aphrodite/) — the sm_70 build-rewire + MoE-PR plan.

**Upstream:** **[aphrodite-engine](https://github.com/dphnAI/aphrodite-engine)**
(dphnAI, formerly PygmalionAI; maintained by AlpinDale) — a
community vLLM-derived serving engine with a deliberately broad hardware-support stance.
**License: AGPL-3.0.**

## Why it's here
Early on we evaluated aphrodite as the engine to **freeze high on** — the "upstream-and-ride"
alternative to pinning a vLLM source build. The reasoning: vLLM dropped sm\_70 by policy (a Volta PR
there reverses project direction and is unlikely to land), whereas aphrodite's single-maintainer,
broad-arch culture makes a clean Volta contribution *reinforce* the project rather than fight it.

We ultimately made **vLLM 0.21 the primary frozen engine** (it proved the more reliable, broadest-model
line, and needs no arch patch at all on CUDA 12.6). aphrodite is **parked, not abandoned** — and on the
one model we measured head-to-head it reached **parity** (gemma-4-31B ≈ 29 tok/s, matching the
vLLM-0.19 path), so the alternative is real if the vLLM line ever closes.

## What we'd send back
A contribution is **prepared and clean**, should the project (or its users) want it:
- **sm\_70 build re-enable** — a small commit series: re-add `7.0` to the CUDA ≥12.8 arch list, exclude
  the sm\_80-only EXL3 kernels, guard the Marlin op registrations (an undefined-symbol trap on Volta),
  and add V100 smoke tooling. Validated: the four-commit set builds and serves modern models on V100.
- **The FP16-MoE fix as a real PR** — unlike vLLM, sm\_70 is in scope here, so the **`BLOCK_K` decode
  heuristic** plus the **autotuned V100 config JSONs** (Chapter 2) can go in as code, crediting the
  shared lineage.
- **A path for compressed-tensors FP8 on sm\_70** — lowering aphrodite's sm\_75-gated
  `compressed_tensors_w8a16_fp8` capability gate to 70 and wiring a CUDA-core dequant fallback. The
  *kernel* behind this still needs extracting from our plugin into a standalone form before it's
  PR-ready; the gate-lowering and fallback shape are the contributable part today.

## Credit & honesty
This page is deliberate about not over-claiming: aphrodite is the road we *scouted and parked*, not the
one we shipped on. The work above is offered in good faith and kept ready; if the maintainer wants it,
the build rewire and the MoE PR are the two pieces that land cleanly now.
