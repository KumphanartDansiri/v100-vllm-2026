# Chapter 6 — Model pages

Chapter 1 gives the fleet matrix; these pages are the per-model deployment notes — reach for them when
you want to know what TP fits, which engine/image to use, what precision is worth serving, and what
caveats apply.

## How to read a model page
- **Headline numbers are base cudagraph decode, not MTP** (MTP is Chapter 4).
- **TP is fit-bounded** — an absent TP row means *not measured / not feasible*, not zero.
- **FP8 rows use the companion V100 plugin** unless noted; FP16/BF16 is stock vLLM.
- **Cold TTFT** is cold monolithic prefill; warm / chunked-prefill TTFT is a future SSOT add.
- **Quality / exactness** follows the methodology 5-test suite (Exact / Stable — see `methodology.md`).

## Engine / image / patch compatibility

Which runtime each model needs, and which of our patches apply. *Stock* = unmodified vLLM; the **FP8
plugin** and the **MoE patch** are the companion `fp8-w8a16-sm70` package.

| Official checkpoint(s) | vLLM 0.19 stock | vLLM 0.19 + transformers 5 | vLLM 0.21 stock¹ | MoE patch | FP8 plugin |
|---|---|---|---|---|---|
| `Qwen/Qwen3.6-27B` (+ `-FP8`) | Works | not needed | Works | — (dense) | Works, both engines |
| `Qwen/Qwen3.6-35B-A3B` (+ `-FP8`) | Works | not needed | Works | required for fast FP16 MoE | Works, both engines |
| `Qwen/Qwen3.5-122B-A10B-FP8` (+ `-GPTQ-Int4`) | Works | not needed | Works | — (no FP16 path) | Works, both engines |
| `google/gemma-4-31B-it` (+ RedHatAI `-FP8`) | needs tf5 | Works | Works | — (dense) | Works, both engines |
| `google/gemma-4-26B-A4B-it` (+ RedHatAI `-FP8`) | needs tf5 | FP16 works; FP8 MoE path errors | Works | required for FP16 MoE | **0.21 only**² |
| `zai-org/GLM-4.5-Air-FP8` | Works | not needed | Works | — (FP8 path)³ | Works, both engines |
| `zai-org/GLM-4.7-Flash` | needs tf5 + MLA patches⁴ | Works (tf5 + MLA patches) | Works (MLA patches⁴) | — (BF16 MLA) | no FP8 checkpoint |

¹ The 0.21 base image already ships **transformers 5.x**, so Gemma-4 / GLM-4.7-Flash run on it with no
separate upgrade — the tf5 step is a 0.19-only thing (0.19's base is transformers 4.57).
² gemma-4-26B FP8 runs **only on 0.21**: the `gemma4.py` fused-MoE path errors on 0.19 even with tf5;
FP16 runs on both.
³ GLM-4.5-Air's recommended config bundles the FP16-MoE Volta tune for the fallback path, but the
headline FP8 win is the routed-expert kernels, not the patch.
⁴ MLA models load on stock Volta but **crash on the first token** (Ampere-only MLA prefill); the
env-gated MLA patches (local, sm_70) unblock it on either engine. Decode rides stock TritonMLA.

## Pages
- [Qwen3.6-27B](../models/qwen3_6_27b.md) — dense; FP8 low-user speed win, FP16 takes high concurrency.
- [Qwen3.6-35B-A3B](../models/qwen3_6_35b_a3b.md) — the clean MoE FP8 showcase (FP8 wins every concurrency).
- [Qwen3.5-122B-A10B](../models/qwen3_5_122b_a10b.md) — flagship; FP8 vs GPTQ-Int4 at TP8.
- [gemma-4-31B](../models/gemma4_31b.md) — dense Gemma-4; FP8 residency + low-user win.
- [gemma-4-26B-A4B](../models/gemma4_26b_a4b.md) — MoE; the 0.21-only FP8 compatibility case.
- [GLM-4.5-Air](../models/glm4_5_air.md) — large FP8 MoE; broad concurrency win.
- [GLM-4.7-Flash](../models/glm4_7_flash.md) — BF16 MLA path; cudagraph mandatory.

*Every page's decode table auto-renders from the SSOT (`data/benchmark_matrix.csv`) between
`<!-- render:model:<name> -->` markers; table numbers are never hand-typed.*
