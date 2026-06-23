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
