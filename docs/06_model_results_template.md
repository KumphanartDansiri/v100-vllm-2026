# Chapter 6 — Model-family pages

Chapter 1 gives the fleet matrix; these **model-family pages** are the per-family deployment notes —
each covers the official checkpoints in that family (typically an FP16/BF16 baseline and an FP8/Int4
sibling). Reach for them when you want to know what TP fits, which engine/image to use, what precision
is worth serving, and what caveats apply.

## How to read a model-family page
- **Headline numbers are base cudagraph decode, not MTP** (MTP is Chapter 4).
- **TP is fit-bounded** — an absent TP row means *not measured / not feasible*, not zero.
- **FP8 rows use the companion V100 plugin** unless noted; FP16/BF16 is stock vLLM.
- **TTFT** has its own per-page table: **cold** (cache-cold full prefill, worst case) vs
  **prefix-cache-hit** (repeated/shared prefix, best case). FP8 pays a prefill dequant tax even where
  it wins decode.
- **Quality / exactness** follows the methodology 5-test suite (Exact / Stable — see `methodology.md`).

## Engine / image / patch compatibility

Which runtime each model needs, and which of our patches apply. *Stock* = unmodified vLLM; the **FP8
plugin** and the **MoE patch** are the companion `fp8-w8a16-sm70` package.

One row per official checkpoint, grouped by family.
**Legend:** ✓ = works / applies · — = not needed or not applicable · *note text* = special requirement
or limitation (footnoted). Under **MoE patch**, ✓ = required for usable FP16-MoE decode (Chapter 2).

| Official model / checkpoint | 0.19<br>stock | 0.19<br>+ tf5 | 0.21<br>stock¹ | MoE<br>patch | FP8<br>plugin |
|---|:--:|:--:|:--:|:--:|:--:|
| **Qwen3.5-27B family** (featured) |  |  |  |  |  |
| `Qwen/Qwen3.5-27B` | ✓⁵ | — | ✓ | — | — |
| `Qwen/Qwen3.5-27B-FP8` | ✓⁵ | — | ✓ | — | ✓ |
| `Qwen/Qwen3.5-27B-GPTQ-Int4` | ✓⁵ | — | ✓ | — | — |
| **Qwen3.5-35B-A3B family** (featured) |  |  |  |  |  |
| `Qwen/Qwen3.5-35B-A3B` | ✓⁵ | — | ✓ | ✓ | — |
| `Qwen/Qwen3.5-35B-A3B-FP8` | ✓⁵ | — | ✓ | — | ✓ |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | ✓⁵ | — | ✓ | — | — |
| **Qwen3.6-27B family** |  |  |  |  |  |
| `Qwen/Qwen3.6-27B` | ✓ | — | ✓ | — | — |
| `Qwen/Qwen3.6-27B-FP8` | ✓ | — | ✓ | — | ✓ |
| **Qwen3.6-35B-A3B family** |  |  |  |  |  |
| `Qwen/Qwen3.6-35B-A3B` | ✓ | — | ✓ | ✓ | — |
| `Qwen/Qwen3.6-35B-A3B-FP8` | ✓ | — | ✓ | — | ✓ |
| **Qwen3.5-122B-A10B family** |  |  |  |  |  |
| `Qwen/Qwen3.5-122B-A10B-FP8` | ✓ | — | ✓ | — | ✓ |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | ✓ | — | ✓ | — | — |
| **gemma-4-31B family** |  |  |  |  |  |
| `google/gemma-4-31B-it` | needs tf5 | ✓ | ✓ | — | — |
| `RedHatAI/gemma-4-31B-it-FP8-Dynamic` | needs tf5 | ✓ | ✓ | — | ✓ |
| **gemma-4-26B-A4B family** |  |  |  |  |  |
| `google/gemma-4-26B-A4B-it` | needs tf5 | ✓ | ✓ | ✓ | — |
| `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` | needs tf5 | errors² | ✓ | — | 0.21 only² |
| **GLM-4.5-Air family** |  |  |  |  |  |
| `zai-org/GLM-4.5-Air-FP8` | ✓ | — | ✓ | —³ | ✓ |
| **GLM-4.7-Flash family** |  |  |  |  |  |
| `zai-org/GLM-4.7-Flash` | needs tf5⁴ | ✓⁴ | ✓⁴ | — | — |

¹ The 0.21 base image already ships **transformers 5.x**, so Gemma-4 / GLM-4.7-Flash run on it with no
separate upgrade — the tf5 step is a 0.19-only thing (0.19's base is transformers 4.57).
² gemma-4-26B FP8 runs **only on 0.21**: the `gemma4.py` fused-MoE path errors on 0.19 even with tf5;
FP16 runs on both.
³ GLM-4.5-Air's recommended config bundles the FP16-MoE Volta tune for the fallback path, but the
headline FP8 win is the routed-expert kernels, not the patch.
⁴ MLA models load on stock Volta but **crash on the first token** (Ampere-only MLA prefill); the
env-gated MLA patches (local, sm_70) unblock it on either engine. Decode rides stock TritonMLA.
⁵ The featured **Qwen3.5** pair was benchmarked on **0.21** (FP16/FP8) and stock **0.18** (GPTQ-Int4);
the **0.19** mark is by model-class equivalence with the architecturally-identical **Qwen3.6** family
(which passes on 0.19), not a separate Qwen3.5 run.

## Family pages
- [**Qwen3.5-27B**](../models/qwen3_5_27b.md) — **featured** dense pair; FP8 beats FP16 to ~C4, FP16 reclaims C8. Full precision×TP / TTFT / faithfulness profile in [Ch.5](05_fp8_plugin.md).
- [**Qwen3.5-35B-A3B**](../models/qwen3_5_35b_a3b.md) — **featured** MoE pair; FP8 wins every concurrency and is the only faithful format that fits at half-TP.
- [Qwen3.6-27B](../models/qwen3_6_27b.md) — dense; FP8 low-user speed win, FP16 takes high concurrency.
- [Qwen3.6-35B-A3B](../models/qwen3_6_35b_a3b.md) — the clean MoE FP8 showcase (FP8 wins every concurrency).
- [Qwen3.5-122B-A10B](../models/qwen3_5_122b_a10b.md) — flagship; FP8 vs GPTQ-Int4 at TP8.
- [gemma-4-31B](../models/gemma4_31b.md) — dense Gemma-4; FP8 residency + low-user win.
- [gemma-4-26B-A4B](../models/gemma4_26b_a4b.md) — MoE; the 0.21-only FP8 compatibility case.
- [GLM-4.5-Air](../models/glm4_5_air.md) — large FP8 MoE; broad concurrency win.
- [GLM-4.7-Flash](../models/glm4_7_flash.md) — BF16 MLA path; cudagraph mandatory.

*Every page's decode table auto-renders from the SSOT (`data/benchmark_matrix.csv`) between
`<!-- render:model:<name> -->` markers; table numbers are never hand-typed.*
