# Chapter 8 — Acknowledgements & upstream feedback

> **Status: Final** — credits and contribution status; no benchmark figures here (those live in
> Chapters 1–6, rendered from the SSOT).

Almost nothing here is ours alone. The V100 "comes back to life" story is really a story of *other
people's* engines, kernels, and checkpoints — re-pointed at sm\_70 and measured honestly. This chapter
names what we built on and, just as importantly, **what we send back**: an open-source result that
takes a fix or a kernel and never reports the findings upstream is only half-finished. Each project
page below records, in the same place, *what it gave us* and *what we're contributing back to it*.

The discipline is the one from Chapter 7: a fix is only worth sending once it's measured. Every
contribution below carries a reproducer or a number, not just a claim.

## A note on contributing back

I have used open source for years as a consumer and operator; this is my first serious attempt to
contribute findings back upstream. I may not always know the right channel, format, or maintainer path
for each project. The intent is gratitude and useful feedback, not drive-by criticism.

If any of these notes fail to reach the right maintainer — or should be reshaped as an issue, PR,
email, or discussion thread — **please help route them.** The goal is simple: give the relevant
projects the reproducers, patches, and measurements in a form they can actually use. The prepared
materials are gathered in [`upstream_feedback/`](../upstream_feedback/) so anyone can see exactly what
is being offered, and to whom.

## Foundations (used as published, unchanged)
We changed nothing in these; they are the ground the rest stands on, and we're grateful for them:
- **PyTorch** and **Triton** — the tensor runtime and the kernel language the FP8/MoE work is written in.
- **NVIDIA CUTLASS** — its memory-efficient attention is the SDPA baseline that, on V100's short-sequence
  regime, our own FlashAttention experiment *couldn't beat* (Chapter 7, dead end #4). A baseline worth
  losing to.
- **Hugging Face `transformers`** — model definitions; the `transformers-5` line is what unlocks
  Gemma-4 / GLM-4.7 on the 0.19 image.
- **Model & checkpoint providers** — Alibaba **Qwen**, Google **Gemma**, Z.ai **GLM**, the **RedHatAI**
  compressed-tensors FP8 checkpoints, and the **GPTQ** authors whose Int4 is our honest comparison
  baseline at 122B.

## Project pages (what we built on · what we send back)
- [flash-attention-v100 (ai-bond)](../acknowledgements/flash_attention_v100.md) — the V100 FlashAttention
  kernels behind every prefill/TTFT win and the first MLA model on Volta. We send back **3 fixes**
  (paged-KV straddle, CUDA-12.6 build, strided-Q) with reproducers.
- [vLLM](../acknowledgements/vllm.md) — both engines (0.19 + 0.21); the whole thing runs on its source.
  We send back a **diagnostic + V100 MoE config data** (the `BLOCK_K=128` finding, Chapter 2).
- [aphrodite-engine](../acknowledgements/aphrodite.md) — the "upstream-and-ride" alternative. We have a
  clean **sm\_70 build rewire** and a **MoE-heuristic PR** prepared for it.
- [1Cat-vLLM (1CatAI)](../acknowledgements/onecat_vllm.md) — an independent V100/sm\_70 vLLM effort run
  in parallel with ours. No code imported, but real lessons exchanged — credited here in full.

## Feedback we're contributing back — at a glance
| To | What | Form | State |
|---|---|---|---|
| **ai-bond / flash-attention-v100** | Paged-KV straddle fix, CUDA-12.6 build (`__tanhf`→`tanhf` + torch pin), strided-Q note | Source patches + a written report with reproducers | Prepared; offered to the maintainer |
| **vLLM** | The FP16-MoE `BLOCK_K=128`-on-Volta root cause (4–9× cost) + two autotuned V100 config JSONs | Diagnostic issue + data-only config contribution | Prepared |
| **aphrodite-engine** | sm\_70 build re-enable (arch list / EXL3 / Marlin guard) + the MoE heuristic & V100 configs | A small commit series + a PR | Prepared (parked behind the vLLM line) |
| **1Cat-vLLM** | Cross-checks on the shared MTP CUDA-graph hazard and FA integration shape | Notes / mutual acknowledgement | Documented here |

Each row's prepared packet — reproducers, patches, configs, and a draft message — lives under
[`upstream_feedback/`](../upstream_feedback/), one self-contained folder per project.

## Licensing & reuse
Each dependency keeps its own license — see its project page and upstream repository. The
FlashAttention-V100 fork is **BSD-3-Clause** (© D. Skryabin); vLLM is **Apache-2.0**. Our own additions
(the FP8 W8A16 sm\_70 kernels and the MoE fix) live in the companion **fp8-w8a16-sm70** repo and are
offered in the same spirit: take them, measure them, report back.

*If you maintain one of these projects and want a contribution shaped differently — a PR instead of a
report, or vice-versa — that's welcome; each project's packet under
[`upstream_feedback/`](../upstream_feedback/) opens with a five-minute "maintainer quick path."*
