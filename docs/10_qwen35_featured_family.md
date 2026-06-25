# Chapter 10 — Qwen3.5 featured family: the full profile (27B dense + 35B-A3B MoE)

> **Status: DRAFT** — consolidates the 2026-06-24/25 exact-triad + perf_v2 + reliability runs for the two featured Qwen3.5 checkpoints. Decode/capacity detail is in [Chapter 9](09_precision_tp_comparison.md); this chapter adds prefill latency (TTFT), faithfulness, and startup to give the complete picture. Sources cited per section.

Two Qwen3.5 checkpoints are the worked examples of this report because they sit on
opposite sides of the architecture-fit line and cover the three formats a V100
operator actually ships:

- **Qwen3.5-27B** — dense (every weight active per token).
- **Qwen3.5-35B-A3B** — sparse MoE (~3B of 35B active per token).

These are the **featured pair** — each has its own model page
([Qwen3.5-27B](../models/qwen3_5_27b.md) · [Qwen3.5-35B-A3B](../models/qwen3_5_35b_a3b.md));
this chapter is the cross-cutting profile. The broader 7-model
reliability/TTFT matrix elsewhere in this report uses the **Qwen3.6** equivalents;
3.6 is architecturally the same config, so — as the cross-generation check at the
end shows — the two behave the same once tensor-parallelism is matched. Treat 3.5
here as the deep-dive, 3.6 as the wide baseline.

## 1. Decode + capacity (summary; full tables in Ch.9)

The precision × tensor-parallelism story is [Chapter 9](09_precision_tp_comparison.md).
Headline: at **TP4**, dense-FP8 beats FP16 at 1–2 users and ties at 4 (loses at 8);
**MoE-FP8 wins at every concurrency** (1.40–2.45×). At **TP2 (half-GPU)**, both
FP16 checkpoints fall out of the serving envelope while **FP8 and Int4 still fit** —
so FP8 is the faithful format that enables half-GPU deployment of the MoE.

## 2. Prefill latency (TTFT) — the FP8 cost side

Decode and memory favor FP8; prefill does **not**. TTFT, C1, TP4, vLLM 0.21:

| Model | Format | short input (~2k) | long input (~24k) | long **+FA-V100** |
|---|---|---:|---:|---:|
| 27B dense | FP16 | 0.97 s | 27.0 s | **11.2 s** |
| 27B dense | FP8 | 1.41 s | 32.2 s | **17.4 s** |
| 35B-A3B MoE | FP16 | 0.78 s | 14.2 s | **9.1 s** |
| 35B-A3B MoE | FP8 | 4.88 s | 55.1 s | 49.9 s |

Reading it:

- **FP8 prefill is slower than FP16** in every cell — modest on the dense 27B
  (~1.2× on long input), **dramatic on the MoE** (55 s vs 14 s). The cause is the
  unoptimized block-FP8 MoE prefill path (Volta WMMA dequant→FP16), not attention.
- **The FA-V100 bridge helps where attention dominates** — it roughly halves the
  dense long-input TTFT (FP8 32→17 s, FP16 27→11 s) and the MoE-FP16 (14→9 s) — but
  **barely moves the MoE-FP8** (55→50 s), because there the bottleneck is FP8 MoE
  prefill *compute*, not the attention kernel.
- So the honest trade is: **FP8 buys decode speed + half the memory at the cost of
  cold-prefill latency.** For chat/decode-heavy traffic that cost is paid once; for
  long-context, short-output (RAG, summarize) it dominates, and FP16/Int4 prefill
  faster.

*Source: `vllm-fp8-w8a16-sm70/results/perf_v2_q27b35_*_021_20260625_*`, `…q35b35_*`.*

## 3. Faithfulness — self-stability and FP8-vs-FP16

Two axes, temperature 0, the reliability harness (Q1 essay ×5 for self-stability;
Q2–Q5 factual/reasoning/structure/code for cross-precision agreement):

| Cell | Self-stability (Axis-1) | vs FP16 gold (Axis-2) |
|---|---|---|
| 27B FP16 | **Exact** (1 sha / 5) | — (gold) |
| 27B FP8 | **Exact** (1 sha / 5) | Stable (coherent; greedy 8–84%) |
| 35B-A3B FP16 | **Exact** (1 sha / 5) | — (gold) |
| 35B-A3B FP8 | **Exact** (1 sha / 5) | Stable (coherent; greedy 4–22%) |

- **All four cells are bit-deterministic run-to-run** — including the MoE-FP8.
  (The 3.6-35B-A3B-FP8 was only "Stable" self-stable; this 3.5 checkpoint is tighter.)
- **FP8-vs-FP16 is "Stable", never "Exact"** — and that is the *correct* grade: FP8
  and FP16 are different numerics, so greedy decoding picks different-but-equivalent
  tokens. Both outputs are coherent and on-topic; the low prefix-agreement on some
  prompts is synonym-level divergence, not error. (See the saved `*_q*_run*.txt`
  side-by-sides; this is the same framework as the rest of the report — byte-identity
  is the wrong bar, coherent-equivalent is the bar, and FP8 clears it.)

*Source: `/tmp/v100_ch1/manifest.csv` → `tools/ch1_report.py` (Axis-1/2).*

## 4. Startup

Cold start with `--skip-mm-profiling` (text models): **9.8–12.3 s** across all four
cells — no material difference between FP8 and FP16. (The plugin's JIT kernels are
cache-resident after first build.)

## 5. Cross-generation check — 3.5 vs 3.6 (matched TP4)

Because 3.6 shares 3.5's architecture, the featured-pair numbers should generalize.
They do — FP8 long-input TTFT, TP4, side by side:

| Model · FP8 | 3.5 | 3.6 |
|---|---:|---:|
| 27B short / long / +FA | 1.41 / 32.2 / 17.4 | 1.41 / 32.2 / 16.7 |
| 35B-A3B short / long / +FA | 4.88 / 55.1 / 49.9 | 4.78 / 53.3 / 48.2 |

Essentially identical. Two notes this resolved:

- An earlier 3.6-27B-FP8 long-TTFT of **62 s** was a **TP2** artifact (old min-TP
  rule), not a generation difference — re-run at TP4 it lands on **32 s**, matching 3.5.
- The 3.6-35B-A3B-FP8 long-TTFT had a prefix-cache contamination (a stray 14 s
  reading); the clean TP4 re-measure is **53 s**, alongside 3.5's 55 s.

*Source: `…/perf_v2_q27b_fp8_021_20260625_*`, `…/perf_v2_q35b_fp8_021_20260625_*`.*

## The complete FP8 trade, in one line

For these Qwen3.5 checkpoints on V100, block-FP8 W8A16 gives **faster decode
(low-concurrency dense, all-concurrency MoE) + half the memory (fits at half TP) +
bit-deterministic output**, at the cost of **slower cold prefill** (steep for the
MoE) and **token-level divergence from FP16** (coherent, not identical). Pick FP8
for decode-heavy serving and for fitting big MoEs on fewer cards; reach for FP16/Int4
when long-context prefill latency dominates.
