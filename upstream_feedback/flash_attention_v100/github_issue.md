# Copy-paste submission — flash-attention-v100

*Paste the body into a GitHub issue on `ai-bond/flash-attention-v100` (or use it as a PR description if
opening a PR from a fork with `ccb6557`). Self-contained: the two code fixes are inlined. Tone:
grateful, practical — a user reporting findings, not a demand.*

---

**Title:** V100 paged-KV straddle fix, CUDA 12.6 build fix, and a strided-Q integration note

---

Hi — first, thank you for this kernel. We used flash-attention-v100 as the **prefill backend for vLLM
serving on 8×V100-32GB** (paged KV via the low-level `varlen_fwd`), and it's the engine behind every
prefill/TTFT win in our write-up — including getting an **MLA model (GLM-4.7-Flash) to generate on
Volta at all**. Headline: GLM-4.5-Air TTFT@24k went **51.8 s → 19.45 s (2.66×)**, and we measured
~8× your kernel vs vLLM's Triton attention at 26k. Validated end-to-end across 8 models (dense / MoE /
GDN-hybrid, D=128 & 256).

Wiring it into a real serving stack surfaced three things — one correctness bug, one build fix, and one
integration note. Reproducers for each, and the two code fixes are tiny (3 files, 4 lines; committed in
our fork as `ccb6557`, happy to open a PR instead if you prefer).

**1. Paged KV: `BLOCK_N` must divide the page (D=128 tile straddles pages) — correctness.**
With paged KV (`page_block_size % 256 == 0`), the varlen kernel resolves one physical page per KV tile
and loads `valid_kv_rows` linearly. `BLOCK_N_128 = 160` makes a tile starting at row 160 span 160–319,
crossing the 256-row page boundary — with a non-contiguous block table (the normal vLLM case) it reads
another sequence's KV. The real invariant is `page_block_size % BLOCK_N == 0` (D=32/64/256 satisfy it;
D=128 at 160 and D=16 at 512 don't).
```diff
--- a/include/forward.h
+++ b/include/forward.h
-#define BLOCK_N_128 160
+#define BLOCK_N_128 128
```
Repro: paged varlen, D=128, two sequences with interleaved physical blocks (seq0→[0,2], seq1→[1,3]),
sentinel `100.0` in unused slots → output contains the sentinel (`cos ≈ 0.004`); after the fix, exact
(`cos = 1.000000`) up to 24k tokens with shuffled tables. Perf cost ≈ nil (112.6 → 109.9 ms/layer @26k).

**2. CUDA 12.6 support: `__tanhf` is 12.8+/12.9-only; the pin can be 12.6 — portability.**
`setup.py` requires `torch.version.cuda >= 12.9`, but the actual blocker is `__tanhf` in the softcap
path (host-only before ~12.8, so device compile fails on nvcc 12.6). Plain `tanhf` is valid on every
CUDA and is the softcap-only path. This builds + passes tests on **torch 2.11+cu126 / nvcc 12.6** — the
*other* Volta-supporting torch island besides 2.10+cu129 (cu128/cu129 wheels dropped Volta in 2.11; cu126
kept it). Perf equal across both.
```diff
--- a/include/mat_mul.h        (2 sites in the softcap path)
-   ... __tanhf(x) ...
+   ... tanhf(x) ...
--- a/setup.py
-   torch.version.cuda >= "12.9"
+   torch.version.cuda >= "12.6"
```
(Or guard the swap with `#if CUDART_VERSION >= 12080` if you'd rather keep `__tanhf` on newer toolchains.)

**3. Low-level `varlen_fwd` silently misreads strided Q (dense-row assumption) — integration note.**
The low-level entry assumes dense `[T, H_Q*D]` query rows (host checks only `stride(-1)==1`; the kernel
indexes `q_base + row*H_Q*D`). A fused-QKV `.split()` view (row stride = full qkv width) is silently
wrong at full speed (`cos ≈ 0.5` at GQA 12/1). The python wrapper hides this via `.contiguous()`, but
direct low-level callers (any serving integration) hit it. We densify Q in our adapter (~0.2 ms @24k).
Not patched — your call on the preferred shape: a loud doc note at the binding, a host-side stride check
(`q.stride(0) == H_Q*D`), or threading q strides through to the kernel like upstream FA2.

We're new to contributing upstream, so if any of this is better as a PR, separate issues, or a different
format, just say and we'll reshape it. Thanks again for the kernel — happy to share the full reproducer
scripts and the 8-model validation matrix.
