# Upstream feedback — prepared packets

The practical companion to [Chapter 8](../docs/08_acknowledgements.md). Chapter 8 is the public
*acknowledgement*; this folder is the *deliverable* — one self-contained packet per project, holding
the reproducers, patches, configs, and a draft message in a form a maintainer can pick up and use.

> **First-timer's note (see Chapter 8).** This is the author's first attempt at contributing findings
> back upstream. If a packet hasn't reached the right maintainer, or should be reshaped (issue vs PR vs
> email vs discussion), **please help route it** — corrections on the *how* are as welcome as the
> contributions themselves.

## Packets
| Project | Channel | What we're offering | Status |
|---|---|---|---|
| [flash_attention_v100/](flash_attention_v100/) | GitHub `ai-bond/flash-attention-v100` (PR/issue) · Telegram @ai_bond007 | 1 correctness fix + 1 CUDA-12.6 build fix + 1 documented strided-Q gotcha, with reproducers | Prepared, not yet sent |
| [vllm/](vllm/) | GitHub `vllm-project/vllm` (issue + data/config PR) | The FP16-MoE `BLOCK_K=128`-on-Volta diagnostic + 2 tuned V100 config JSONs | Prepared, not yet sent |
| [aphrodite/](aphrodite/) | GitHub `dphnAI/aphrodite-engine` (PR) | sm_70 build re-enable (4 commits) + the MoE heuristic & V100 configs | Prepared (parked behind the vLLM line) |
| [onecat_vllm/](onecat_vllm/) | GitHub `1CatAI/1Cat-vLLM` | Cross-checks / mutual notes — no code | Notes only |

## Conventions
- **Canonical code lives in the implementation repos, not here.** Where a fix is a real commit (e.g.
  ai-bond `ccb6557`), the packet links the repo + hash and includes a convenience `.patch`; it does not
  fork the code into this publication repo. Config data and reports — small, static, and meant to be
  shared — are copied in so the packet is self-contained.
- **Every packet README follows the same shape:** *what we observed · why it matters · environment ·
  reproducer · proposed fix / patch status · what we're asking upstream to do.*
- **Nothing here has been sent yet.** Status reflects "prepared / offered," and is updated as real
  issues/PRs/messages go out.
