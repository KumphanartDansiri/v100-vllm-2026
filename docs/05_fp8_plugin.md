# Chapter 05 — The FP8 plugin

> **Status: DRAFT** — numbers provisional until the final freeze rerun ([docs/FINAL_RERUN.md](FINAL_RERUN.md)). Tables auto-render from `data/benchmark_matrix.csv`.


> STUB — backed by data in `data/benchmark_matrix.csv`; prose to be filled before posting.

Custom W8A16 sm_70 kernels (FP8 weights resident in HBM, in-kernel dequant). Where resident FP8 wins (large MoE that wouldn't fit / bandwidth-bound decode) and where it loses (dense, compute-bound).

<!-- render: relevant pivot of scripts/render_tables.py -->