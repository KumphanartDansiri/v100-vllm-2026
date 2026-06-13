# Qwen3.6-27B

> STUB — fill from `python3 scripts/render_tables.py model:Qwen3.6-27B` once the feasible-TP ×
> concurrency sweep lands. Follow docs/06_model_results_template.md.

- What fits / feasible TP: TBD
- Best TP / flags: TBD
- FP16 baseline / FP8 result: see matrix rows
- single / multi-user tok/s: see matrix rows
- caveats: TBD


<!-- render:model:Qwen3.6-27B -->
| variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|
| fp16 | TP4 | 1 | stock | 37.4 | - | 0.26 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp8 | TP4 | 1 | stock | 35.05 | - | 1.21 | results/ch1_20260611/ch1.1_021/manifest.csv |
<!-- endrender -->
