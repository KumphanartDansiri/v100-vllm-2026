# gemma-4-31B-it

> STUB — fill from `python3 scripts/render_tables.py model:gemma-4-31B-it` once the feasible-TP ×
> concurrency sweep lands. Follow docs/06_model_results_template.md.

- What fits / feasible TP: TBD
- Best TP / flags: TBD
- FP16 baseline / FP8 result: see matrix rows
- single / multi-user tok/s: see matrix rows
- caveats: TBD


<!-- render:model:gemma-4-31B-it -->
| variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|
| fp16 | TP4 | 1 | stock-vllm | 17.61 | - | 0.16 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp8 | TP4 | 1 | fp8-plugin+coalesced | 17.53 | - | 0.45 | results/ch1_20260611/ch1.1_021/manifest.csv |
<!-- endrender -->
