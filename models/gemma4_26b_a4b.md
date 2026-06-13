# gemma-4-26B-A4B-it

> STUB — fill from `python3 scripts/render_tables.py model:gemma-4-26B-A4B-it` once the feasible-TP ×
> concurrency sweep lands. Follow docs/06_model_results_template.md.

- What fits / feasible TP: TBD
- Best TP / flags: TBD
- FP16 baseline / FP8 result: see matrix rows
- single / multi-user tok/s: see matrix rows
- caveats: TBD


<!-- render:model:gemma-4-26B-A4B-it -->
| variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|
| fp16 | TP4 | 1 | stock(pre-moe-patch) | 10.2 | - | 0.3 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp8 | TP4 | 1 | fp8-plugin+coalesced | 46.48 | - | 0.18 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp16 | TP4 | 1 | stock(pre-moe-patch) | 10.91 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| fp16 | TP4 | 8 | stock(pre-moe-patch) | 3.58 | 28.3 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| fp16 | TP4 | 1 | +moe_patch(heuristic) | 43.66 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| fp16 | TP4 | 8 | +moe_patch(heuristic) | 19.1 | 145.15 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
| fp16 | TP4 | 1 | +moe_patch(tuned-json) | 43.71 | - | - | results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt |
| fp16 | TP4 | 8 | +moe_patch(tuned-json) | 20.23 | 155.94 | - | results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt |
<!-- endrender -->
