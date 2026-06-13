# Qwen3.6-35B-A3B

> STUB — fill from `python3 scripts/render_tables.py model:Qwen3.6-35B-A3B` once the feasible-TP ×
> concurrency sweep lands. Follow docs/06_model_results_template.md.

- What fits / feasible TP: TBD
- Best TP / flags: TBD
- FP16 baseline / FP8 result: see matrix rows
- single / multi-user tok/s: see matrix rows
- caveats: TBD


<!-- render:model:Qwen3.6-35B-A3B -->
| variant | TP | users | config | per-user | agg | TTFT | result_path |
|---|---|---|---|---|---|---|---|
| fp16 | TP4 | 1 | stock(pre-moe-patch) | 15.44 | - | 0.74 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp8 | TP4 | 1 | fp8-plugin+coalesced | 67.6 | - | 1.86 | results/ch1_20260611/ch1.1_021/manifest.csv |
| fp16 | TP4 | 1 | stock(pre-moe-patch) | 15.56 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| fp16 | TP4 | 8 | stock(pre-moe-patch) | 3.16 | 24.93 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| fp16 | TP4 | 1 | +moe_patch(heuristic) | 65.91 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| fp16 | TP4 | 8 | +moe_patch(heuristic) | 20.98 | 137.2 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| fp16 | TP4 | 1 | +moe_patch(tuned-json) | 65.85 | - | - | results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt |
| fp16 | TP4 | 8 | +moe_patch(tuned-json) | 22.8 | 173.92 | - | results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt |
| fp8 | TP | 1 | +mtp(k=1) | 66.38 | - | - | results/ch2_mtp_20260612/CHAIN_SUMMARY.txt |
| fp16 | TP | 1 | +mtp(k=1) | 17.2 | - | - | results/ch2_mtp_20260612/CHAIN_SUMMARY.txt |
<!-- endrender -->
