# Model page template (Chapter 6+)

Each model family uses this exact schema so pages are comparable. Numbers come from the SSOT
(`scripts/render_tables.py model:<name>`); only feasible TP sizes are listed.

## <Model name>
- **What fits / feasible TP:** (min TP set by VRAM; per variant)
- **Best TP:** (throughput-optimal) · **Best serving flags:**
- **FP16/BF16 baseline:** (if it fits) · **FP8 result:** (plugin)
- **single-user tok/s** · **multi-user aggregate tok/s**
- **startup notes** · **memory notes** · **caveats**
- **raw result links** (result_path from the matrix)

<!-- render: python3 scripts/render_tables.py model:<name> -->
