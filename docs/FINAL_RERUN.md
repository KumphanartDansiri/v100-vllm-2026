# Final freeze — the one authoritative rerun

Numbers stay **provisional** (chapters carry a DRAFT banner) until a single coherent rerun at the
very end, when no experiments remain. This separates durable work (prose, structure, definitions,
the story) from volatile work (exact figures), and locks the numbers ONCE from one consistent run.

## Why one coherent sweep (not patch-by-cell)
The current CSV mixes runs from different dates (Ch1 06-11, MoE 06-12/13). For publication, the
final matrix should come from **one campaign**: same image (`vllm-v100:vllm021-cu126`), same flags,
clean idle box, all cells — so there are zero "measured under possibly-different conditions"
caveats. Internal consistency > squeezing the last %.

## Freeze procedure (one pass, mostly one command)
1. **Run the full bench** on a clean box — Ch1 reliability + MoE A/B + MTP + per-model feasible-TP ×
   concurrency. (Harnesses in the fp8-w8a16-sm70 repo: `ch1_reliability_bench.sh`,
   `moe_stages_ab_vllm021.sh`, `ch2_mtp_*`, and the TP-sweep harness for model pages.)
2. **Rebuild the SSOT:** `python3 scripts/build_matrix_from_results.py` → regenerates
   `data/benchmark_matrix.csv` from the fresh results.
3. **Re-render every table:** `python3 scripts/render_tables.py --inject` → all chapter/model
   tables update from the CSV in one shot.
4. **Sanity pass:** skim prose for any hardcoded exact figure that drifted (there should be none —
   prose carries claims/ranges, tables carry exact numbers). Fix the few that exist.
5. **Flip banners:** remove the DRAFT banner from each chapter (numbers now final).
6. **Publish:** push repo public, then post chapters on the agreed cadence.

## The discipline that keeps step 4 cheap
- **Prose** = claims + ranges robust to ±small drift ("4–9×", "beats dense", "FP8 wins on MoE").
- **Tables** = exact figures, auto-rendered from the CSV (never hand-typed in prose).
If you follow this, the final freeze is steps 2–3 (two commands) + a quick read, not a rewrite.
