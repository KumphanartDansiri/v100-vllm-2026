#!/usr/bin/env python3
"""Render markdown tables from data/benchmark_matrix.csv (the SSOT).
Every published table comes from here so numbers never diverge across chapters.

Usage: python3 scripts/render_tables.py            # prints all tables
       python3 scripts/render_tables.py overview   # one table by name
Tables: overview (Ch1), moe_fix (Ch2), model:<NAME substring> (per-model page).
"""
import csv, os, sys

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "benchmark_matrix.csv")
rows = list(csv.DictReader(open(CSV)))


def md(headers, lines):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in lines]
    return "\n".join(out)


def overview():
    # Ch1: single-user reliability/baseline, one row per model+variant (stock config)
    seen, lines = set(), []
    for r in rows:
        if r["users"] != "1":
            continue
        k = (r["model"], r["variant"])
        if k in seen or "moe_patch" in r["config"]:
            continue
        seen.add(k)
        lines.append([r["model"], r["variant"], f"{r['params_total_b']}B/{r['params_active_b']}B",
                      f"TP{r['tp']}", r["tok_s_per_user"], r["ttft_s"], r["config"]])
    return "### Ch1 — model support & single-user baseline (vLLM 0.21+cu126, cudagraph)\n\n" + \
        md(["model", "prec", "total/active", "TP", "decode tok/s", "TTFT s", "config"], lines)


def moe_fix():
    # Ch2: FP16 MoE stock vs patched, single + 8-user
    lines = []
    for r in rows:
        if "moe_patch" not in r["config"] and "pre-moe-patch" not in r["config"]:
            continue
        if r["variant"] != "fp16":
            continue
        lines.append([r["model"], r["config"], f"{r['users']}u",
                      r["tok_s_per_user"], r["tok_s_aggregate"] or "-"])
    return "### Ch2 — FP16 MoE: stock vs Volta fix (TP4, cudagraph)\n\n" + \
        md(["model", "config", "users", "per-user tok/s", "aggregate tok/s"], lines)


def model(sub):
    lines = [r for r in rows if sub.lower() in r["model"].lower()]
    if not lines:
        return f"(no rows match '{sub}')"
    out = [f"### {lines[0]['model']} — all measured cells\n"]
    out.append(md(["variant", "TP", "users", "config", "per-user", "agg", "TTFT", "result_path"],
                  [[r["variant"], f"TP{r['tp']}", r["users"], r["config"],
                    r["tok_s_per_user"], r["tok_s_aggregate"] or "-", r["ttft_s"] or "-",
                    r["result_path"]] for r in lines]))
    return "\n".join(out)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "overview":
        print(overview())
    elif arg == "moe_fix":
        print(moe_fix())
    elif arg.startswith("model:"):
        print(model(arg.split(":", 1)[1]))
    else:
        print(overview(), "\n\n", moe_fix())
