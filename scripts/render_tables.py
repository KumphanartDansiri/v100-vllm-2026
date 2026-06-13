#!/usr/bin/env python3
"""Render markdown tables from data/benchmark_matrix.csv (the SSOT), and optionally INJECT
them into chapter/model docs between markers. This is what makes "draft now, freeze numbers
last" a one-command operation: write prose with claims/ranges + an empty marker block; at the
final freeze, rebuild the CSV and run `--inject` to slot every exact number in at once.

Marker convention in any .md:
    <!-- render:overview -->
    (anything here is replaced)
    <!-- endrender -->

Usage:
  python3 scripts/render_tables.py                 # print all tables to stdout
  python3 scripts/render_tables.py overview        # print one (overview|moe_fix|model:<name>)
  python3 scripts/render_tables.py --inject         # inject into all docs/ + models/ files
  python3 scripts/render_tables.py --inject FILE    # inject into one file
"""
import csv, glob, os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "benchmark_matrix.csv")
rows = list(csv.DictReader(open(CSV)))


def md(headers, lines):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in lines]
    return "\n".join(out)


def overview():
    seen, lines = set(), []
    for r in rows:
        if r["users"] != "1" or "moe_patch" in r["config"]:
            continue
        k = (r["model"], r["variant"])
        if k in seen:
            continue
        seen.add(k)
        lines.append([r["model"], r["variant"], f"{r['params_total_b']}B/{r['params_active_b']}B",
                      f"TP{r['tp']}", r["tok_s_per_user"], r["ttft_s"], r["config"]])
    return md(["model", "prec", "total/active", "TP", "decode tok/s", "TTFT s", "config"], lines)


def moe_fix():
    lines = [[r["model"], r["config"], f"{r['users']}u", r["tok_s_per_user"],
              r["tok_s_aggregate"] or "-"]
             for r in rows
             if r["variant"] == "fp16" and ("moe_patch" in r["config"] or "pre-moe-patch" in r["config"])]
    return md(["model", "config", "users", "per-user tok/s", "aggregate tok/s"], lines)


def model(sub):
    sel = [r for r in rows if sub.lower() in r["model"].lower()]
    if not sel:
        return f"_(no rows match '{sub}' yet — pending measurement)_"
    return md(["variant", "TP", "users", "config", "per-user", "agg", "TTFT", "result_path"],
              [[r["variant"], f"TP{r['tp']}", r["users"], r["config"], r["tok_s_per_user"],
                r["tok_s_aggregate"] or "-", r["ttft_s"] or "-", r["result_path"]] for r in sel])


def resolve(cmd):
    cmd = cmd.strip()
    if cmd == "overview":
        return overview()
    if cmd == "moe_fix":
        return moe_fix()
    if cmd.startswith("model:"):
        return model(cmd.split(":", 1)[1])
    return f"_(unknown render command: {cmd})_"


MARK = re.compile(r"(<!--\s*render:(.+?)\s*-->)(.*?)(<!--\s*endrender\s*-->)", re.S)


def inject(path):
    txt = open(path).read()
    n = [0]

    def repl(m):
        n[0] += 1
        table = resolve(m.group(2))
        return f"{m.group(1)}\n{table}\n{m.group(4)}"

    new = MARK.sub(repl, txt)
    if n[0]:
        open(path, "w").write(new)
    return n[0]


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--inject":
        targets = args[1:] or glob.glob(f"{BASE}/docs/*.md") + glob.glob(f"{BASE}/models/*.md")
        total = sum(inject(p) for p in targets)
        print(f"injected {total} table block(s) across {len(targets)} file(s)")
    elif args:
        print(resolve(args[0]))
    else:
        print("## overview\n", overview(), "\n\n## moe_fix\n", moe_fix())
