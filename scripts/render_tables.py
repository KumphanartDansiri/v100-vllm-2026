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
    def row(r):
        return [r["model"], r["variant"], f"{r['params_total_b']}B/{r['params_active_b']}B",
                f"TP{r['tp']}", r["tok_s_per_user"], r["ttft_s"] or "-", r["config"]]
    lines, seen = [], set()
    # baseline: Ch1 single-user reliability rows (one per checkpoint), or a later revisit's C1 row
    for r in rows:
        if r["users"] == "1" and ("Ch1 reliability" in r["notes"] or "revisit" in r["notes"]):
            k = (r["model"], r["variant"])
            if k not in seen:
                seen.add(k); lines.append(row(r))
    # + the patched FP16-MoE single-user rows, so the MoE models show non-patch vs patch
    for r in rows:
        if r["users"] == "1" and r["config"] == "+moe_patch(tuned-json)":
            lines.append(row(r))
    lines.sort(key=lambda x: (x[0], 0 if "stock" in x[6] else 1))  # group by model; stock (non-patch) before +moe_patch
    return md(["model", "prec", "total/active", "TP", "decode tok/s", "TTFT s", "config"], lines)


def moe_fix():
    # Only the Ch2 A/B rows (base/kbest/auto, single+8u) — excludes the Ch1 stock duplicate.
    lines = [[r["model"], r["config"], f"{r['users']}u", r["tok_s_per_user"],
              r["tok_s_aggregate"] or "-"]
             for r in rows if "Ch2 MoE fix" in r["notes"]]
    return md(["model", "config", "users", "per-user tok/s", "aggregate tok/s"], lines)


def model(sub):
    sel = [r for r in rows if sub.lower() in r["model"].lower()]
    if not sel:
        return f"_(no rows match '{sub}' yet — pending measurement)_"
    # show the engine column when a model was measured on >1 vLLM/stack
    eng = lambda r: f"{r['vllm_version']}/{r['torch_cuda']}"
    multi = len({eng(r) for r in sel}) > 1
    head = (["vLLM"] if multi else []) + ["variant", "TP", "users", "config",
            "per-user", "agg", "TTFT", "result_path"]
    return md(head,
              [([eng(r)] if multi else []) +
               [r["variant"], f"TP{r['tp']}", r["users"], r["config"], r["tok_s_per_user"],
                r["tok_s_aggregate"] or "-", r["ttft_s"] or "-", r["result_path"]] for r in sel])


def mtp():
    lines = []
    for r in rows:
        if not r["config"].startswith("+mtp"):
            continue
        n = r["notes"]
        g = lambda k: (re.search(rf"{k}=(\S+?)[;\s]", n + " ") or [None, "-"])[1]
        lines.append([r["model"], r["variant"], r["config"].replace("+mtp", "").strip("()"),
                      g("off"), r["tok_s_per_user"], g("speedup"), g("accept"), g("exactness")])
    return md(["model", "prec", "k", "off tok/s", "mtp tok/s", "speedup", "accept", "exactness"], lines)


def eager_cudagraph():
    p = os.path.join(BASE, "data", "eager_vs_cudagraph.csv")
    if not os.path.exists(p):
        return "_(pending the paired eager-vs-cudagraph run)_"
    rr = list(csv.DictReader(open(p)))
    lines = [[r["model"], r["eager_tok_s"], r["cudagraph_tok_s"], r["improvement"]] for r in rr]
    return md(["model", "eager tok/s", "cudagraph tok/s", "improvement"], lines)


# ---- Chapter 1 dual-engine tables (perf_v2 rows = implementation_ref set) -------------------
def _pv2(r):
    return bool(r["implementation_ref"])


def short(name):                       # official checkpoint -> family short name
    s = name.split("/")[-1]
    for suf in ("-FP8-Dynamic", "-GPTQ-Int4", "-FP8", "-it"):
        s = s.replace(suf, "")
    return s


def fmt(v):
    return {"fp16": "FP16", "fp8": "FP8", "bf16": "BF16"}.get(v, v)


def _blank_repeat(lines, col=0):       # blank a column when equal to the previous row (rowspan look)
    prev = None
    for ln in lines:
        if ln[col] == prev:
            ln[col] = ""
        else:
            prev = ln[col]
    return lines


def models_tested():                   # Table 1: identity primer (short | type | format | checkpoint)
    seen, lines = set(), []
    for r in rows:
        if not _pv2(r):
            continue
        k = (r["model"], r["variant"])
        if k in seen:
            continue
        seen.add(k)
        lines.append([short(r["model"]), r["model_type"], fmt(r["variant"]), f"`{r['model']}`"])
    lines.sort(key=lambda x: (x[0], x[2]))
    prev = None                        # blank short-name AND type within a family
    for ln in lines:
        if ln[0] == prev:
            ln[1] = ln[0] = ""
        else:
            prev = ln[0]
    return md(["Short name", "Model type", "Format", "Official checkpoint"], lines)


def baseline():                        # Table 2: single-user, same-TP, dual-engine C1
    pick = {}                          # (model,variant) -> {ver: row}; prefer same-TP, drop half-GPU
    for r in rows:
        if not _pv2(r) or r["users"] != "1" or "TP2 half-GPU" in r["notes"]:
            continue
        d = pick.setdefault((r["model"], r["variant"]), {})
        cur = d.get(r["vllm_version"])
        if cur is None or ("same-TP" in r["notes"] and "same-TP" not in cur["notes"]):
            d[r["vllm_version"]] = r
    lines = []
    for (mdl, var), d in sorted(pick.items(), key=lambda kv: (short(kv[0][0]), kv[0][1])):
        c19 = d.get("0.19.0", {}).get("tok_s_per_user", "—")
        c21 = d.get("0.21.0", {}).get("tok_s_per_user", "—")
        note = []
        if short(mdl) == "GLM-4.7-Flash":
            note.append("MLA path")
        if c19 == "—" and "gemma" in mdl.lower():
            note.append("0.19 n/a (gemma4.py)")
        elif "0.19.0" in d and "tf5" in d["0.19.0"]["flags"]:
            note.append("tf5 on 0.19")
        lines.append([short(mdl), fmt(var), c19, c21, "; ".join(note)])
    _blank_repeat(lines, 0)
    return md(["Model", "Format", "vLLM 0.19 C1", "vLLM 0.21 C1", "Notes"], lines)


def engine_matrix(ver):                # Tables 3/4: full per-engine decode + cold TTFT
    cells, order = {}, []
    for r in rows:
        if not _pv2(r) or r["vllm_version"] != ver:
            continue
        key = (r["model"], r["variant"], r["tp"], r["max_model_len"], r["notes"])
        c = cells.get(key)
        if c is None:
            c = cells[key] = {"u": {}, "agg8": "", "ttft": ""}
            order.append(key)
        c["u"][r["users"]] = r["tok_s_per_user"]
        if r["users"] == "8":
            c["agg8"] = r["tok_s_aggregate"]
        if r["users"] == "1":
            c["ttft"] = r["ttft_s"]
    lines = []
    for (mdl, var, tp, mlen, notes) in order:
        c = cells[(mdl, var, tp, mlen, notes)]
        u = c["u"]
        tag = "same-TP" if "same-TP" in notes else ("½-GPU (short ctx)" if "TP2 half-GPU" in notes else "")
        lines.append([short(mdl), fmt(var), f"TP{tp}", mlen,
                      u.get("1", "—"), u.get("2", "—"), u.get("4", "—"), u.get("8", "—"),
                      c["agg8"] or "—", c["ttft"] or "—", tag])
    lines.sort(key=lambda x: (x[0], x[1], x[2]))
    _blank_repeat(lines, 0)
    return md(["Model", "Format", "TP", "Max len", "C1", "C2", "C4", "C8/u", "C8 agg",
               "Cold TTFT", "Notes"], lines)


def resolve(cmd):
    cmd = cmd.strip()
    if cmd == "overview":
        return overview()
    if cmd == "moe_fix":
        return moe_fix()
    if cmd == "mtp":
        return mtp()
    if cmd == "eager_cudagraph":
        return eager_cudagraph()
    if cmd == "models_tested":
        return models_tested()
    if cmd == "baseline":
        return baseline()
    if cmd == "matrix19":
        return engine_matrix("0.19.0")
    if cmd == "matrix21":
        return engine_matrix("0.21.0")
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
