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


def md(headers, lines, align=None):
    # align: optional per-column 'l'|'r'|'c' list; default (None) = all left, as before.
    seps = ["|"]
    for i in range(len(headers)):
        a = align[i] if align and i < len(align) else "l"
        seps.append({"r": "---:", "c": ":---:"}.get(a, "---") + "|")
    out = ["| " + " | ".join(headers) + " |", "".join(seps)]
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
    # One row per (model, users) x {Stock, MoE patch}, so each pair is a direct same-model/same-users
    # A/B. "MoE patch" = the tuned-json config (Ch2's headline uses the best patched result); the
    # default-on heuristic rows stay in the CSV but are omitted here (the prose covers the nuance).
    STOCK, TUNED = "stock(pre-moe-patch)", "+moe_patch(tuned-json)"
    cells, order = {}, []
    for r in rows:
        if "Ch2 MoE fix" not in r["notes"] or r["config"] not in (STOCK, TUNED):
            continue
        key = (r["model"], r["users"])
        if key not in cells:
            cells[key] = {}
            order.append(key)
        cells[key][r["config"]] = (r["tok_s_per_user"], r["tok_s_aggregate"] or "-")

    def ratio(num, den, tag=""):
        try:
            return f"{float(num) / float(den):.1f}x{tag}"
        except (ValueError, ZeroDivisionError):
            return "-"

    lines = []
    for (mdl, users) in order:
        c = cells[(mdl, users)]
        sp = c.get(STOCK, ("-", "-"))
        tp = c.get(TUNED, ("-", "-"))
        imp = ratio(tp[1], sp[1], " agg") if users == "8" else ratio(tp[0], sp[0])
        lines.append([mdl, users, "Stock", sp[0], sp[1], "baseline"])
        lines.append([mdl, users, "MoE patch", tp[0], tp[1], imp])
    # rowspan look: blank the repeated Users within each (model, users) pair, then the repeated Model
    prev = None
    for ln in lines:
        k = (ln[0], ln[1])
        if k == prev:
            ln[1] = ""
        else:
            prev = k
    _blank_repeat(lines, 0)
    # Two-line header (Markdown-native via <br>): label on line 1, unit "(tok/s)" on line 2.
    return md(["Model", "Users", "Config", "Per-user<br>(tok/s)", "Aggregate<br>(tok/s)",
               "Improvement"], lines, align=["l", "r", "l", "r", "r", "r"])


def model(sub):
    # Model pages = normal serving configs only. MTP is a separate execution mode (different tok/s
    # semantics — acceptance/exactness/k) and lives in Chapter 4, so exclude +mtp rows here.
    sel = [r for r in rows if sub.lower() in r["model"].lower() and not r["config"].startswith("+mtp")]
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
    # One row per (model, k); Model label folds in Format + Dense/MoE (Qwen3.6-35B-A3B appears as both
    # FP8 and FP16 — the contrast is the point). Blank-repeat Model (Chapter-2 rowspan look).
    n_of = lambda n, key: (re.search(rf"{key}=(\S+?)[;\s]", n + " ") or [None, "-"])[1]
    ex = lambda v: {"EXACT": "Exact", "DIFF": "Diff"}.get(v, v)
    typ = lambda r: "dense" if r["params_total_b"] == r["params_active_b"] else "MoE"
    groups, meta = {}, {}
    for r in rows:
        if not r["config"].startswith("+mtp"):
            continue
        label = f"{short(r['model'])} ({fmt(r['variant'])}, {typ(r)})"
        n = r["notes"]
        line = [label, int(r["mtp"]), n_of(n, "off"), r["tok_s_per_user"], n_of(n, "speedup"),
                n_of(n, "accept"), ex(n_of(n, "exactness"))]
        groups.setdefault(label, []).append((int(r["mtp"]), line))
        meta[label] = (float(r["params_total_b"]), 0 if r["variant"] == "fp8" else 1)  # 27<35<122; fp8 before fp16
    lines = []
    for label in sorted(groups, key=lambda L: meta[L]):
        for _, line in sorted(groups[label], key=lambda kl: kl[0]):
            lines.append(line)
    _blank_repeat(lines, 0)
    return md(["Model", "k", "off tok/s", "MTP tok/s", "speedup", "accept", "exactness"],
              lines, align=["l", "r", "r", "r", "r", "r", "l"])


def eager_cudagraph():
    p = os.path.join(BASE, "data", "eager_vs_cudagraph.csv")
    if not os.path.exists(p):
        return "_(pending the paired eager-vs-cudagraph run)_"
    rr = list(csv.DictReader(open(p)))
    # show only measured pairs (unmeasured families are omitted, not listed as "pending")
    lines = [[r["model"], r["eager_tok_s"], r["cudagraph_tok_s"], r["improvement"]]
             for r in rr if r["improvement"] != "pending" and r["eager_tok_s"] != "pending"]
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


# ---- Chapter-6 digest views: curated, FROZEN (perf_v2-only) render views per model family ------
# A digest NEVER selects legacy Ch1/TP-sweep rows — only implementation_ref == fp8-v100-2026-matrix.
# One shared per-family config list drives both: single_user pivots it to columns, concurrency to rows.
PV2 = "fp8-v100-2026-matrix"
DIGEST_SPECS = {
    "qwen3_6_27b": {
        "match": "Qwen3.6-27B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16 TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4"),
                    ("FP8 TP2", "fp8", "2", "single_user")],   # half-GPU = a fit option, not scaling
    },
    "qwen3_6_35b_a3b": {
        "match": "Qwen3.6-35B-A3B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16 TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4"),
                    ("FP8 TP2", "fp8", "2", "single_user")],   # half-GPU = a fit option, not scaling
    },
    "qwen3_5_122b_a10b": {
        "match": "Qwen3.5-122B-A10B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP8 TP8", "fp8", "8"), ("GPTQ-Int4 TP8", "GPTQ-Int4", "8")],
    },
    "gemma4_31b": {
        "match": "gemma-4-31B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16 TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4"),
                    ("FP8 TP2", "fp8", "2", "single_user")],
    },
    "gemma4_26b_a4b": {
        "match": "gemma-4-26B-A4B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16 TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4"),
                    ("FP8 TP2", "fp8", "2", "single_user")],
    },
    "glm4_5_air": {
        "match": "GLM-4.5-Air",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP8 TP8", "fp8", "8")],
    },
    "glm4_7_flash": {
        "match": "GLM-4.7-Flash",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("BF16 TP4", "bf16", "4")],
    },
}


def _digest_row(match, eng, variant, tp, users):
    for r in rows:
        if (r["implementation_ref"] == PV2 and match in r["model"] and r["vllm_version"] == eng
                and r["variant"] == variant and r["tp"] == tp and r["users"] == users):
            return r
    return None


def _eng(v):
    return v[:-2] if v.endswith(".0") else v   # 0.19.0 -> 0.19


def _cfg(c):                           # (label, variant, tp, [flag]) -> (label, variant, tp, su_only)
    return c[0], c[1], c[2], (len(c) > 3 and c[3] == "single_user")


TTFT_NOTE = (
    "\n\n¹ **Warm TTFT** = warm / prefix-cache-hit / chunked-prefill serving latency — **pending SSOT "
    "refresh**. **Cold TTFT** is cold *monolithic* prefill from the representative SSOT row: a "
    "**worst-case** number, *not* warm serving latency — don't read it as steady interactive response.")


def single_user(key):                  # C1 deployment summary: config rows x {per-engine decode, TTFT}
    s = DIGEST_SPECS[key]
    cfgs = [_cfg(c) for c in s["configs"]]
    engs = s["engines"]
    tt = "0.21.0" if "0.21.0" in engs else engs[-1]   # representative engine for the (cold) TTFT cell
    head = (["Choice"] + [f"{_eng(e)} C1 decode" for e in engs]
            + [f"{_eng(tt)} Cold TTFT", f"{_eng(tt)} Warm TTFT¹"])
    lines = []
    for lbl, var, tp, _ in cfgs:
        row = [lbl]
        for e in engs:
            r = _digest_row(s["match"], e, var, tp, "1")
            row.append(f"{r['tok_s_per_user']} tok/s" if r else "—")
        rt = _digest_row(s["match"], tt, var, tp, "1")
        row.append(f"{rt['ttft_s']} s" if (rt and rt["ttft_s"]) else "—")
        row.append("pending")
        lines.append(row)
    return md(head, lines, align=["l"] + ["r"] * (len(engs) + 2)) + TTFT_NOTE


def concurrency(key):                  # same-TP scaling: per-config per-user/aggregate x C1/C2/C4/C8
    s = DIGEST_SPECS[key]
    lines = []
    for eng in s["engines"]:
        for lbl, var, tp, su_only in (_cfg(c) for c in s["configs"]):
            if su_only:                # half-GPU / fit options belong in the single-user table, not here
                continue
            pu = [f"{_eng(eng)} {lbl}", "per-user"]
            ag = ["", "aggregate"]
            seen = False
            for u in ("1", "2", "4", "8"):
                r = _digest_row(s["match"], eng, var, tp, u)
                pu.append(r["tok_s_per_user"] if r else "—")
                ag.append((r["tok_s_aggregate"] or "—") if r else "—")
                seen = seen or bool(r)
            if seen:                   # skip configs not measured on this engine
                lines += [pu, ag]
    return md(["Config", "Type", "C1", "C2", "C4", "C8"], lines,
              align=["l", "l", "r", "r", "r", "r"])


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
    if cmd.startswith("single_user:"):
        return single_user(cmd.split(":", 1)[1])
    if cmd.startswith("concurrency:"):
        return concurrency(cmd.split(":", 1)[1])
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
