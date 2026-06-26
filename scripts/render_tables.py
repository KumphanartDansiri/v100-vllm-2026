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
  python3 scripts/render_tables.py overview        # print one (overview|moe_fix|model:<ModelName>, e.g. model:Qwen3.5-27B)
  #   digest views take the page KEY, not the model name: single_user|ttft|concurrency:<key>  (e.g. single_user:qwen3_5_27b)
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
        return [r["model"], fmt(r["variant"]), f"{r['params_total_b']}B/{r['params_active_b']}B",
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
    return md(["Model", "Format", "Total/Active", "TP", "Decode tok/s", "TTFT (s)", "Config"], lines)


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
    head = (["vLLM"] if multi else []) + ["Variant", "TP", "Users", "Config",
            "Per-user", "Aggregate", "Cold TTFT", "FA Cold", "Prefix Hit", "Result path"]
    return md(head,
              [([eng(r)] if multi else []) +
               [fmt(r["variant"]), f"TP{r['tp']}", r["users"], r["config"], r["tok_s_per_user"],
                r["tok_s_aggregate"] or "-", r["ttft_s"] or "-", r.get("ttft_fa_cold_s") or "-",
                r.get("ttft_prefix_hit_s") or "-", r["result_path"]] for r in sel])


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
    return md(["Model", "k", "Base tok/s", "MTP tok/s", "Speedup", "Accept", "Exactness"],
              lines, align=["l", "r", "r", "r", "r", "r", "l"])


def eager_cudagraph():
    p = os.path.join(BASE, "data", "eager_vs_cudagraph.csv")
    if not os.path.exists(p):
        return "_(pending the paired eager-vs-cudagraph run)_"
    rr = list(csv.DictReader(open(p)))
    # show only measured pairs (unmeasured families are omitted, not listed as "pending")
    lines = [[r["model"], r["eager_tok_s"], r["cudagraph_tok_s"], r["improvement"]]
             for r in rr if r["improvement"] != "pending" and r["eager_tok_s"] != "pending"]
    return md(["Model", "Eager tok/s", "Cudagraph tok/s", "Improvement"], lines)


# ---- Chapter 1 dual-engine tables (perf_v2 rows = implementation_ref set) -------------------
def _pv2(r):
    # Ch1 dual-engine matrix = the broad 32768-ctx perf_v2 rows. Exclude the Qwen3.5
    # short-context (4096) exact-triad study rows — they live in their own model pages + Ch9/10.
    return bool(r["implementation_ref"]) and r["implementation_ref"] != "qwen35-exact-triad"


def short(name):                       # official checkpoint -> family short name
    s = name.split("/")[-1]
    for suf in ("-FP8-Dynamic", "-GPTQ-Int4", "-FP8", "-it"):
        s = s.replace(suf, "")
    return s


def fmt(v):       # PERFORMANCE/runtime label: V100 (sm_70) has no usable BF16 path, so every
    # full-precision checkpoint is served `--dtype float16` — labelled FP16* (* -> the footnote below).
    return {"fp16": "FP16*", "bf16": "FP16*", "fp8": "FP8"}.get(v, v)


def fmt_src(v):   # IDENTITY/source label: the full-precision base checkpoints all ship in BF16
    # (Qwen, Gemma, GLM). Used only by model-info tables (no perf numbers) — see methodology.
    return {"fp16": "BF16", "bf16": "BF16", "fp8": "FP8"}.get(v, v)


# Footnote auto-appended under any PERFORMANCE table that carries an FP16* label (full-precision
# rows = BF16 checkpoints executed as FP16 on V100). Skipped on FP8/Int4-only tables (no FP16*).
FP16_NOTE = ("\n\n_\\*BF16 checkpoint, served as FP16 on V100 (sm_70 has no native BF16; "
             "`--dtype float16`) — the decode/latency numbers are FP16 runtime._")


def _fp16note(s):
    return s + FP16_NOTE if "FP16*" in s else s


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
        lines.append([short(r["model"]), r["model_type"], fmt_src(r["variant"]), f"`{r['model']}`"])
    lines.sort(key=lambda x: (x[0], x[2]))
    prev = None                        # blank short-name AND type within a family
    for ln in lines:
        if ln[0] == prev:
            ln[1] = ln[0] = ""
        else:
            prev = ln[0]
    return md(["Short Name", "Model Type", "Format", "Official Checkpoint"], lines)


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
        if any("quality=fail" in r.get("notes", "") for r in d.values()):
            note.append("⚠ degenerate output (GPTQ-on-Volta) — speed-only")
        lines.append([short(mdl), fmt(var), c19, c21, "; ".join(note)])
    _blank_repeat(lines, 0)
    return md(["Model", "Format", "vLLM 0.19 C1", "vLLM 0.21 C1", "Notes"], lines)


def engine_matrix(ver):                # Tables 3/4: per-engine decode, per-user + aggregate Type rows
    cells, order = {}, []
    for r in rows:
        if not _pv2(r) or r["vllm_version"] != ver:
            continue
        key = (r["model"], r["variant"], r["tp"], r["max_model_len"], r["notes"])
        c = cells.get(key)
        if c is None:
            c = cells[key] = {"pu": {}, "ag": {}}
            order.append(key)
        c["pu"][r["users"]] = r["tok_s_per_user"]
        c["ag"][r["users"]] = r["tok_s_aggregate"]
    order.sort(key=lambda k: (short(k[0]), k[1], k[2]))   # model, format, tp
    lines = []
    for (mdl, var, tp, mlen, notes) in order:
        c = cells[(mdl, var, tp, mlen, notes)]
        tag = "same-TP" if "same-TP" in notes else ("½-GPU (short ctx)" if "TP2 half-GPU" in notes else "")
        head2 = [short(mdl), fmt(var), f"TP{tp}", mlen]
        pu = head2 + ["Per-user"] + [c["pu"].get(u, "—") for u in ("1", "2", "4", "8")] + [tag]
        ag = ["", "", "", ""] + ["Aggregate"] + [c["ag"].get(u, "—") for u in ("1", "2", "4", "8")] + [""]
        lines += [pu, ag]
    _blank_repeat(lines, 0)                                # blank repeated model (now only on per-user rows)
    return md(["Model", "Format", "TP", "Max Context", "Type", "C1", "C2", "C4", "C8", "Notes"],
              lines, align=["l", "l", "l", "r", "l", "r", "r", "r", "r", "l"])


def ttft_matrix():                     # Ch1: dual-engine first-token latency (single-stream, same-TP C1)
    pick = {}                          # (model,variant) -> {ver: row}; same selection as baseline()
    for r in rows:
        if not _pv2(r) or r["users"] != "1" or "TP2 half-GPU" in r["notes"]:
            continue
        d = pick.setdefault((r["model"], r["variant"]), {})
        cur = d.get(r["vllm_version"])
        if cur is None or ("same-TP" in r["notes"] and "same-TP" not in cur["notes"]):
            d[r["vllm_version"]] = r
    s_ = lambda v: f"{v} s" if v else "—"
    lines = []
    for (mdl, var), d in sorted(pick.items(), key=lambda kv: (short(kv[0][0]), kv[0][1])):
        r21, r19 = d.get("0.21.0"), d.get("0.19.0")
        rep = r21 or r19 or {}
        tp = rep.get("tp", "")
        cold19 = s_(r19["ttft_s"]) if r19 and r19.get("ttft_s") else "—"
        cold21 = s_(r21["ttft_s"]) if r21 and r21.get("ttft_s") else "—"
        warm = s_(rep.get("ttft_prefix_hit_s")) if rep.get("ttft_prefix_hit_s") else "—"
        fa = s_(rep.get("ttft_fa_cold_s")) if rep.get("ttft_fa_cold_s") else "—"
        lines.append([short(mdl), fmt(var), f"TP{tp}", cold19, cold21, fa, warm])
    _blank_repeat(lines, 0)
    return (md(["Model", "Format", "TP", "Cold 0.19", "Cold 0.21", "FA-on Cold<br>(0.21)",
                "Prefix-cache Hit<br>(0.21)"], lines,
               align=["l", "l", "l", "r", "r", "r", "r"]) + TTFT_NOTE)


# ---- Chapter-6 digest views: curated, FROZEN (perf_v2-only) render views per model family ------
# A digest NEVER selects legacy Ch1/TP-sweep rows — only implementation_ref == fp8-v100-2026-matrix.
# One shared per-family config list drives both: single_user pivots it to columns, concurrency to rows.
PV2 = "fp8-v100-2026-matrix"
DIGEST_SPECS = {
    "qwen3_6_27b": {
        "match": "Qwen3.6-27B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16* TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4")],   # fleet TP4; TP2 = Ch5 capacity note, not scaling
    },
    "qwen3_6_35b_a3b": {
        "match": "Qwen3.6-35B-A3B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16* TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4")],   # fleet TP4; TP2 = Ch5 capacity note, not scaling
    },
    "qwen3_5_27b": {
        "match": "Qwen3.5-27B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16* TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4")],   # fleet TP4; TP2 = Ch5 capacity note
    },
    "qwen3_5_35b_a3b": {
        "match": "Qwen3.5-35B-A3B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16* TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4")],   # fleet TP4; TP2 = Ch5 capacity note
    },
    "qwen3_5_122b_a10b": {
        "match": "Qwen3.5-122B-A10B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP8 TP8", "fp8", "8"), ("GPTQ-Int4 TP8", "GPTQ-Int4", "8")],
    },
    "gemma4_31b": {
        "match": "gemma-4-31B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16* TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4"),
                    ("FP8 TP2", "fp8", "2", "single_user")],
    },
    "gemma4_26b_a4b": {
        "match": "gemma-4-26B-A4B",
        "engines": ["0.19.0", "0.21.0"],
        "configs": [("FP16* TP4", "fp16", "4"), ("FP8 TP4", "fp8", "4"),
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
        "configs": [("FP16* TP4", "bf16", "4")],   # perf label = FP16 (runtime); BF16 ckpt noted in prose
    },
}


def _digest_row(match, eng, variant, tp, users):
    for r in rows:
        if (r["implementation_ref"] in (PV2, "qwen35-exact-triad") and match in r["model"] and r["vllm_version"] == eng
                and r["variant"] == variant and r["tp"] == tp and r["users"] == users):
            return r
    return None


def _eng(v):
    return v[:-2] if v.endswith(".0") else v   # 0.19.0 -> 0.19


def _cfg(c):                           # (label, variant, tp, [flag]) -> (label, variant, tp, su_only)
    return c[0], c[1], c[2], (len(c) > 3 and c[3] == "single_user")


TTFT_NOTE = (
    "\n\nAll TTFT is single-stream, chunked-prefill **on** (the project-standard serve — disabling chunked "
    "prefill is a known V100 crash-causer). **Cold first-token** = a fresh, cache-cold request prefilling "
    "the full ~22.6k-token prompt (worst case); **Prefix-cache-hit** = the same prompt with its prefix "
    "already cached — repeated or shared context (best case). Cold TTFT is prefill-bound, and the Qwen "
    "**block-FP8** checkpoints carry a large prefill penalty on V100 (an unoptimized FP8-prefill path, "
    "worst on the MoE models) — a latency-side current-state limit, not where FP8's *decode* win lives; "
    "compressed-tensors FP8 (Gemma/GLM) and FP16/Int4 prefill cheaper.")


def single_user(key):                  # C1 deployment summary: per-engine decode only (TTFT -> its own table)
    s = DIGEST_SPECS[key]
    cfgs = [_cfg(c) for c in s["configs"]]
    engs = s["engines"]
    head = ["Choice"] + [f"{_eng(e)} C1 Decode" for e in engs]
    lines = []
    for lbl, var, tp, _ in cfgs:
        row = [lbl]
        for e in engs:
            r = _digest_row(s["match"], e, var, tp, "1")
            row.append(f"{r['tok_s_per_user']} tok/s" if r else "—")
        lines.append(row)
    return md(head, lines, align=["l"] + ["r"] * len(engs))


def ttft(key):                         # per-engine first-token latency: cold / (FA-on cold) / prefix-cache-hit
    s = DIGEST_SPECS[key]
    cfgs = [_cfg(c) for c in s["configs"]]
    s_ = lambda v: f"{v} s" if v else "—"
    picked = []                        # (config_label, engine, row) for every config × engine that has TTFT
    for lbl, var, tp, _ in cfgs:
        for e in s["engines"]:
            r = _digest_row(s["match"], e, var, tp, "1")
            if r and (r.get("ttft_s") or r.get("ttft_prefix_hit_s") or r.get("ttft_fa_cold_s")):
                picked.append((lbl, e, r))
    has_fa = any(r.get("ttft_fa_cold_s") for _, _, r in picked)
    head = ["Choice", "Engine", "Cold First Token"] + (["FA-on Cold"] if has_fa else []) + ["Prefix-cache Hit"]
    lines = []
    for lbl, e, r in picked:
        row = [lbl, _eng(e), s_(r.get("ttft_s"))]
        if has_fa:
            row.append(s_(r.get("ttft_fa_cold_s")))
        row.append(s_(r.get("ttft_prefix_hit_s")))
        lines.append(row)
    _blank_repeat(lines, 0)            # blank repeated Choice — engine rows fold under their config
    return md(head, lines, align=["l", "l"] + ["r"] * (len(head) - 2)) + TTFT_NOTE


def concurrency(key):                  # same-TP scaling: per-config per-user/aggregate x C1/C2/C4/C8
    s = DIGEST_SPECS[key]
    lines = []
    for eng in s["engines"]:
        for lbl, var, tp, su_only in (_cfg(c) for c in s["configs"]):
            if su_only:                # half-GPU / fit options belong in the single-user table, not here
                continue
            pu = [f"{_eng(eng)} {lbl}", "Per-user"]
            ag = ["", "Aggregate"]
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


def triad(key):                        # Ch5 dual-engine triad: FP16/FP8/Int4 x {0.19,0.21}, per-user C1-C8 @TP4
    s = DIGEST_SPECS[key]
    match = s["match"]
    precs = [("FP16*", "fp16"), ("FP8", "fp8"), ("GPTQ-Int4", "GPTQ-Int4")]
    lines = []
    for plabel, var in precs:
        for e in ("0.19.0", "0.21.0"):
            cells = [(_digest_row(match, e, var, "4", u) or {}).get("tok_s_per_user", "—")
                     for u in ("1", "2", "4", "8")]
            lines.append([plabel, _eng(e)] + cells)
    _blank_repeat(lines, 0)            # blank repeated Precision — the two engine rows fold under it
    return md(["Precision", "Engine", "C1", "C2", "C4", "C8"], lines, align=["l", "l", "r", "r", "r", "r"])


def resolve(cmd):                      # public entry: dispatch, then attach the FP16* footnote if present
    return _fp16note(_dispatch(cmd.strip()))


def _dispatch(cmd):
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
    if cmd.startswith("ttft:"):
        return ttft(cmd.split(":", 1)[1])
    if cmd.startswith("concurrency:"):
        return concurrency(cmd.split(":", 1)[1])
    if cmd.startswith("triad:"):
        return triad(cmd.split(":", 1)[1])
    if cmd == "ttft_matrix":
        return ttft_matrix()
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
