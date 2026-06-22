#!/usr/bin/env python3
"""Build data/benchmark_matrix.csv (the SSOT) from the fp8-w8a16-sm70 working repo's
results/. NO hand-typed numbers — every row is extracted from a committed result file,
and result_path traces back to it. Robust to warmup outliers (uses medians / steady min).

Usage: python3 scripts/build_matrix_from_results.py [FP8_REPO_ROOT]
  default FP8_REPO_ROOT = /home/kumphanartd/vllm-fp8-w8a16-sm70
"""
import csv, glob, os, re, statistics, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/kumphanartd/vllm-fp8-w8a16-sm70"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "benchmark_matrix.csv")

COLS = ["model", "variant", "params_total_b", "params_active_b", "quant",
        "vllm_version", "torch_cuda", "gpu", "tp", "max_model_len", "users",
        "mode", "cudagraph", "mtp", "config", "tok_s_per_user", "tok_s_aggregate",
        "ttft_s", "memory_gb", "flags", "result_path", "notes",
        "model_type", "consolidated_path", "implementation_ref"]

# Official HF checkpoint id per (model key, quant) — shown verbatim in tables for reproducibility.
PARAMS = {"q27b": (27, 27), "q35b": (35, 3), "q122b": (122, 10),
          "g31b": (31, 31), "g26b": (26, 4), "glm": (106, 12), "glm47": (31, 3)}
TYPE = {"q27b": "Dense", "q35b": "MoE", "q122b": "MoE", "g31b": "Dense",
        "g26b": "MoE", "glm": "MoE", "glm47": "MLA MoE"}   # archetype (Models-Tested table)
CHECKPOINT = {
    ("q27b", "fp16"): "Qwen/Qwen3.6-27B",   ("q27b", "fp8"): "Qwen/Qwen3.6-27B-FP8",
    ("q35b", "fp16"): "Qwen/Qwen3.6-35B-A3B", ("q35b", "fp8"): "Qwen/Qwen3.6-35B-A3B-FP8",
    ("q122b", "fp8"): "Qwen/Qwen3.5-122B-A10B-FP8",
    ("q122b", "int4"): "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    ("g31b", "fp16"): "google/gemma-4-31B-it", ("g31b", "fp8"): "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
    ("g26b", "fp16"): "google/gemma-4-26B-A4B-it", ("g26b", "fp8"): "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
    ("glm", "fp8"): "zai-org/GLM-4.5-Air-FP8",
    ("glm47", "fp16"): "zai-org/GLM-4.7-Flash",
}
QLABEL = {"int4": "GPTQ-Int4"}  # show the quant method, not bare "int4"
def mname(key, q): return CHECKPOINT.get((key, q), key)
def mparams(key): return PARAMS.get(key, ("", ""))
def qlabel(q): return QLABEL.get(q, q)
def mtype(key): return TYPE.get(key, "")
CH1_TP = {"q27b": 4, "q35b": 4, "g26b": 4, "g31b": 4, "q122b": 8}
rows = []


def med(xs):
    return round(statistics.median(xs), 2) if xs else ""


# ---- Ch1 reliability/baseline (vLLM 0.21+cu126, cudagraph, single-user, TP per CH1_TP) ----
# decode anchor = Q1 essay rows (qid==1); take MEDIAN decode (drops the 160/warmup outliers),
# steady-state ttft = min over OK rows. These FP16-MoE cells are STOCK (pre-MoE-patch, 06-11).
ch1 = glob.glob(f"{REPO}/results/ch1_20260611/ch1.1_021/manifest.csv")
if ch1:
    by = {}
    with open(ch1[0]) as f:
        for r in csv.DictReader(f):
            if r["qid"] != "1" or r["tag"] != "OK":
                continue
            try:
                dec = float(r["decode_tps"]); tt = float(r["ttft_s"])
            except ValueError:
                continue
            by.setdefault((r["model"], r["prec"]), []).append((dec, tt))
    rel_path = "results/ch1_20260611/ch1.1_021/manifest.csv"
    for (label, prec), v in sorted(by.items()):
        key = label.split("-")[0]
        name = mname(key, prec); pt, pa = mparams(key)
        decs = [d for d, _ in v]
        # drop gross outliers (>3x median) before median — warmup/measurement glitches
        m0 = statistics.median(decs)
        decs = [d for d in decs if d <= 3 * m0]
        tts = [t for _, t in v]
        # config column is unambiguous (Codex review): the FP16-MoE Ch1 cells are pre-patch
        # stock; dense fp16 / GPTQ-int4 are unmodified vLLM; FP8 uses our sm_70 plugin.
        is_moe = key in ("q35b", "g26b")
        if is_moe and prec == "fp16":
            cfg = "stock(pre-moe-patch)"
        elif prec == "fp8":
            cfg = "fp8-plugin+coalesced"
        else:
            cfg = "stock-vllm"
        rows.append({"model": name, "variant": qlabel(prec), "params_total_b": pt,
                     "params_active_b": pa, "quant": qlabel(prec), "vllm_version": "0.21.0",
                     "torch_cuda": "cu126", "gpu": "V100-32GB", "tp": CH1_TP.get(key, ""),
                     "max_model_len": 8192, "users": 1, "mode": "cudagraph",
                     "cudagraph": 1, "mtp": 0, "config": cfg,
                     "tok_s_per_user": med(decs), "tok_s_aggregate": "",
                     "ttft_s": round(min(tts), 2) if tts else "", "memory_gb": "",
                     "flags": "skip-mm,ns8", "result_path": rel_path,
                     "notes": "Ch1 reliability; decode=median Q1, ttft=steady min"})


# ---- Ch2 MoE FP16 fix: stock vs +patch, single-stream + 8-user (q35b, g26b) ----
def parse_ab_single(path):  # returns {arm: decode_mean}
    out = {}
    if not os.path.exists(path):
        return out
    for m in re.finditer(r"^(\w+): decode_mean=([0-9.]+)", open(path).read(), re.M):
        out[m.group(1)] = float(m.group(2))
    return out


def parse_ab_8user(path):  # returns {arm: (per_user_mean_median, agg_median)}
    out = {}
    if not os.path.exists(path):
        return out
    per, agg = {}, {}
    for m in re.finditer(r"^(\w+) run\d+: users_ok=\S+ per-user decode min=[0-9.]+ mean=([0-9.]+) max=[0-9.]+ tok/s \| aggregate=([0-9.]+)", open(path).read(), re.M):
        per.setdefault(m.group(1), []).append(float(m.group(2)))
        agg.setdefault(m.group(1), []).append(float(m.group(3)))
    for arm in per:
        out[arm] = (med(per[arm]), med(agg[arm]))
    return out


MOE = {
    "q35b": ("results/moe_stages_ab_q35b_20260613_051056/SUMMARY.txt",
             "results/moe_stages_ab_q35b_20260613_053101/SUMMARY.txt"),
    "g26b": ("results/moe_stages_ab_g26b_20260613_051056/SUMMARY.txt",
             "results/moe_stages_ab_g26b_20260613_053254/SUMMARY.txt"),
}
ARM_CFG = {"base": "stock(pre-moe-patch)", "kbest": "+moe_patch(heuristic)",
           "auto": "+moe_patch(tuned-json)"}
for key, (sp, mp) in MOE.items():
    name = mname(key, "fp16"); pt, pa = mparams(key)
    single = parse_ab_single(f"{REPO}/{sp}")
    multi = parse_ab_8user(f"{REPO}/{mp}")
    for arm in ("base", "kbest", "auto"):
        if arm in single:
            rows.append({"model": name, "variant": "fp16", "params_total_b": pt,
                         "params_active_b": pa, "quant": "fp16", "vllm_version": "0.21.0",
                         "torch_cuda": "cu126", "gpu": "V100-32GB", "tp": 4,
                         "max_model_len": 8192, "users": 1, "mode": "cudagraph",
                         "cudagraph": 1, "mtp": 0, "config": ARM_CFG[arm],
                         "tok_s_per_user": single[arm], "tok_s_aggregate": "",
                         "ttft_s": "", "memory_gb": "", "flags": "skip-mm,ns8",
                         "result_path": sp, "notes": "Ch2 MoE fix A/B (single-stream)"})
        if arm in multi:
            pu, ag = multi[arm]
            rows.append({"model": name, "variant": "fp16", "params_total_b": pt,
                         "params_active_b": pa, "quant": "fp16", "vllm_version": "0.21.0",
                         "torch_cuda": "cu126", "gpu": "V100-32GB", "tp": 4,
                         "max_model_len": 8192, "users": 8, "mode": "cudagraph",
                         "cudagraph": 1, "mtp": 0, "config": ARM_CFG[arm],
                         "tok_s_per_user": pu, "tok_s_aggregate": ag, "ttft_s": "",
                         "memory_gb": "", "flags": "skip-mm,ns8", "result_path": mp,
                         "notes": "Ch2 MoE fix A/B (8 concurrent)"})

# ---- TP × concurrency sweeps (model pages) ----
for sumf in sorted(glob.glob(f"{REPO}/results/tp_sweep_*/SUMMARY.txt")):
    dname = os.path.basename(os.path.dirname(sumf))      # tp_sweep_<key>_<prec>_<stamp>
    parts = dname.split("_")
    if len(parts) < 4:
        continue
    key, prec = parts[2], parts[3]
    name = mname(key, prec); pt, pa = mparams(key)
    rel = f"results/{dname}/SUMMARY.txt"
    txt = open(sumf).read()
    cell = {}   # (tp, users) -> {"pu":[...], "ag":[...]}
    for m in re.finditer(r"TP(\d+) users=(\d+) run\d+: per_user=([0-9.]+) tok/s aggregate=([0-9.]+)", txt):
        tp, users, pu, ag = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
        d = cell.setdefault((tp, users), {"pu": [], "ag": []})
        d["pu"].append(pu); d["ag"].append(ag)
    is_moe = key in ("q35b", "g26b", "q122b", "glm")
    cfg = "fp8-plugin+coalesced" if prec == "fp8" else ("+moe_patch" if is_moe else "stock-vllm")
    for (tp, users), d in sorted(cell.items()):
        rows.append({"model": name, "variant": prec, "params_total_b": pt,
                     "params_active_b": pa, "quant": prec, "vllm_version": "0.21.0",
                     "torch_cuda": "cu126", "gpu": "V100-32GB", "tp": int(tp),
                     "max_model_len": 4096, "users": int(users), "mode": "cudagraph",
                     "cudagraph": 1, "mtp": 0, "config": cfg,
                     "tok_s_per_user": med(d["pu"]), "tok_s_aggregate": med(d["ag"]),
                     "ttft_s": "", "memory_gb": "", "flags": "skip-mm,ns8",
                     "result_path": rel, "notes": "TP×concurrency sweep (model page)"})

# ---- Ch4 MTP A/B (off vs mtp k, accept%, exactness) from the chain summary ----
chain = f"{REPO}/results/ch2_mtp_20260612/CHAIN_SUMMARY.txt"
if os.path.exists(chain):
    rel = "results/ch2_mtp_20260612/CHAIN_SUMMARY.txt"
    text = open(chain).read()
    parts = re.split(r"────\s*(\S+)\s*────", text)
    for keyprec, body in zip(parts[1::2], parts[2::2]):
        key, _, prec = keyprec.rpartition("_")
        name = mname(key, prec); pt, pa = mparams(key)
        for km in re.finditer(
            r"k=(\d+): SPEEDUP off=([0-9.]+) -> mtp=([0-9.]+|nan) tok/s = (\S+) \| "
            r"accept=(\S+) \| EXACTNESS: (\w+)", body):
            k, offv, mtpv, spd, acc, exact = km.groups()
            if mtpv == "nan":
                continue  # failed/unsupported cell (e.g. gemma NotImplementedError) — noted in prose
            rows.append({"model": name, "variant": prec, "params_total_b": pt,
                         "params_active_b": pa, "quant": prec, "vllm_version": "0.21.0",
                         "torch_cuda": "cu126", "gpu": "V100-32GB", "tp": "",
                         "max_model_len": 4096, "users": 1, "mode": "cudagraph",
                         "cudagraph": 1, "mtp": int(k), "config": f"+mtp(k={k})",
                         "tok_s_per_user": float(mtpv), "tok_s_aggregate": "", "ttft_s": "",
                         "memory_gb": "", "flags": "skip-mm,ns8",
                         "result_path": rel,
                         "notes": f"Ch4 MTP; off={offv}; speedup={spd}; accept={acc}; exactness={exact}"})

# ---- Ch4 MTP k>=2 sweep: per-model dirs (single model each, same "k=N: SPEEDUP ..." line format as
# the chain). dir -> (key, prec, which k's to take) — the k-filter picks the canonical run where a
# (model,k) was measured in more than one dir (e.g. 122B k=2). k=1 still comes from the chain above.
MTP_KGE2 = [
    ("ch2_mtp_k2_slots512_27b",   "q27b",  "fp8", (2,)),
    ("ch2_mtp2_slots512_35b",     "q35b",  "fp8", (2,)),
    ("ch2_mtp_k34_slots512_35b",  "q35b",  "fp8", (3, 4)),
    ("ch2_mtp2_slots512_122b",    "q122b", "fp8", (2,)),   # the 1.45x "breakthrough" k=2 run
    ("ch2_mtp_k23_slots512_122b", "q122b", "fp8", (3,)),   # k=3 (this dir's k=2 duplicates the line above)
    ("ch2_mtp_k4_slots512_122b",  "q122b", "fp8", (4,)),
]
MTP_LINE = re.compile(
    r"k=(\d+): SPEEDUP off=([0-9.]+) -> mtp=([0-9.]+|nan) tok/s = (\S+) \| "
    r"accept=(\S+?)(?:\s*\([^)]*\))? \| EXACTNESS: (\w+)")  # tolerate a trailing "(parser fixed…)" note
for d, key, prec, ks in MTP_KGE2:
    dd = f"{REPO}/results/{d}"
    if not os.path.isdir(dd):
        continue
    seen = {}
    for fn in sorted(glob.glob(f"{dd}/*.txt") + glob.glob(f"{dd}/*.log")):
        for m in MTP_LINE.finditer(open(fn, errors="ignore").read()):
            kk = int(m.group(1))
            if kk in ks and kk not in seen and m.group(3) != "nan":
                seen[kk] = m.groups()
    name = mname(key, prec); pt, pa = mparams(key)
    for kk in ks:
        if kk not in seen:
            continue
        _, offv, mtpv, spd, acc, exact = seen[kk]
        rows.append({"model": name, "variant": prec, "params_total_b": pt,
                     "params_active_b": pa, "quant": prec, "vllm_version": "0.21.0",
                     "torch_cuda": "cu126", "gpu": "V100-32GB", "tp": "",
                     "max_model_len": 4096, "users": 1, "mode": "cudagraph",
                     "cudagraph": 1, "mtp": kk, "config": f"+mtp(k={kk})",
                     "tok_s_per_user": float(mtpv), "tok_s_aggregate": "", "ttft_s": "",
                     "memory_gb": "", "flags": "skip-mm,ns8", "result_path": f"results/{d}",
                     "notes": f"Ch4 MTP; off={offv}; speedup={spd}; accept={acc}; exactness={exact}"})

# ---- Ch3 eager-vs-cudagraph: pivot to one row per model (official name), columns
# eager | cudagraph | improvement(cg/eager). CH3_MODELS tracks all candidate families, but only those
# with a measured eager+cudagraph pair are emitted (unmeasured families are omitted, not 'pending').
# Kept SEPARATE from the main matrix so eager numbers can never leak into serving tables.
# One representative serving config per family. ----
CH3_MODELS = [("q27b", "fp16"), ("q35b", "fp8"), ("glm47", "fp16"), ("q122b", "fp8"),
              ("g31b", "fp16"), ("g26b", "fp8"), ("glm", "fp8")]
evc = sorted(glob.glob(f"{REPO}/results/eager_vs_cudagraph_*/SUMMARY.csv"))
meas, evc_src = {}, ""
if evc:
    evc_src = os.path.relpath(os.path.dirname(evc[-1]), REPO)
    for r in csv.DictReader(open(evc[-1])):
        meas.setdefault((r["model"], r["prec"]), {})[r["mode"]] = (r["decode_tps"], r["result_log"])

# GLM-4.7-Flash's eager/cudagraph pair is a one-off MLA A/B with a different summary format
# (free-text, not SUMMARY.csv) — extract its short-prompt steady-state tok/s from those files so the
# Ch3 row stays reproducible (no hand-typed numbers; survives a rebuild).
G47_DIR = f"{REPO}/results/glm47_mla_v100_20260615"
def _g47_short_tps(fname):
    p = os.path.join(G47_DIR, fname)
    if os.path.exists(p):
        for ln in open(p):
            if "[ON/short]" in ln:
                m = re.search(r"\(([\d.]+) tok/s\)", ln)
                if m:
                    return m.group(1), f"results/glm47_mla_v100_20260615/{fname}"
    return ("", "")
_g47e = _g47_short_tps("eager_SUMMARY.txt")
_g47c = _g47_short_tps("cudagraph_ON_SUMMARY.txt")
if _g47e[0] and _g47c[0]:
    meas[("glm47", "fp16")] = {"eager": _g47e, "cudagraph": _g47c}

ch3_out = os.path.join(os.path.dirname(OUT), "eager_vs_cudagraph.csv")
with open(ch3_out, "w", newline="") as f:
    # LF line endings (the repo convention for hand-reviewed files); csv defaults to CRLF.
    w = csv.DictWriter(f, fieldnames=["model", "eager_tok_s", "cudagraph_tok_s", "improvement",
                                      "eager_log", "cudagraph_log"], lineterminator="\n")
    w.writeheader()
    nmeas, nskip = 0, 0
    for key, prec in CH3_MODELS:
        m = meas.get((key, prec), {})
        eg, eg_log = m.get("eager", ("", ""))
        cg, cg_log = m.get("cudagraph", ("", ""))
        if not (eg and cg):          # emit ONLY models with a clean eager+cudagraph pair;
            nskip += 1               # unmeasured families stay in CH3_MODELS and auto-appear when run
            continue
        imp = f"{float(cg) / float(eg):.2f}x"
        nmeas += 1
        w.writerow({"model": mname(key, prec), "eager_tok_s": eg, "cudagraph_tok_s": cg,
                    "improvement": imp, "eager_log": eg_log, "cudagraph_log": cg_log})
    print(f"wrote Ch3 eager/cudagraph ({nmeas} measured, {nskip} unmeasured/omitted; "
          f"src={evc_src or 'none'})")

# ---- perf_v2 dual-engine matrix (Tables 1-3 source): 0.19+0.21, FP8/FP16/BF16/Int4, C1-C8 ----
# From results/perf_v2_COMBINED.csv (reconciled, with per-metric raw-dir provenance). One row per
# (model,prec,engine,tp) x users in {1,2,4,8}. result_path = the raw decode dir; the frozen impl is
# stamped via consolidated_path + implementation_ref. Restores glm/glm47/gemma with rigorous dual-
# engine numbers (the older single-engine/hand-added rows are superseded).
IMPL_REF = "fp8-v100-2026-matrix"
PV2_VARIANT = {"q27b4": ("q27b", "same-TP (TP4) precision comparison"),
               "q35b2": ("q35b", "TP2 half-GPU (reduced max-len)"),
               "g31b2": ("g31b", "TP2 half-GPU (reduced max-len)"),
               "g26b2": ("g26b", "TP2 half-GPU (reduced max-len)")}
PV2_VER = {"021": ("0.21.0", "cu126"), "019": ("0.19.0", "cu126")}
TF5 = {"g31b", "g26b", "glm47"}   # gemma-4 + GLM-4.7 need transformers-5 (cu128 image on 0.19)
combined = f"{REPO}/results/perf_v2_COMBINED.csv"
npv2 = 0
if os.path.exists(combined):
    for r in csv.DictReader(open(combined)):
        if r.get("quality") == "MISSING" or not r.get("dC1"):
            continue
        mk, prec, eng = r["model"], r["prec"], r["engine"]
        base, vnote = PV2_VARIANT.get(mk, (mk, ""))
        name = mname(base, prec); pt, pa = mparams(base)
        ver, tc = PV2_VER.get(eng, (eng, ""))
        if eng == "019" and base in TF5:
            tc = "cu128"
        vlabel = "bf16" if base == "glm47" else qlabel(prec)
        if base == "glm47":
            cfg = "fp16mla+cudagraph"
            flags = "mla-prefill,mla-decode-cudagraph,fa-v100" + (",tf5" if eng == "019" else "")
        elif prec == "fp8":
            cfg, flags = "fp8-plugin+coalesced", "skip-mm,ns8"
        elif mtype(base) != "Dense" and prec in ("fp16", "bf16"):
            cfg, flags = "+moe_patch", "skip-mm,ns8"
        else:
            cfg, flags = "stock-vllm", "skip-mm,ns8"
        if eng == "019" and base in TF5 and base != "glm47":
            flags += ",tf5"
        mlen = 8192 if mk.endswith("2") else 32768
        decode_src = r.get("decode_src", "")
        for users, col in ((1, "dC1"), (2, "dC2"), (4, "dC4"), (8, "dC8")):
            pu = r.get(col, "")
            if pu in ("", None):
                continue
            puf = float(pu)
            agg = float(r["aggC8"]) if (users == 8 and r.get("aggC8")) else round(puf * users, 2)
            note = f"perf_v2 dual-engine; quality={r.get('quality','')}; exactness={r.get('exactness','')}"
            if vnote:
                note += f"; {vnote}"
            rows.append({"model": name, "variant": vlabel, "params_total_b": pt,
                         "params_active_b": pa, "quant": vlabel, "vllm_version": ver,
                         "torch_cuda": tc, "gpu": "V100-32GB", "tp": r.get("tp", ""),
                         "max_model_len": mlen, "users": users, "mode": "cudagraph",
                         "cudagraph": 1, "mtp": 0, "config": cfg, "tok_s_per_user": puf,
                         "tok_s_aggregate": agg,
                         "ttft_s": r.get("ttft_long_cold_mono", "") if users == 1 else "",
                         "memory_gb": "", "flags": flags, "result_path": decode_src,
                         "notes": note, "model_type": mtype(base),
                         "consolidated_path": "results/perf_v2_COMBINED.csv",
                         "implementation_ref": IMPL_REF})
            npv2 += 1
print(f"perf_v2: {npv2} rows from COMBINED.csv")

# Fill model_type + provenance for EVERY row (SSOT-driven; older sections get type from the name).
NAME_TYPE = {nm: mtype(k) for (k, q), nm in CHECKPOINT.items()}
for r in rows:
    r.setdefault("model_type", NAME_TYPE.get(r["model"], ""))
    r.setdefault("consolidated_path", "")
    r.setdefault("implementation_ref", "")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote {OUT}: {len(rows)} rows")
print("models:", sorted({r['model'] for r in rows}))
