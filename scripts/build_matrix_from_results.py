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
        "ttft_s", "memory_gb", "flags", "result_path", "notes"]

# static model facts (params in billions; total/active)
MODEL = {
    "q27b":  ("Qwen3.6-27B",        27, 27),
    "q35b":  ("Qwen3.6-35B-A3B",    35, 3),
    "q122b": ("Qwen3.5-122B-A10B",  122, 10),
    "g31b":  ("gemma-4-31B-it",     31, 31),
    "g26b":  ("gemma-4-26B-A4B-it", 26, 4),
    "glm":   ("GLM-4.5-Air",        106, 12),
}
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
        name, pt, pa = MODEL.get(key, (label, "", ""))
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
        rows.append({"model": name, "variant": prec, "params_total_b": pt,
                     "params_active_b": pa, "quant": prec, "vllm_version": "0.21.0",
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
    name, pt, pa = MODEL[key]
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
    name, pt, pa = MODEL.get(key, (key, "", ""))
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
        name, pt, pa = MODEL.get(key, (keyprec, "", ""))
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

# ---- Ch3 eager-vs-cudagraph paired data: copy the latest SUMMARY.csv verbatim into data/
# (kept SEPARATE from the main matrix so eager numbers can never leak into serving tables) ----
import shutil
evc = sorted(glob.glob(f"{REPO}/results/eager_vs_cudagraph_*/SUMMARY.csv"))
if evc:
    shutil.copy(evc[-1], os.path.join(os.path.dirname(OUT), "eager_vs_cudagraph.csv"))
    print(f"copied Ch3 paired data from {evc[-1]}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote {OUT}: {len(rows)} rows")
print("models:", sorted({r['model'] for r in rows}))
