#!/usr/bin/env python3
"""V4.1 vs 0731 matched-point comparison (mean ± std over valid repeats).
Ratio = V4.1 / 0731: > 1 is better for throughput, worse for latency.
A difference is 'inconclusive' when |mean diff| < 2 * sqrt(sd_a^2 + sd_b^2).
Usage: bench/compare.py [--study v41-vs-0731] [--csv out.csv]"""
import argparse, csv, glob, json, math, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Launch configs that hold each model's matched points. V4.1's idle-profiler
# launch was shown equivalent to `off` by a c1/c64 control (report section 4).
CFG = {
    "v41": {"concurrency": "off", "context": "off", "isolated": "off-profidle",
            "prefix:cache-off": "off-cache-off", "prefix:cold": "off-cold",
            "prefix:prewarmed": "off-prewarmed"},
    "v4-0731": {"concurrency": "off-profidle", "context": "off-profidle",
                "isolated": "off-profidle", "prefix:cache-off": "off-profidle-cache-off",
                "prefix:cold": "off-cold", "prefix:prewarmed": "off-prewarmed"},
}
METRICS = [("output_throughput", "Out tok/s", True), ("p50_ttft_ms", "TTFT p50", False),
           ("p95_ttft_ms", "TTFT p95", False), ("p50_tpot_ms", "TPOT p50", False),
           ("p95_itl_ms", "ITL p95", False), ("p50_e2el_ms", "E2E p50", False)]


def load(study, mk, wl_key):
    wl = wl_key.split(":")[0]
    cfg = CFG[mk][wl_key]
    out = defaultdict(list)
    for f in glob.glob(str(ROOT / "results" / study / mk / wl / cfg / "*" / "repeat-*" / "summary.json")):
        s = json.loads(Path(f).read_text())
        if s.get("valid"):
            out[Path(f).parts[-3]].append(s)
    return out


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--study", default="v41-vs-0731")
    ap.add_argument("--csv"); a = ap.parse_args()
    rows = []
    for wl_key in CFG["v41"]:
        A, B = load(a.study, "v41", wl_key), load(a.study, "v4-0731", wl_key)
        points = sorted(set(A) | set(B), key=lambda p: [int(x) for x in re.findall(r"\d+", p)])
        if not points:
            continue
        print(f"\n### {wl_key}\n")
        print("| Point | Metric | V4.1 | 0731 | V4.1/0731 | Verdict |")
        print("|---|---|---:|---:|---:|---|")
        for p in points:
            for k, label, higher_better in METRICS:
                ma, sa, na = ms([s.get(k) for s in A.get(p, [])])
                mb, sb, nb = ms([s.get(k) for s in B.get(p, [])])
                if ma is None or mb is None:
                    ratio, verdict = None, "pending" if mb is None else "V4.1 missing"
                else:
                    ratio = ma / mb if mb else None
                    noise = 2 * math.sqrt(sa ** 2 + sb ** 2)
                    if abs(ma - mb) <= noise:
                        verdict = "inconclusive"
                    else:
                        v41_better = (ma > mb) == higher_better
                        verdict = "V4.1 better" if v41_better else "0731 better"
                fa = "—" if ma is None else f"{ma:,.1f} ± {sa:,.1f} (n={na})"
                fb = "—" if mb is None else f"{mb:,.1f} ± {sb:,.1f} (n={nb})"
                fr = "—" if ratio is None else f"{ratio:.2f}"
                print(f"| {p} | {label} | {fa} | {fb} | {fr} | {verdict} |")
                rows.append(dict(workload=wl_key, point=p, metric=k, v41_mean=ma, v41_sd=sa, v41_n=na,
                                 v0731_mean=mb, v0731_sd=sb, v0731_n=nb, ratio_v41_over_0731=ratio,
                                 verdict=verdict))
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
