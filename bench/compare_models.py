#!/usr/bin/env python3
"""Cross-study matched-point comparison: one model vs reference models on the same
workload axes (same node, runtime, harness, seeds; tokenizers differ).
Mean ± SD over valid repeats; ratio = model / reference (> 1 better for throughput).
'inconclusive' when |mean diff| <= 2 * sqrt(sd_a^2 + sd_b^2), as in compare.py.
Usage: bench/compare_models.py [--csv out.csv] > tables.md"""
import argparse, csv, glob, json, math, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# name: (study, model key, {workload[:state]: config})
MODELS = {
    "MiMo": ("mimo-v26", "mimo-v26", {
        "concurrency": "off-profidle", "context": "off-profidle", "isolated": "off-profidle",
        "prefix:cache-off": "off-profidle-cache-off", "prefix:cold": "off-cold",
        "prefix:prewarmed": "off-prewarmed"}),
    "0731": ("v41-vs-0731", "v4-0731", {
        "concurrency": "off-profidle", "context": "off-profidle", "isolated": "off-profidle",
        "prefix:cache-off": "off-profidle-cache-off", "prefix:cold": "off-cold",
        "prefix:prewarmed": "off-prewarmed"}),
    "V4.1": ("v41-vs-0731", "v41", {
        "concurrency": "off", "context": "off", "isolated": "off-profidle",
        "prefix:cache-off": "off-cache-off", "prefix:cold": "off-cold",
        "prefix:prewarmed": "off-prewarmed"}),
}
SUBJECT, REFS = "MiMo", ["0731", "V4.1"]
METRICS = [("output_throughput", "Out tok/s", True), ("median_ttft_ms", "TTFT p50", False),
           ("median_tpot_ms", "TPOT p50", False)]


def load(name, wl_key):
    study, mk, cfgs = MODELS[name]
    out = defaultdict(list)
    for f in glob.glob(str(ROOT / "results" / study / mk / wl_key.split(":")[0] / cfgs[wl_key]
                           / "*" / "repeat-*" / "summary.json")):
        s = json.loads(Path(f).read_text())
        if s.get("valid"):
            out[Path(f).parts[-3]].append(s)
    return out


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    rows = []
    for wl in MODELS[SUBJECT][2]:
        data = {n: load(n, wl) for n in MODELS}
        points = sorted(data[SUBJECT], key=lambda p: [int(x) for x in re.findall(r"\d+", p)])
        if not points:
            continue
        print(f"\n### {wl}\n")
        print(f"| Point | Metric | {SUBJECT} | " + " | ".join(REFS) + " | "
              + " | ".join(f"{SUBJECT}/{r}" for r in REFS) + " |")
        print("|---|---|" + "---:|" * (1 + 2 * len(REFS)))
        for p in points:
            for k, label, hb in METRICS:
                m, s, n = ms([x[k] for x in data[SUBJECT][p]])
                cells, ratios = [], []
                for r in REFS:
                    mr, sr, nr = ms([x[k] for x in data[r].get(p, [])])
                    cells.append("—" if mr is None else f"{mr:,.1f} ± {sr:,.1f}")
                    if mr is None:
                        ratios.append("—"); continue
                    verdict = ("inconclusive" if abs(m - mr) <= 2 * math.sqrt(s ** 2 + sr ** 2)
                               else ("better" if (m > mr) == hb else "worse"))
                    ratios.append(f"{m / mr:.2f} {verdict}")
                    rows.append(dict(workload=wl, point=p, metric=k, subject=SUBJECT, ref=r,
                                     subject_mean=m, subject_sd=s, subject_n=n,
                                     ref_mean=mr, ref_sd=sr, ref_n=nr, ratio=m / mr, verdict=verdict))
                print(f"| {p} | {label} | {m:,.1f} ± {s:,.1f} (n={n}) | " + " | ".join(cells)
                      + " | " + " | ".join(ratios) + " |")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
