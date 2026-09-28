#!/usr/bin/env python3
"""DSpark arms vs each checkpoint's own ar (speculation-off) concurrency points.
Mean ± SD over valid repeats, plus median (0731 shows slow first repeats at new shapes).
Ratio = arm / ar; 'inconclusive' when |mean diff| <= 2 * sqrt(sd_a^2 + sd_b^2), as in compare.py.
Acceptance: accepted / scheduled (proposed) drafts, tokens per round = 1 + accepted / drafts,
per-position acceptance from telemetry metrics_{before,after}.prom deltas.
Usage: bench/spec_compare.py [--study v41-vs-0731] [--csv out.csv]"""
import argparse, csv, glob, json, math, statistics as st
from pathlib import Path

from run_matrix import PROM_RE

ROOT = Path(__file__).resolve().parent.parent
AR = {"v41": "off", "v4-0731": "off-profidle"}  # see compare.py CFG for the profidle control
ARMS = ["dspark-fixed-k5", "dspark-adaptive-k5"]
CONC = [1, 4, 16, 64]
METRICS = [("output_throughput", "Out tok/s", True), ("median_tpot_ms", "TPOT p50", False),
           ("median_ttft_ms", "TTFT p50", False), ("p95_itl_ms", "ITL p95", False)]


def runs(study, mk, cfg, c):
    out = []
    for f in sorted(glob.glob(str(ROOT / "results" / study / mk / "concurrency" / cfg /
                                  f"isl16384_osl256_c{c}" / "repeat-*" / "summary.json"))):
        s = json.loads(Path(f).read_text())
        if s.get("valid"):
            s["_dir"] = Path(f).parent
            out.append(s)
    return out


def per_pos(rdir):
    def read(name):
        d = {}
        for line in (rdir / "telemetry" / name).read_text().splitlines():
            m = PROM_RE.match(line)
            if m and m.group(1) == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                pos = int(m.group(2).split('position="')[1].split('"')[0])
                d[pos] = d.get(pos, 0) + float(m.group(3))
        return d
    a, b = read("metrics_after.prom"), read("metrics_before.prom")
    return {p: a[p] - b.get(p, 0) for p in a}


def stats(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), st.median(vals), len(vals)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--study", default="v41-vs-0731")
    ap.add_argument("--csv"); a = ap.parse_args()
    rows = []
    for mk in AR:
        print(f"\n### {mk}: DSpark arms / own ar ({AR[mk]})\n")
        print("| Load | Metric | ar | " + " | ".join(ARMS) + " |")
        print("|---|---|---:|" + "---:|" * len(ARMS))
        for c in CONC:
            base = runs(a.study, mk, AR[mk], c)
            arm_runs = {arm: runs(a.study, mk, arm, c) for arm in ARMS}
            for k, label, hb in METRICS:
                mb, sb, _, nb = stats([s[k] for s in base])
                cells = []
                for arm in ARMS:
                    if not arm_runs[arm]:
                        cells.append("pending"); continue
                    m, sd, med, n = stats([s[k] for s in arm_runs[arm]])
                    verdict = ("inconclusive" if abs(m - mb) <= 2 * math.sqrt(sd ** 2 + sb ** 2)
                               else ("better" if (m > mb) == hb else "worse"))
                    cells.append(f"{m:,.1f} ± {sd:,.1f} (med {med:,.1f}, n={n}) = {m / mb:.2f}× {verdict}")
                    rows.append(dict(model=mk, arm=arm, concurrency=c, metric=k, ar_mean=mb, ar_sd=sb,
                                     ar_n=nb, arm_mean=m, arm_sd=sd, arm_median=med, arm_n=n,
                                     ratio_arm_over_ar=m / mb, verdict=verdict))
                print(f"| c{c} | {label} | {mb:,.1f} ± {sb:,.1f} | " + " | ".join(cells) + " |")
        print(f"\n#### {mk}: acceptance (sums over valid repeats)\n")
        print("| Arm | Load | Drafts | Proposed | Accepted | Accepted/proposed | Tokens/round | Per-position acceptance (pos 0..4) |")
        print("|---|---|---:|---:|---:|---:|---:|---|")
        for arm in ARMS:
            for c in CONC:
                rs = runs(a.study, mk, arm, c)
                if not rs:
                    continue
                d = sum(s["spec_num_drafts"] for s in rs); p = sum(s["spec_num_draft_tokens"] for s in rs)
                acc = sum(s["spec_num_accepted_tokens"] for s in rs)
                pp = {}
                for s in rs:
                    for pos, v in per_pos(s["_dir"]).items():
                        pp[pos] = pp.get(pos, 0) + v
                pos_s = ", ".join(f"{pp[i] / d:.3f}" for i in sorted(pp))
                print(f"| {arm} | c{c} | {d:,.0f} | {p:,.0f} | {acc:,.0f} | {acc / p:.3f} | {1 + acc / d:.2f} | {pos_s} |")
                rows.append(dict(model=mk, arm=arm, concurrency=c, metric="acceptance",
                                 drafts=d, proposed=p, accepted=acc, accepted_over_proposed=acc / p,
                                 tokens_per_round=1 + acc / d,
                                 per_position=";".join(f"{pp[i] / d:.4f}" for i in sorted(pp))))
    print("\n### V4.1 / 0731 per arm (output tok/s)\n")
    print("| Load | ar | " + " | ".join(ARMS) + " |")
    print("|---|---:|" + "---:|" * len(ARMS))
    for c in CONC:
        cells = []
        for cfgs in [(AR["v41"], AR["v4-0731"])] + [(arm, arm) for arm in ARMS]:
            x = [s["output_throughput"] for s in runs(a.study, "v41", cfgs[0], c)]
            y = [s["output_throughput"] for s in runs(a.study, "v4-0731", cfgs[1], c)]
            cells.append(f"{st.mean(x) / st.mean(y):.2f}" if x and y else "pending")
        print(f"| c{c} | " + " | ".join(cells) + " |")
    if a.csv and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
