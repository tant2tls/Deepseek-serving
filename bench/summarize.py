#!/usr/bin/env python3
"""Aggregate summary.json files across repeats into markdown tables + CSV.
Usage: bench/summarize.py <study> <model-key> [--csv out.csv]
Only valid repeats enter the aggregates; invalid ones are listed separately."""
import argparse, csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = [("output_throughput", "Out tok/s", 1), ("request_throughput", "Req/s", 3),
          ("p50_ttft_ms", "TTFT p50", 0), ("p95_ttft_ms", "TTFT p95", 0),
          ("p50_tpot_ms", "TPOT p50", 1), ("p95_tpot_ms", "TPOT p95", 1),
          ("p50_itl_ms", "ITL p50", 1), ("p95_itl_ms", "ITL p95", 1),
          ("p50_e2el_ms", "E2E p50", 0), ("p95_e2el_ms", "E2E p95", 0)]
EXTRA = [("peak_running", "Peak running"), ("peak_waiting", "Peak waiting"),
         ("peak_kv_cache_usage_frac", "Peak KV frac"), ("preemptions", "Preempt"),
         ("prefix_cache_hits", "Prefix hits"), ("prefix_cache_queries", "Prefix queries"),
         ("peak_gpu_mem_mib", "Peak GPU MiB"), ("gpu_seconds_per_output_token", "GPU-s/out tok")]


def sort_key(p):
    import re
    return [int(x) for x in re.findall(r"\d+", p)]


def fmt(vals, nd):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "unavailable"
    m = st.mean(vals)
    if len(vals) > 1:
        return f"{m:,.{nd}f} ± {st.stdev(vals):,.{nd}f}"
    return f"{m:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study"); ap.add_argument("model")
    ap.add_argument("--csv")
    a = ap.parse_args()
    base = ROOT / "results" / a.study / a.model
    groups, invalid = defaultdict(list), []
    for f in sorted(base.glob("*/*/*/repeat-*/summary.json")):
        wl, cfg, point = f.parts[-5], f.parts[-4], f.parts[-3]
        s = json.loads(f.read_text())
        s["_dir"] = str(f.parent.relative_to(ROOT))
        (groups[(wl, cfg, point)] if s.get("valid") else invalid).append(s)
        if not s.get("valid"):
            s["_key"] = (wl, cfg, point)
    rows = []
    for (wl, cfg, point) in sorted(groups, key=lambda k: (k[0], k[1], sort_key(k[2]))):
        g = groups[(wl, cfg, point)]
        row = {"workload": wl, "config": cfg, "point": point, "n_repeats": len(g),
               "requests_per_repeat": g[0]["num_prompts"]}
        for k, _, nd in FIELDS:
            row[k] = fmt([s.get(k) for s in g], nd)
        for k, _ in EXTRA:
            vals = [s.get(k) for s in g if s.get(k) is not None]
            if not vals:
                row[k] = "unavailable"
            elif k in ("peak_kv_cache_usage_frac", "gpu_seconds_per_output_token"):
                row[k] = f"{max(vals):.3f}"
            else:
                row[k] = f"{max(vals):,.0f}"
        rows.append(row)
    for wl in dict.fromkeys(r["workload"] for r in rows):
        print(f"\n### {wl}\n")
        hdr = ["Config", "Point", "Reps×req"] + [h for _, h, _ in FIELDS]
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for r in rows:
            if r["workload"] == wl:
                print("| " + " | ".join([r["config"], r["point"], f"{r['n_repeats']}×{r['requests_per_repeat']}"]
                                        + [r[k] for k, _, _ in FIELDS]) + " |")
        print(f"\n| Config | Point | " + " | ".join(h for _, h in EXTRA) + " |")
        print("|" + "---|" * (len(EXTRA) + 2))
        for r in rows:
            if r["workload"] == wl:
                print(f"| {r['config']} | {r['point']} | " + " | ".join(r[k] for k, _ in EXTRA) + " |")
    if invalid:
        print("\n### Invalid runs (excluded from aggregates)\n")
        for s in invalid:
            print(f"- `{s['_dir']}`: {'; '.join(s['problems'])}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
