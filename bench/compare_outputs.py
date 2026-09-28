#!/usr/bin/env python3
"""Compare generated texts of a speculative arm with the matching ar run (same seed/prompts,
greedy target). Reports exact-match rate and mean common-prefix length per point.
Not a losslessness proof: greedy ar itself may be batch/numerics sensitive.
Usage: bench/compare_outputs.py <model-key> <spec-config> <ar-config> [--study v41-vs-0731]"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cpl(a, b):
    return len(os.path.commonprefix([a, b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("spec"); ap.add_argument("ar")
    ap.add_argument("--study", default="v41-vs-0731")
    a = ap.parse_args()
    base = ROOT / "results" / a.study / a.model / "concurrency"
    print(f"| Point | Repeat | Prompts | Exact match | Mean common prefix (chars) | Mean ar length |")
    print("| --- | --- | --- | --- | --- | --- |")
    for sb in sorted((base / a.spec).glob("*/repeat-*/bench.json")):
        ab = base / a.ar / sb.parent.parent.name / sb.parent.name / "bench.json"
        if not ab.exists():
            continue
        s, r = json.loads(sb.read_text())["generated_texts"], json.loads(ab.read_text())["generated_texts"]
        n = min(len(s), len(r))
        exact = sum(s[i] == r[i] for i in range(n))
        print(f"| {sb.parent.parent.name} | {sb.parent.name} | {n} | {exact}/{n} | "
              f"{sum(cpl(s[i], r[i]) for i in range(n)) / n:.0f} | {sum(len(x) for x in r[:n]) / n:.0f} |")


if __name__ == "__main__":
    main()
