"""Sequential c1 prefix-reuse check: when does a 64K shared prefix become a cache hit?

Reconstructs the ad hoc sequence first run on 0731 (that script was not saved):
chat completions, thinking=false, max_tokens=1, prompts from
results/<study>/_prompts/prefix_n4_rep1 (prefix i = prewarm[i] and its measured suffixes).
  sequential.json   (after reset): A prewarm p0; B, C new suffixes of p0; D exact repeat of C
  sequential_2.json (after reset): E, F, G three suffixes of p1; H, I, J three suffixes of p2
Hits/queries are /metrics deltas per request (vllm:prefix_cache_{queries,hits}_total).
Usage: python bench/prefix_reuse_check.py --model v41 [--study v41-vs-0731]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from run_matrix import MODELS, EXTRA_BODY, http_get, http_post, BASE

ROOT = Path(__file__).resolve().parent.parent


def counter(txt, name):
    return sum(float(l.rsplit(" ", 1)[1]) for l in txt.splitlines()
               if l.startswith(name + "{") or l.startswith(name + " "))


def ask(model, prompt):
    body = dict(model=model, max_tokens=1, messages=[{"role": "user", "content": prompt}], **EXTRA_BODY)
    before = http_get("/metrics")
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)
    wall = round((time.perf_counter() - t0) * 1000)
    after = http_get("/metrics")
    d = {k: counter(after, f"vllm:prefix_cache_{k}_total") - counter(before, f"vllm:prefix_cache_{k}_total")
         for k in ("queries", "hits")}
    return d["queries"], d["hits"], wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--study", default="v41-vs-0731")
    a = ap.parse_args()
    model = MODELS[a.model]["model"]
    served = json.loads(http_get("/v1/models"))["data"][0]["id"]
    assert served == model, f"server serves {served}, expected {model}"
    pdir = ROOT / "results" / a.study / "_prompts" / "prefix_n4_rep1"
    prewarm = [json.loads(l)["prompt"] for l in open(pdir / "prewarm.jsonl")]
    measured = [json.loads(l)["prompt"] for l in open(pdir / "measured.jsonl")]
    order = json.loads((pdir / "order.json").read_text())
    suf = {i: [p for p, o in zip(measured, order) if o == i] for i in set(order)}

    seqs = {
        "sequential.json": [
            ("A: prewarm prompt (prefix+suffix0), after reset", prewarm[0]),
            ("B: same prefix, new suffix1", suf[0][0]),
            ("C: same prefix, new suffix2", suf[0][1]),
            ("D: exact repeat of C", suf[0][1]),
        ],
        "sequential_2.json": [
            ("E: prefix#2 suffix a (first after reset)", suf[1][0]),
            ("F: prefix#2 suffix b", suf[1][1]),
            ("G: prefix#2 suffix c", suf[1][2]),
            ("H: prefix#3 suffix a (NOT first after reset)", suf[2][0]),
            ("I: prefix#3 suffix b", suf[2][1]),
            ("J: prefix#3 suffix c", suf[2][2]),
        ],
    }
    out = ROOT / "results" / a.study / a.model / "_prefix_reuse_check"
    out.mkdir(parents=True, exist_ok=True)
    for fname, steps in seqs.items():
        http_post("/reset_prefix_cache")
        rows = []
        for step, prompt in steps:
            q, h, wall = ask(model, prompt)
            rows.append(dict(step=step, queried=q, hit=h, wall_ms=wall))
            print(json.dumps(rows[-1]))
        (out / fname).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
