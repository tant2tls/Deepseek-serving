#!/usr/bin/env python3
"""4-prompt sanity check (thinking off, greedy) matching the DeepSeek _quality_smoke files.
Checks the thinking switch too: reasoning must be empty and finish_reason 'stop'.
Usage: bench/quality_smoke.py --model <key> [--study <study>]"""
import argparse, json, urllib.request
from pathlib import Path

from run_matrix import MODELS, extra_body, BASE

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = [("What is 17 * 19? Answer with just the number.", "323"),
           ("What is the capital of Australia? One word.", "Canberra"),
           ("Write a Python function is_prime(n). Only code.", "def is_prime"),
           ("If a train travels 120 km in 1.5 hours, what is its average speed in km/h? Just the number.", "80")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--study", default="v41-vs-0731")
    a = ap.parse_args()
    out = []
    for q, exp in PROMPTS:
        body = dict(model=MODELS[a.model]["model"], temperature=0, max_tokens=512,
                    messages=[{"role": "user", "content": q}], **extra_body(a.model))
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        out.append(dict(q=q, expected=exp, content=content, reasoning=reasoning,
                        finish=resp["choices"][0]["finish_reason"], usage=resp["usage"],
                        ok=exp in content and not reasoning))
        print(f"ok={out[-1]['ok']} finish={out[-1]['finish']} reasoning={bool(reasoning)} {content[:80]!r}")
    d = ROOT / "results" / a.study / a.model / "_quality_smoke"
    d.mkdir(parents=True, exist_ok=True)
    (d / "responses.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
