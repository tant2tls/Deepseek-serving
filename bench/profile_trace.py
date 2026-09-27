#!/usr/bin/env python3
"""Capture one labeled torch-profiler trace around a single controlled request.

Diagnostic only: traces are not timing results. Requires a server started with
`bench/serve.sh <model> off-profidle` and no other traffic.
Usage: bench/profile_trace.py --label prefill16k --isl 16384 --max-tokens 8
"""
import argparse, datetime, json, time, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:8000"
MODELS = {  # same pins as bench/run_matrix.py
    "v41": ("deepseek-ai/DeepSeek-V4.1-Flash", "dba1be0a40aa45a94ad051997016db3960a90277", "deepseek_v41"),
    "v4-0731": ("deepseek-ai/DeepSeek-V4-Flash-0731", "7872f01b1d1fe23eabc4c98b48bffcef5a386062", "deepseek_v4"),
}


def post(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="v41", choices=MODELS)
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=90001)
    ap.add_argument("--profile-dir")
    a = ap.parse_args()
    MODEL, REV, TOK_MODE = MODELS[a.model]
    a.profile_dir = a.profile_dir or str(ROOT / f"results/v41-vs-0731/{a.model}/profiles")

    from vllm.tokenizers import get_tokenizer
    from vllm.benchmarks.datasets import gen_prompt_decode_to_target_len
    tok = get_tokenizer(MODEL, tokenizer_mode=TOK_MODE, revision=REV)
    ids = np.random.default_rng(a.seed).integers(0, tok.vocab_size, size=a.isl).tolist()
    prompt, _, _ = gen_prompt_decode_to_target_len(
        tokenizer=tok, token_sequence=ids, target_token_len=a.isl, add_special_tokens=False)

    pdir = Path(a.profile_dir)
    before = {p for p in pdir.rglob("*") if p.is_file()}
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "ignore_eos": True, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    t_start = time.time()
    print("start_profile", post("/start_profile")[0])
    t0 = time.time()
    status, resp = post("/v1/chat/completions", body)
    t1 = time.time()
    print("stop_profile", post("/stop_profile")[0])
    usage = json.loads(resp)["usage"]
    # Trace files are flushed asynchronously by each rank.
    for _ in range(120):
        new = sorted(p for p in pdir.rglob("*") if p.is_file() and p not in before)
        if len(new) >= 8:
            break
        time.sleep(5)
    time.sleep(10)
    new = sorted(p for p in pdir.rglob("*") if p.is_file() and p not in before)
    meta = dict(label=a.label, model=MODEL, revision=REV, isl=a.isl, max_tokens=a.max_tokens, seed=a.seed, usage=usage,
                request_wall_s=t1 - t0, profile_started=datetime.datetime.fromtimestamp(t_start).isoformat(),
                files=[str(p.relative_to(ROOT)) for p in new],
                note="diagnostic profiled run; not a timing result")
    out = pdir / f"{a.label}.manifest.json"
    out.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
