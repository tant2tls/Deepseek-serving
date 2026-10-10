#!/usr/bin/env python3
"""Build the frozen public-text request lists for study blog-architecture-h100-v1.

Every request is one user message: a short intro, whole public documents of one
domain (code / math / chat), and a closing task. The same text goes to all three
models; lengths are cut with the 0731 tokenizer as the reference and the actual
count under each model's tokenizer is saved. The server applies the chat template
(exactly once); nothing here renders a template.

  python bench/blog_inputs.py            # after bench/blog_corpus.py
Outputs under results/<study>/_inputs/: <bucket>/<split>.jsonl and manifest.json.
"""
import bisect, hashlib, itertools, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = os.environ.get("BLOG_STUDY", "blog-architecture-h100-v1")
INP = ROOT / "results" / STUDY / "_inputs"
RERUN = "five-model-rerun-h100-v1"
OSL = 2048 if STUDY == RERUN else 256
REF = "v4-0731"
DOMAINS = ["code", "math", "chat"]
# The frozen manifest counts tokens for the first three models only; later models are added with
# `extra` (extra_tokens.json), so a rebuild on a fresh node reproduces the same request lists.
MANIFEST_MODELS = ("v41", "v4-0731", "mimo-v26", "glm-53", "qwen-38-bf16") if STUDY == RERUN else \
    ("v41", "v4-0731", "mimo-v26")
# Reference-tokenizer content budget per bucket, and requests per split (multiples of 3).
# Splits are disjoint inside a bucket: timing is measured, aux serves pilots, warmups
# and diagnostic captures, heldout is reserved for confirmation.
BUCKETS = {
    "1k": dict(target=1024, timing=240, aux=36, heldout=24),
    "16k": dict(target=16384, timing=150, aux=24, heldout=12),
    "64k": dict(target=65536, timing=45, aux=9, heldout=3),
}
if STUDY == RERUN:
    # Same construction and the same 1K/16K lists as October (the files differ only in the requested
    # output length). 64K needs 48 timing requests for 16 clients, so the three former held-out
    # slots move into the timing split: its first 45 requests are October's. 128K serves the
    # context diagnostics only (prefill capture, decode at engine batch 1 and 8).
    BUCKETS["64k"] = dict(target=65536, timing=48, aux=9, heldout=0)
    BUCKETS["128k"] = dict(target=131072, timing=0, aux=9, heldout=0)
# Short-output copies of the aux requests, used only to warm shapes after a launch.
WARM_OSL = 128
INTRO = {
    "code": "Below are source files from the CPython standard library.\n\n",
    "math": "Below are worked mathematics problems with their solutions.\n\n",
    "chat": "Below are transcripts of conversations between a user and an assistant.\n\n",
}
TASK = {
    "code": "\nExplain what the code above does, file by file, and point out any subtle behaviour a maintainer should know.",
    "math": "\nWrite one new problem in the style of the problems above and solve it step by step, explaining each step.",
    "chat": "\nSummarise the main topics of the conversations above, then write a helpful, detailed reply to the last user message.",
}


def tokenizers():
    import sys
    sys.path.insert(0, str(ROOT / "bench"))
    from run_matrix import MODELS
    from vllm.tokenizers import get_tokenizer
    return {mk: get_tokenizer(m["model"], tokenizer_mode=m["tokenizer_mode"], revision=m["revision"],
                              trust_remote_code=True) for mk, m in MODELS.items() if mk in MANIFEST_MODELS}


def ntok(tok, text):
    return len(tok.encode(text, add_special_tokens=False))


def main():
    toks = tokenizers()
    ref = toks[REF]
    corpus, counts = {}, {}
    for d in DOMAINS:
        docs = [json.loads(l) for l in open(INP / "corpus" / f"{d}.jsonl")]
        corpus[d] = docs
        counts[d] = [ntok(ref, x["text"]) for x in docs]
        print(d, len(docs), "docs", sum(counts[d]), "reference tokens", flush=True)

    manifest = dict(study=STUDY, output_tokens=OSL, reference_tokenizer=REF, buckets={},
                    sources=json.loads((INP / "corpus" / "sources.json").read_text()),
                    note="Same text for every model; token counts differ by tokenizer. "
                         "Documents are unique within a bucket and may recur across buckets.")
    for bname, b in BUCKETS.items():
        cursor = {d: 0 for d in DOMAINS}
        fixed = {d: ntok(ref, INTRO[d]) + ntok(ref, TASK[d]) for d in DOMAINS}

        def build(domain):
            budget = b["target"] - fixed[domain]
            parts, used, ids = [], 0, []
            while used < budget:
                i = cursor[domain]
                if i >= len(corpus[domain]):
                    raise SystemExit(f"{bname}: {domain} corpus exhausted")
                cursor[domain] += 1
                text, n = corpus[domain][i]["text"], counts[domain][i]
                if used + n > budget:  # cut the last document at a line boundary
                    lines = text.splitlines(keepends=True)
                    acc = list(itertools.accumulate(ntok(ref, l) for l in lines))
                    k = bisect.bisect_right(acc, budget - used)
                    text, n = "".join(lines[:k]), (acc[k - 1] if k else 0)
                    if not k:
                        break
                parts.append(text); used += n; ids.append(corpus[domain][i]["doc_id"])
                if n == 0:
                    break
            return INTRO[domain] + "\n".join(parts) + TASK[domain], ids

        entry = dict(target_reference_tokens=b["target"], splits={})
        (INP / bname).mkdir(parents=True, exist_ok=True)
        for split in ("timing", "aux", "heldout"):
            rows, meta = [], []
            for j in range(b[split]):
                d = DOMAINS[j % 3]
                prompt, ids = build(d)
                rid = f"{bname}-{split}-{j:03d}"
                rows.append(dict(prompt=prompt, output_tokens=OSL))
                meta.append(dict(request_id=rid, domain=d, sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                 chars=len(prompt), n_docs=len(ids), first_doc=ids[0] if ids else None,
                                 content_tokens={mk: ntok(t, prompt) for mk, t in toks.items()}))
            body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            (INP / bname / f"{split}.jsonl").write_text(body)
            entry["splits"][split] = dict(file=f"{bname}/{split}.jsonl", n=len(rows),
                                          file_sha256=hashlib.sha256(body.encode()).hexdigest(), requests=meta)
            if STUDY == RERUN and split == "aux":
                (INP / bname / "warm.jsonl").write_text("".join(
                    json.dumps(dict(r, output_tokens=WARM_OSL), ensure_ascii=False) + "\n" for r in rows))
            mean = {mk: round(sum(m["content_tokens"][mk] for m in meta) / max(1, len(meta))) for mk in toks}
            print(bname, split, len(rows), "mean content tokens", mean, flush=True)
        manifest["buckets"][bname] = entry
    (INP / "manifest.json").write_text(json.dumps(manifest, indent=1))


def extra(models):
    """Token counts of the frozen requests under additional tokenizers (manifest stays unchanged)."""
    import sys
    sys.path.insert(0, str(ROOT / "bench"))
    from run_matrix import MODELS
    from vllm.tokenizers import get_tokenizer
    path = INP / "extra_tokens.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    man = json.loads((INP / "manifest.json").read_text())
    for mk in models:
        m = MODELS[mk]
        tok = get_tokenizer(m["model"], tokenizer_mode=m["tokenizer_mode"], revision=m["revision"], trust_remote_code=True)
        out[mk] = {}
        for bname, b in man["buckets"].items():
            for split, s in b["splits"].items():
                rows = [json.loads(l) for l in open(INP / s["file"])]
                for row, req in zip(rows, s["requests"]):
                    out[mk][req["request_id"]] = ntok(tok, row["prompt"])
            ts = [out[mk][r["request_id"]] for r in b["splits"]["timing"]["requests"]]
            print(mk, bname, "timing mean content tokens", round(sum(ts) / len(ts)), flush=True)
    path.write_text(json.dumps(out))


if __name__ == "__main__":
    import sys as _sys
    extra(_sys.argv[2:]) if len(_sys.argv) > 2 and _sys.argv[1] == "extra" else main()
