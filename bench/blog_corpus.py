#!/usr/bin/env python3
"""Normalise the public source texts for study blog-architecture-h100-v1.

Runs in a scratch environment that has pyarrow (the pinned vLLM venv does not):
  /tmp/blog-inputs-venv/bin/python bench/blog_corpus.py
Writes results/<study>/_inputs/corpus/{code,math,chat}.jsonl and sources.json.
Texts are public and regenerable, so only hashes and provenance are published.
"""
import gzip, hashlib, json, os, sys, sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = os.environ.get("BLOG_STUDY", "blog-architecture-h100-v1")
OUT = ROOT / "results" / STUDY / "_inputs" / "corpus"
HUB = Path("/workspace/hf/hub")
MATH_REV = "21a5633873b6a120296cce3e2df9d5550074f4a3"
CHAT_REV = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
CHAT_FILE = "2023-04-12_oasst_ready.trees.jsonl.gz"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def code_docs():
    """CPython standard-library modules as installed on the node (PSF-2.0)."""
    std = Path(sysconfig.get_paths()["stdlib"])
    for p in sorted(std.rglob("*.py")):
        rel = p.relative_to(std).as_posix()
        if any(x in rel for x in ("site-packages", "dist-packages", "test", "idlelib", "__pycache__")):
            continue
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if 4000 <= len(t) <= 200_000:
            yield rel, f"# File: {rel}\n{t.rstrip()}\n"


def math_docs():
    """EleutherAI/hendrycks_math (MIT): problem plus worked solution."""
    import pyarrow.parquet as pq
    snap = HUB / "datasets--EleutherAI--hendrycks_math" / "snapshots" / MATH_REV
    for f in sorted(snap.rglob("*.parquet")):
        subj, split = f.parent.name, f.name.split("-")[0]
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            yield (f"{subj}/{split}/{i}",
                   f"Problem ({r.get('type', subj)}, {r.get('level', '')}):\n{r['problem'].strip()}\n\n"
                   f"Solution:\n{r['solution'].strip()}\n")


def chat_docs():
    """OpenAssistant/oasst1 (Apache-2.0): one English thread per tree, first reply at each turn."""
    snap = HUB / "datasets--OpenAssistant--oasst1" / "snapshots" / CHAT_REV
    with gzip.open(snap / CHAT_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            tree = json.loads(line)
            node = tree.get("prompt") or {}
            if node.get("lang") != "en":
                continue
            turns = []
            while node:
                who = "User" if node.get("role") == "prompter" else "Assistant"
                turns.append(f"{who}: {node.get('text', '').strip()}")
                kids = [k for k in node.get("replies", []) if k.get("lang") == "en" and not k.get("deleted")]
                node = kids[0] if kids else None
            if len(turns) >= 2:
                yield tree["message_tree_id"], "\n\n".join(turns) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = {
        "code": dict(source="CPython standard library as installed on the node", license="PSF-2.0",
                     version=sys.version.split()[0], path=sysconfig.get_paths()["stdlib"]),
        "math": dict(source="EleutherAI/hendrycks_math", license="MIT", revision=MATH_REV),
        "chat": dict(source="OpenAssistant/oasst1", license="Apache-2.0", revision=CHAT_REV, file=CHAT_FILE),
    }
    for name, gen in (("code", code_docs), ("math", math_docs), ("chat", chat_docs)):
        n = chars = 0
        h = hashlib.sha256()
        with open(OUT / f"{name}.jsonl", "w") as f:
            for doc_id, text in gen():
                line = json.dumps(dict(doc_id=doc_id, sha256=sha(text), text=text), ensure_ascii=False)
                f.write(line + "\n"); h.update(line.encode()); n += 1; chars += len(text)
        info[name].update(docs=n, chars=chars, corpus_sha256=h.hexdigest())
        print(name, n, "docs", chars, "chars")
    (OUT / "sources.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
