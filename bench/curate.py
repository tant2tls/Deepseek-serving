#!/usr/bin/env python3
"""Copy one model's reviewed raw runs from git-ignored results/<study>/<model>/
into the publishable reports/<study>/data/<model>/ tree, recording every change.

Rules (see AGENTS.md): numbers are never altered. Large files are gzipped
losslessly; server logs have private IPs and ANSI codes removed; prompt files and
full profiler traces are excluded. Every file gets a SHA-256 of its source and of
the published copy in CURATION.json.
Usage: bench/curate.py v41-vs-0731 v41
"""
import gzip, hashlib, json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GZIP_OVER = 256 * 1024          # gzip files larger than this (lossless)
IP = re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}(?:\.\d{1,3}){3}\b")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EXCLUDE_DIRS = {"_prompts", "_routing_raw"}  # _routing_raw: per-request expert IDs at batch 8 and 64 (tens of MB)
EXCLUDE_SUFFIX = (".pt.trace.json.gz",)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    study, model = sys.argv[1], sys.argv[2]
    src_root = ROOT / "results" / study / model
    dst_root = ROOT / "reports" / study / "data" / model
    if dst_root.exists():
        shutil.rmtree(dst_root)
    ledger, excluded = [], []
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = src.relative_to(src_root)
        if set(rel.parts) & EXCLUDE_DIRS or src.name.endswith(EXCLUDE_SUFFIX) or \
                (rel.parts[0] == "profiles" and src.name.startswith("profiler_out_") and src.name != "profiler_out_0.txt") or \
                (rel.parts[0] == "profiles" and src.name.endswith(".profiler_out.txt") and "_rank0." not in src.name):
            excluded.append(str(rel)); continue
        raw = src.read_bytes()
        out, ops = raw, []
        if rel.parts[0] == "_server" and src.suffix == ".log":
            text = raw.decode("utf-8", errors="replace")
            t2 = IP.sub("<PRIVATE_IP>", text)
            if t2 != text: ops.append("private IPv4 -> <PRIVATE_IP>")
            t3 = ANSI.sub("", t2)
            if t3 != t2: ops.append("ANSI escape codes removed")
            out = t3.encode()
        dst = dst_root / rel
        if len(out) > GZIP_OVER and not src.name.endswith(".gz"):
            out = gzip.compress(out, mtime=0); dst = dst.with_name(dst.name + ".gz"); ops.append("gzip (lossless)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(out)
        ledger.append(dict(source=str(Path("results") / study / model / rel), published=str(dst.relative_to(ROOT)),
                           source_sha256=sha(raw), published_sha256=sha(out), operations=ops or ["copied"]))
    (dst_root / "CURATION.json").write_text(json.dumps(dict(
        study=study, model=model,
        policy="numbers unaltered; gzip is lossless; logs: private IPs and ANSI codes removed",
        excluded=dict(
            reason={"_prompts": "prefix prompt JSONL (~400 MB), regenerable with bench/run_matrix.py build_prefix_files and the recorded seeds",
                    "*.pt.trace.json.gz": "full per-rank torch profiler traces (~73 MB); kept locally per AGENTS.md",
                    "profiler_out_[1-7].txt": "per-rank duplicates of the rank-0 kernel summary",
                    "*_rank[1-7].profiler_out.txt": "the same per-rank summaries under the newer build's file names"},
            files=excluded),
        files=ledger), indent=1))
    print(f"published {len(ledger)} files, excluded {len(excluded)} -> {dst_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
