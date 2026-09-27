"""Validate the portable reference export; no GPU or third-party packages needed.

Optional: --source-root PATH also checks hashes against the original repository.
This validates integrity and saved arithmetic, not experimental validity.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    ledger = json.loads((ROOT / "references/provenance.json").read_text(encoding="utf-8"))
    arms = json.loads((ROOT / "references/arms.json").read_text(encoding="utf-8"))["arms"]
    records = {entry["path"]: entry for entry in ledger["files"]}
    check(len(records) == len(ledger["files"]), "Duplicate provenance paths")
    for relative, entry in records.items():
        path = ROOT / relative
        check(path.is_file(), f"Missing export: {relative}")
        if not path.is_file():
            continue
        check(digest(path) == entry["export_sha256"], f"Export hash mismatch: {relative}")
        check(path.stat().st_size == entry["export_bytes"], f"Export size mismatch: {relative}")
        if not entry["transformations"]:
            check(entry["source_sha256"] == entry["export_sha256"], f"Undocumented change: {relative}")
        if path.suffix == ".json":
            check(entry["source_sha256"] == entry["export_sha256"], f"JSON changed: {relative}")
        if args.source_root:
            original = args.source_root / entry["source"]
            check(original.is_file(), f"Missing original: {entry['source']}")
            if original.is_file():
                check(digest(original) == entry["source_sha256"], f"Source hash mismatch: {entry['source']}")
    if args.source_root:
        for entry in ledger["source_documents"]:
            original = args.source_root / entry["source"]
            check(original.is_file() and digest(original) == entry["source_sha256"],
                  f"Source document mismatch: {entry['source']}")

    points = {}
    metadata = {}
    for arm in arms:
        if not arm["included"]:
            check("exclusion" in arm, f"Missing exclusion reason: {arm['source']}")
            continue
        directory = ROOT / arm["path"]
        check((directory / "manifest.txt").is_file(), f"Missing manifest: {arm['path']}")
        files = sorted(directory.glob("*.json"))
        check(len(files) == arm["points"], f"Point count mismatch: {arm['path']}")
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = json.loads(path.read_text(encoding="utf-8"))
            points[relative] = data
            metadata[relative] = arm
            check(relative in records, f"Unregistered point: {relative}")
            check(path.with_suffix(".log").is_file(), f"Missing benchmark log: {relative}")
            check(data["completed"] == data["num_prompts"] and data["failed"] == 0,
                  f"Incomplete retained point: {relative}")
            check(data["duration"] > 0 and data["total_output_tokens"] > 0,
                  f"No useful work: {relative}")
            check(math.isclose(data["output_throughput"],
                               data["total_output_tokens"] / data["duration"], rel_tol=1e-6),
                  f"Throughput arithmetic mismatch: {relative}")
            if "peak_kv_cache_usage_perc" in data:
                check(0 <= data["peak_kv_cache_usage_perc"] <= 1, f"Invalid cache fraction: {relative}")
            if data.get("spec_decode_draft_tokens", 0) > 0:
                expected = 100 * data["spec_decode_accepted_tokens"] / data["spec_decode_draft_tokens"]
                check(math.isclose(data["spec_decode_acceptance_rate"], expected, rel_tol=1e-6),
                      f"Acceptance arithmetic mismatch: {relative}")

    for path in (ROOT / "references").rglob("*"):
        if path.is_file() and path.suffix in (".log", ".txt", ".json"):
            relative = path.relative_to(ROOT).as_posix()
            if path.name not in ("arms.json", "provenance.json"):
                check(relative in records, f"Unregistered evidence file: {relative}")

    with (ROOT / "references/results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == len(points), "CSV point count mismatch")
    check({row["source_json"] for row in rows} == set(points), "CSV point coverage mismatch")
    for row in rows:
        relative = row["source_json"]
        if relative not in points:
            continue
        data, arm = points[relative], metadata[relative]
        expected = dict(data)
        expected.update(model=data["model_id"], arm=Path(relative).parent.name,
                        role=arm["role"], scope=arm["scope"], workload=Path(relative).stem,
                        source_json=relative)
        for key, value in row.items():
            check(value == str(expected.get(key, "")), f"CSV mismatch: {relative}: {key}")

    # Check the rendered per-arm table values as well as the machine-readable CSV.
    for path in (ROOT / "references").glob("*/RESULTS.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        count = 0
        for line in lines:
            match = re.match(r"\| \[.*?\]\((results/[^)]+\.json)\) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|$", line)
            if not match:
                continue
            relative = (path.parent / match[1]).relative_to(ROOT).as_posix()
            if relative not in points:
                check(False, f"Unknown table point: {relative}")
                continue
            data = points[relative]
            for i, key in enumerate(("output_throughput", "median_ttft_ms", "median_tpot_ms"), 2):
                check(match[i] == f"{data[key]:.1f}", f"Rendered table mismatch: {relative}: {key}")
            check(match[5] == f"{data['completed']} / {data['num_prompts']}", f"Table completions: {relative}")
            count += 1
        check(count == sum(p.startswith(path.parent.relative_to(ROOT).as_posix() + "/") for p in points),
              f"Rendered table coverage mismatch: {path.name}")

    link_count = 0
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        body = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", body):
            if re.match(r"[a-zA-Z]+://", target) or target.startswith("#"):
                continue
            link_count += 1
            check((path.parent / target.split("#")[0]).exists(),
                  f"Broken local link in {path.relative_to(ROOT)}: {target}")

    # Report paths only; never print a possible credential's value.
    credential = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                            r"sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
    private_path = re.compile(r"/prj/|/home/[A-Za-z0-9_.-]+/|/tmp/claude-[A-Za-z0-9]+|[A-Z]:\\")
    total_bytes = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in (".git", "__pycache__", ".venv", ".codex", ".agents") for part in path.parts):
            continue
        total_bytes += path.stat().st_size
        check(path.stat().st_size < 10 * 1024 * 1024, f"Unexpected large publication file: {path.relative_to(ROOT)}")
        if path.suffix in (".md", ".json", ".log", ".txt", ".csv", ".sh"):
            body = path.read_text(encoding="utf-8")
            check(not credential.search(body), f"Possible credential: {path.relative_to(ROOT)}")
            check(not private_path.search(body), f"Private absolute path: {path.relative_to(ROOT)}")
            check(not re.search(r"^host:\s*(?!<HOST>\s*$)\S+", body, re.M),
                  f"Unredacted host: {path.relative_to(ROOT)}")

    if errors:
        print("FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(records)} imported files; {len(points)} result points; "
          f"{link_count} local links; {total_bytes / 1024 / 1024:.2f} MiB publication files.")
    print("Hashes, CSV/table values, saved arithmetic, exclusions, and targeted publication scan passed.")
    print("This does not establish telemetry validity, causal control, quality parity, or GPU reproducibility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
