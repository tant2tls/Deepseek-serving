"""Audit the selected publication: exact evidence bytes, arithmetic, scope, layout and links.

  python tools/audit_references.py     # run on a clean clone before publishing

Standard library only, no GPU. It fails when evidence bytes change, a generated table or the
article data is stale, a link or anchor is broken, a file lands outside the agreed layout, or a
model is pinned differently in two places. tools/README.md lists every check.
"""
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_blog_results import FIRST, SECOND, MODELS, POINTS, generate
from build_blog_page import check as page_check

LEDGER = 'reports/provenance.json'
ARCHIVE = 'reports/glm-53-september'  # September GLM export, hashed by its own ledger
ARCHIVE_INDEX = {'README.md', 'arms.json', 'results.csv', 'provenance.json',
                 'glm-5.3-flash/README.md', 'glm-5.3-flash/RESULTS.md'}
# The agreed layout. A new top-level entry is a decision: add it here and to README.md together.
FOLDERS = ('bench', 'docs', 'reports', 'reproduce', 'teach_me', 'tools')
ROOT_FILES = ('.gitattributes', '.gitignore', 'AGENTS.md', 'README.md', 'target.md', 'index.html', 'install.sh')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def anchors(p, body):
    """Link targets inside a page: element ids in HTML, GitHub-style heading slugs in Markdown."""
    if p.suffix == '.html':
        return set(re.findall(r'\bid="([^"]+)"', body))
    seen, found = collections.Counter(), set()
    for line in re.sub(r'```.*?```', '', body, flags=re.S).splitlines():
        heading = re.match(r'#{1,6}\s+(.*?)\s*$', line)
        if heading:
            title = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', heading.group(1))
            slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
            found.add(f'{slug}-{seen[slug]}' if seen[slug] else slug)
            seen[slug] += 1
    return found


def layout(check):
    """Top-level entries, folder indexes and one consistent pin per model."""
    listed = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode == 0 and listed.stdout:
        top = {line.split('/')[0] for line in listed.stdout.splitlines()}
    else:  # an exported tree without Git metadata
        top = {p.name for p in ROOT.iterdir() if p.name not in ('.git', '.audit', '.worktrees')}
    for name in sorted(top - set(FOLDERS) - set(ROOT_FILES)):
        check(False, f'Outside the agreed layout: {name}')
    for name in FOLDERS + ROOT_FILES:
        check((ROOT / name).exists(), f'Missing from the layout: {name}')
    for name in FOLDERS:
        check((ROOT / name / 'README.md').is_file(), f'Folder without an index: {name}/README.md')
    serve = (ROOT / 'bench/serve.sh').read_text(encoding='utf-8')
    served = {k: (m, r) for k, m, r in re.findall(
        r'^\s*([a-z0-9-]+)\)\s*\n\s*MODEL=(\S+)\s*\n\s*REVISION=([0-9a-f]{40})', serve, re.M)}
    matrix = (ROOT / 'bench/run_matrix.py').read_text(encoding='utf-8')
    measured = {k: (m, r) for k, m, r in re.findall(
        r'"([a-z0-9-]+)": dict\(model="([^"]+)",\s*revision="([0-9a-f]{40})"', matrix)}
    check(bool(served) and served == measured, 'bench/serve.sh and bench/run_matrix.py pin different models')
    check({mk for mk, _, _ in MODELS} == set(served), 'Published models differ from the launchable keys')
    target = (ROOT / 'target.md').read_text(encoding='utf-8')
    index = (ROOT / 'reproduce/README.md').read_text(encoding='utf-8')
    described = (ROOT / 'docs/models.md').read_text(encoding='utf-8')
    for key, (model, revision) in served.items():
        page = ROOT / 'reproduce' / f'{key}.md'
        check(page.is_file(), f'Model without a reproduce page: reproduce/{key}.md')
        if page.is_file():
            body = page.read_text(encoding='utf-8')
            check(model in body and revision in body, f'reproduce/{key}.md does not state the pinned checkpoint')
        check(f'({key}.md)' in index, f'reproduce/README.md does not list {key}')
        check(model in target and revision in target, f'target.md does not pin {key}')
        check(f'`{key}`' in described and revision in described, f'docs/models.md does not cover {key}')
    for page in (ROOT / 'reproduce').glob('*.md'):
        check(page.stem in served or page.name == 'README.md', f'Reproduce page for an unknown key: {page.name}')
    return len(served)


def main():
    errors = []
    def check(ok, msg):
        if not ok:
            errors.append(msg)
    pinned = layout(check)
    ledger = json.loads((ROOT / LEDGER).read_text(encoding='utf-8'))
    records = {r['path']: r for r in ledger['files']}
    check(len(records) == len(ledger['files']), 'Duplicate provenance paths')
    for rel, r in records.items():
        p = ROOT / rel
        check(p.is_file(), f'Missing evidence: {rel}')
        if p.is_file():
            check(digest(p) == r['export_sha256'], f'Evidence hash mismatch: {rel}')
            if r['operation'] == 'copied byte-for-byte':
                check(r['source_sha256'] == r['export_sha256'], f'Undocumented edit: {rel}')
    refs = json.loads((ROOT / ARCHIVE / 'provenance.json').read_text(encoding='utf-8'))
    archived = {r['path'] for r in refs['files']} | {f'{ARCHIVE}/{name}' for name in ARCHIVE_INDEX}
    # Everything under reports/ is hashed evidence, a generated table, the archive or an index.
    registered = set(records) | generate_paths | archived | {LEDGER, 'reports/README.md'}
    for p in (ROOT / 'reports').rglob('*'):
        if p.is_file():
            check(p.relative_to(ROOT).as_posix() in registered, f'Unregistered evidence: {p.relative_to(ROOT)}')

    # Original per-model curation ledgers remain intact; their published hashes still apply.
    for p in (ROOT / 'reports').glob('*/data/*/CURATION.json'):
        doc = json.loads(p.read_text(encoding='utf-8'))
        for r in doc['files']:
            target = ROOT / r['published']
            check(target.is_file(), f'Missing curated artifact: {r["published"]}')
            if target.is_file():
                check(digest(target) == r['published_sha256'], f'Curation hash mismatch: {r["published"]}')

    # Separate September GLM export: preserve source hashes and verify its 42-point CSV.
    for r in refs['files']:
        p = ROOT / r['path']
        check(p.is_file() and digest(p) == r['export_sha256'], f'Reference hash: {r["path"]}')
        if p.suffix == '.json':
            check(r['source_sha256'] == r['export_sha256'], f'Reference JSON changed: {r["path"]}')
    with (ROOT / ARCHIVE / 'results.csv').open(encoding='utf-8', newline='') as f:
        refrows = list(csv.DictReader(f))
    check(len(refrows) == 42, 'Historical GLM point count')
    for r in refrows:
        raw = json.loads((ROOT / r['source_json']).read_text(encoding='utf-8'))
        check(raw['completed'] == raw['num_prompts'] and raw['failed'] == 0, f'Incomplete reference: {r["source_json"]}')
        check(math.isclose(raw['output_throughput'], raw['total_output_tokens'] / raw['duration'], rel_tol=1e-6), f'Reference throughput: {r["source_json"]}')
        for key in raw.keys() & r.keys():
            check(r[key] == str(raw[key]), f'Reference CSV mismatch: {r["source_json"]}: {key}')

    count = 0
    expected = {FIRST: {'v4-0731': 18, 'v41': 18, 'mimo-v26': 18, 'glm-53': 18},
                SECOND: {'qwen-38-bf16': 18, 'mimo-v26': 6}}
    for study, models in expected.items():
        with (ROOT / 'reports' / study / 'serving.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        check(dict(collections.Counter(r['model'] for r in rows)) == models, f'Selection counts: {study}')
        seen = set()
        for r in rows:
            identity = (r['model'], r['bucket'], r['concurrency'], r['block'])
            check(identity not in seen, f'Duplicate timing identity: {identity}')
            seen.add(identity)
            raw = ROOT / 'reports' / study / 'data' / r['model'] / Path(r['evidence']).relative_to(f'results/{study}/{r["model"]}')
            summary = json.loads((raw / 'summary.json').read_text(encoding='utf-8'))
            validation = json.loads((raw / 'blog_validation.json').read_text(encoding='utf-8'))
            bench_path = raw / 'bench.json'
            opener = open
            if not bench_path.exists():
                bench_path = raw / 'bench.json.gz'
                opener = gzip.open
            with opener(bench_path, 'rt', encoding='utf-8') as f:
                bench = json.load(f)
            n = int(r['n'])
            check(r['valid'] == 'True' and summary['valid'] and validation['valid'], f'Invalid timing: {raw.relative_to(ROOT)}')
            check(bench['completed'] == n and bench['failed'] == 0 and bench['num_prompts'] == n, f'Completions: {identity}')
            check(bench['total_output_tokens'] == n * 256 and bench['output_lens'] == [256] * n, f'Output policy: {identity}')
            check(not validation['prefix_caching'] and not validation['speculation'], f'Deployment flags: {identity}')
            check(validation['runtime']['vllm'] == r['runtime'] == '0.31.1rc1.dev50+g554340f3d', f'Runtime: {identity}')
            for col, key in [('output_tok_s', 'output_throughput'), ('ttft_p50_ms', 'median_ttft_ms'),
                             ('tpot_p50_ms', 'median_tpot_ms'), ('ttft_mean_ms', 'mean_ttft_ms'),
                             ('tpot_mean_ms', 'mean_tpot_ms'), ('e2el_mean_ms', 'mean_e2el_ms'),
                             ('e2el_p50_ms', 'median_e2el_ms')]:
                check(math.isclose(float(r[col]), summary[key], rel_tol=1e-12), f'Summary mismatch: {identity}: {col}')
                check(math.isclose(float(r[col]), bench[key], rel_tol=1e-12), f'Raw mismatch: {identity}: {col}')
            check(math.isclose(float(r['output_tok_s']), n * 256 / bench['duration'], rel_tol=1e-12), f'Throughput arithmetic: {identity}')
            manifest = json.loads((ROOT / 'reports' / study / 'study/_inputs/manifest.json').read_text(encoding='utf-8'))
            split = manifest['buckets'][r['bucket']]['splits']['timing']
            check(validation['list_file_sha256'] == split['file_sha256'] and r['list_sha256'] == split['file_sha256'][:16], f'Input list: {identity}')
            check(validation['request_sha256'] == [x['sha256'] for x in split['requests'][:n]], f'Request pairing: {identity}')
            count += 1
        for kind in ('memory', 'components'):
            with (ROOT / 'reports' / study / (kind + '.csv')).open(encoding='utf-8', newline='') as f:
                diagnostic = list(csv.DictReader(f))
            check(set(r['model'] for r in diagnostic) == set(models), f'Diagnostic model scope: {study}/{kind}')
            for mk in models:
                rs = [r for r in diagnostic if r['model'] == mk]
                if kind == 'memory':
                    check(len(rs) == 6, f'Snapshot count: {study}/{mk}')
                    for r in rs:
                        check(math.isclose(float(r['live_gib_per_rank']), float(r['kv_usage_frac']) * float(r['pool_gib_per_rank']), rel_tol=1e-9), f'Memory arithmetic: {study}/{mk}')
                else:
                    check(len(set(r['capture'] for r in rs)) == 6, f'Trace count: {study}/{mk}')
                    groups = collections.defaultdict(set)
                    for r in rs:
                        groups[r['capture'], r['group']].add(int(r['rank_index']))
                    check(all(ranks == set(range(8)) for ranks in groups.values()), f'Rank coverage: {study}/{mk}')

    for rel, body in generate().items():
        p = ROOT / rel
        check(p.is_file() and p.read_text(encoding='utf-8') == body, f'Stale generated output: {rel}')
    errors.extend(page_check())  # numbers embedded in index.html against the same CSVs

    links, pages = 0, {}
    forbidden = re.compile(r'Qwen/Qwen3\.8-Flash-Next-FP8|qwen3\.8-flash-next-fp8|qwen-38(?!-bf16)[/\s"\x27:,)]', re.I)
    credential = re.compile(r'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    private = re.compile(r'/prj/|/home/[A-Za-z0-9_.-]+/|/tmp/claude-[A-Za-z0-9]+|[A-Z]:\\')
    total = 0
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ('.git', '.audit', '.worktrees', '__pycache__')):
            continue
        rel = p.relative_to(ROOT).as_posix()
        total += p.stat().st_size
        check(p.stat().st_size < 10 * 1024 * 1024, f'Large publication file: {rel}')
        if p.name == 'provenance.json':
            continue  # contains hashes/path-only exclusion inventory, never excluded measurements
        try:
            body = gzip.open(p, 'rt', encoding='utf-8').read() if p.suffix == '.gz' else p.read_text(encoding='utf-8')
        except UnicodeError:
            continue
        if rel != 'tools/audit_references.py':
            check(not forbidden.search(body), f'Out-of-scope deployment: {rel}')
        check(not credential.search(body), f'Possible credential: {rel}')
        path_body = body
        if p.name in ('bench.json', 'bench.json.gz'):
            doc = json.loads(body)
            doc.pop('generated_texts', None)
            path_body = json.dumps(doc)
        if rel != 'tools/audit_references.py':
            check(not private.search(path_body), f'Private path: {rel}')
        check(not re.search(r'^host:\s*(?!<HOST>\s*$)\S+', body, re.M), f'Private host: {rel}')
        if p.suffix in ('.md', '.html'):
            body = re.sub(r'```.*?```', '', body, flags=re.S)
            targets = re.findall(r'\]\(([^)]+)\)', body) if p.suffix == '.md' else re.findall(r'(?:href|src)="([^"]+)"', body)
            for target in targets:
                if re.match(r'[a-zA-Z]+:', target):
                    continue
                links += 1
                path, _, fragment = target.partition('#')
                dest = p.parent / path if path else p
                check(dest.exists(), f'Broken local link: {rel}: {target}')
                if fragment and dest.is_file() and dest.suffix in ('.md', '.html'):
                    if dest not in pages:
                        pages[dest] = anchors(dest, dest.read_text(encoding='utf-8'))
                    check(fragment in pages[dest], f'Broken anchor: {rel}: {target}')
    if errors:
        print('FAILED:\n' + '\n'.join('- ' + e for e in errors))
        return 1
    print(f'PASS: {len(records)} provenance files; {count} timing runs (90 main + 6 control); {pinned} pinned models; '
          f'{links} local links and anchors; {total / 1024**2:.2f} MiB.')
    print('Exact-byte evidence, curation ledgers, arithmetic, inputs, diagnostic counts, generated tables, '
          'article data, layout, model pins and publication scan passed.')
    print('This validates the export, not causal control, answer quality or a fresh GPU reproduction.')
    return 0


generate_paths = set(generate())
if __name__ == '__main__':
    sys.exit(main())
