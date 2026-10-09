"""Keep the numbers inside index.html equal to the retained CSVs; no GPU required.

The article draws its interactive figures from one JSON block, <script id="study-data">.
This tool recomputes that block from reports/<study>/{serving,components,memory}.csv.

  python tools/build_blog_page.py --check   # verify the block and two static tables; change nothing
  python tools/build_blog_page.py           # rewrite the block in index.html

`runtime`, `models`, `componentDefs` and `studies` inside the block are presentation choices and are
kept as written: to show another model, add it to `models` and rerun. Prose, the hero chart and the
other static tables are edited by hand; --check covers the block, #throughput-table and #bf16-latency.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'index.html'
BLOCK = re.compile(r'(<script id="study-data" type="application/json">)(.*?)(</script>)', re.S)
POINTS = [(b, c) for b in ('1k', '16k', '64k') for c in (1, 8)]
METRICS = ('output_tok_s', 'ttft_p50_ms', 'tpot_p50_ms', 'e2el_p50_ms')
CAPTURES = ('prefill16k', 'prefill64k', 'decode1k_B1', 'decode64k_B1', 'decode1k_B8', 'decode64k_B8')
KERNELS = ('span_ms', 'kernel_sum_ms', 'attention_core_ms', 'recurrent_attention_ms', 'indexer_topk_ms',
           'qkv_rope_kvcache_ms', 'dense_gemm_ms', 'moe_expert_gemm_ms', 'moe_route_combine_ms',
           'mhc_residual_norm_ms', 'allreduce_ms', 'allgather_other_comm_ms', 'gemm_input_prep_ms',
           'engram_ms', 'other_elementwise_ms')
# Derived columns of the component figure: attention work without projections, and the rest.
SUMS = {'attention_path_ms': ('attention_core_ms', 'recurrent_attention_ms', 'indexer_topk_ms', 'qkv_rope_kvcache_ms'),
        'remaining_ms': ('gemm_input_prep_ms', 'engram_ms', 'other_elementwise_ms')}
MEMORY = {'live_tokens': 'live_tokens_client_content_plus_generated', 'live_gib_8_ranks': 'live_gib_8_ranks',
          'live_kib_per_token_per_rank': 'live_kib_per_token_per_rank', 'pool_gib_per_rank': 'pool_gib_per_rank'}


def read(study, name):
    with (ROOT / 'reports' / study / (name + '.csv')).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def page_block():
    with PAGE.open(encoding='utf-8', newline='') as f:
        html = f.read()
    found = BLOCK.search(html)
    if not found:
        raise SystemExit('index.html has no <script id="study-data"> block')
    return html, found, json.loads(found.group(2))


def number(text):
    value = float(text)
    return int(value) if value.is_integer() else value


def derive(block):
    """The serving, component and memory rows for the models the block lists."""
    data = {s: {name: read(s, name) for name in ('serving', 'components', 'memory')} for s in block['studies']}
    serving, components, memory = [], [], []
    for model in block['models']:
        mk, rows = model['key'], data[model['study']]
        for bucket, clients in POINTS:
            runs = sorted([r for r in rows['serving'] if (r['model'], r['bucket'], r['concurrency'], r['valid']) ==
                           (mk, bucket, str(clients), 'True')], key=lambda r: int(r['block']))
            assert len(runs) == 3 and len({r['n'] for r in runs}) == 1, (mk, bucket, clients, len(runs))
            row = dict(model=mk, bucket=bucket, concurrency=clients, requests=int(runs[0]['n']),
                       prompt_tokens=st.mean(float(r['server_prompt_tokens_per_request']) for r in runs))
            for metric in METRICS:
                v = [float(r[metric]) for r in runs]
                row[metric] = dict(mean=st.mean(v), sd=st.stdev(v), median=st.median(v), blocks=v)
            serving.append(row)
        for capture in CAPTURES:
            group = 'last_full_chunk' if capture.startswith('prefill') else 'decode_steps'
            ranks = [r for r in rows['components'] if (r['model'], r['capture'], r['group']) == (mk, capture, group)]
            assert len(ranks) == 8, (mk, capture, len(ranks))
            row = dict(model=mk, capture=capture, engine_B=int(float(ranks[0]['engine_B'])))
            for key in KERNELS:  # an empty cell means "not available", never zero
                row[key] = st.mean(float(r[key]) for r in ranks) if all(r[key] for r in ranks) else None
            for key, parts in SUMS.items():
                row[key] = sum(row[p] for p in parts if row[p] is not None)
            components.append(row)
        for r in rows['memory']:
            if r['model'] == mk:
                row = dict(model=mk, bucket=r['bucket'], live_sequences=int(r['live_sequences']))
                row.update({key: number(r[src]) for key, src in MEMORY.items()})
                memory.append(row)
    hashes = {f'reports/{s}/{name}.csv': hashlib.sha256((ROOT / 'reports' / s / f'{name}.csv').read_bytes()).hexdigest()
              for s in block['studies'] for name in ('serving', 'components', 'memory')}
    return dict(serving=serving, components=components, memory=memory, source_sha256=hashes)


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def cells(html, table_id):
    table = re.search(r'<table[^>]*id="%s".*?</table>' % table_id, html, re.S)
    if not table:
        return None
    body = table.group(0)
    text = lambda cell: re.sub(r'<[^>]+>', '', cell).strip()
    head = [text(c) for c in re.findall(r'<th scope="col">(.*?)</th>', body, re.S)]
    rows = [(text(th), re.findall(r'<td>(.*?)</td>', tr, re.S))
            for th, tr in re.findall(r'<tr><th scope="row">(.*?)</th>(.*?)</tr>', body, re.S)]
    return head, rows


def point(label):
    bucket, clients = re.fullmatch(r'(\d+K)\W+c(\d+)', label).groups()
    return bucket.lower(), int(clients)


def check():
    """Return a list of problems; empty when the page agrees with the CSVs."""
    html, _, block = page_block()
    derived, errors = derive(block), []
    identity = dict(serving=('model', 'bucket', 'concurrency'), components=('model', 'capture'),
                    memory=('model', 'bucket', 'live_sequences'))
    for name, keys in identity.items():
        want = {tuple(r[k] for k in keys): r for r in derived[name]}
        have = {tuple(r[k] for k in keys): r for r in block[name]}
        for key in sorted(set(want) | set(have), key=str):
            if key not in have or key not in want or not same(want[key], have[key]):
                errors.append(f'index.html {name} row differs from the CSVs: {key}')
    ledger = {r['path']: r for r in json.loads((ROOT / 'reports/provenance.json').read_text(encoding='utf-8'))['files']}
    for path, current in derived['source_sha256'].items():
        # A row selection keeps the values of its source file, so either hash identifies them.
        if block['source_sha256'].get(path) not in (current, ledger[path]['source_sha256']):
            errors.append(f'index.html source hash is neither the file nor its ledger source: {path}')
    stat = {(r['model'], r['bucket'], r['concurrency']): r for r in derived['serving']}
    short = {m['short']: m['key'] for m in block['models']}
    table = cells(html, 'throughput-table')
    if not table:
        errors.append('index.html has no #throughput-table')
    else:
        head, rows = table
        for label, tds in rows:
            values = [stat[(short[name],) + point(label)]['output_tok_s'] for name in head[1:]]
            best = max(range(len(values)), key=lambda i: values[i]['mean'])
            for i, (name, td, v) in enumerate(zip(head[1:], tds, values)):
                if re.sub(r'<[^>]+>', '', td).strip() != f'{v["mean"]:.1f} ± {v["sd"]:.1f}':
                    errors.append(f'index.html #throughput-table {label} {name} is not the CSV mean ± SD')
                if ('class="winner"' in td) != (i == best):
                    errors.append(f'index.html #throughput-table {label} {name}: winner mark is wrong')
    qwen = next((m['key'] for m in block['models'] if m['node'] == 'second'), None)
    table = cells(html, 'bf16-latency')
    if not table:
        errors.append('index.html has no #bf16-latency table')
    else:
        for label, tds in table[1]:
            r = stat[(qwen,) + point(label)]
            want = [f'{r["ttft_p50_ms"]["mean"]:.0f} ± {r["ttft_p50_ms"]["sd"]:.0f}',
                    f'{r["tpot_p50_ms"]["mean"]:.2f}', f'{r["e2el_p50_ms"]["mean"]:.0f}']
            if [td.strip() for td in tds] != want:
                errors.append(f'index.html #bf16-latency {label} is not the CSV value')
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='verify only; exit 1 if the page is stale')
    args = ap.parse_args()
    if not args.check:
        html, found, block = page_block()
        block.update(derive(block))
        body = json.dumps(block, ensure_ascii=False, separators=(',', ':'))
        assert '</' not in body
        with PAGE.open('w', encoding='utf-8', newline='') as f:
            f.write(html[:found.start(2)] + body + html[found.end(2):])
    errors = check()
    if errors:
        raise SystemExit('Stale article data:\n' + '\n'.join('- ' + e for e in errors))
    block = page_block()[2]
    print(('Checked' if args.check else 'Wrote and checked') + f' index.html: {len(block["serving"])} serving, '
          f'{len(block["components"])} component and {len(block["memory"])} memory rows, '
          'the throughput table and the Qwen latency table agree with the CSVs.')


if __name__ == '__main__':
    main()
