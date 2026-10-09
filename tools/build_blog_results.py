"""Regenerate the five-model publication tables from curated CSVs; no GPU required."""
import argparse
import csv
import io
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
FIRST = 'blog-architecture-h100-v1'
SECOND = 'qwen-bf16-h100-v1'
BUILD = '554340f3d3259e321be4c07282be7a02a5aeef83'
MODELS = [('v4-0731', '0731', FIRST), ('v41', 'V4.1', FIRST),
          ('mimo-v26', 'MiMo', FIRST), ('qwen-38-bf16', 'Qwen', SECOND), ('glm-53', 'GLM', FIRST)]
POINTS = [(b, str(c)) for b in ('1k', '16k', '64k') for c in (1, 8)]


def read(study, filename):
    with (ROOT / 'reports' / study / filename).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def csv_text(rows):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def generate():
    data = {s: {name: read(s, name + '.csv') for name in ('serving', 'memory', 'components')}
            for s in (FIRST, SECOND)}
    def runs(s, mk, b, c):
        return sorted([r for r in data[s]['serving'] if (r['model'], r['bucket'], r['concurrency'], r['valid']) ==
                       (mk, b, c, 'True')], key=lambda r: int(r['block']))
    def values(s, mk, b, c, metric):
        return [float(r[metric]) for r in runs(s, mk, b, c)]
    def cell(v, digits=1):
        return f'{st.mean(v):.{digits}f} ± {st.stdev(v):.{digits}f}' if len(v) > 1 else f'{v[0]:.{digits}f}'

    selected = [r for mk, _, s in MODELS for r in data[s]['serving'] if r['model'] == mk]
    assert len(selected) == 90 and all(r['valid'] == 'True' for r in selected)
    output = {'reports/five-model/serving.csv': csv_text(selected)}
    intro = (f'Build: vLLM `{BUILD}` (`0.31.1rc1.dev50+g554340f3d`). '
             '0731, V4.1, MiMo and GLM are first-node measurements; Qwen is second-node. '
             'See [protocol and limits](../../docs/experiments.md). All values below are generated from the retained CSVs.\n')
    out = ['# Five-model measured results\n', intro, '## Output throughput\n',
           'Output tokens/s, mean ± sample SD across three launch-separated blocks. '
           'Qwen uses the original BF16 checkpoint. This is a throughput-only comparison across nodes; '
           'values are not rescaled by the control. Differences below about 2% across nodes remain unresolved.\n']
    out.append(table(['Input', 'Clients'] + [n for _, n, _ in MODELS],
                     [[b.upper(), c] + [cell(values(s, mk, b, c, 'output_tok_s')) for mk, _, s in MODELS] for b, c in POINTS]))
    out += ['Qwen leads at five of the six measured points; MiMo leads at 1K/c1. '
            'This ranks the saved deployments and workloads, not model quality or architecture in isolation.\n',
            '## MiMo node control\n',
            'One block on the second node against the mean of three on the first. TTFT and TPOT columns '
            'are ratios of block-level per-request medians, not pooled latency percentiles.\n']
    control = []
    for b, c in POINTS:
        row = {'bucket': b, 'concurrency': c}
        for k in ('output_tok_s', 'ttft_p50_ms', 'tpot_p50_ms'):
            row[k + '_second_over_first'] = st.mean(values(SECOND, 'mimo-v26', b, c, k)) / st.mean(values(FIRST, 'mimo-v26', b, c, k))
        control.append(row)
    output['reports/five-model/node_control.csv'] = csv_text(control)
    out.append(table(['Input', 'Clients', 'Throughput ratio', 'TTFT ratio', 'TPOT ratio'],
                     [[r['bucket'].upper(), r['concurrency']] + [f'{v:.3f}' for k, v in r.items() if k.endswith('_first')] for r in control]))
    out += ['The control supports cross-node throughput comparison, but not TTFT or eight-client TPOT rankings.\n',
            '## Latency on the first node\n', 'Each cell is mean ± sample SD of the three block medians, in milliseconds.\n']
    for metric, label, digits in [('ttft_p50_ms', 'TTFT', 0), ('tpot_p50_ms', 'TPOT', 2)]:
        out.append(f'### {label}\n')
        out.append(table(['Input', 'Clients'] + [n for _, n, s in MODELS if s == FIRST],
                         [[b.upper(), c] + [cell(values(s, mk, b, c, metric), digits) for mk, _, s in MODELS if s == FIRST] for b, c in POINTS]))
    out += ['## Qwen latency on the second node\n',
            'Reported separately. Short-prompt TTFT can switch between two levels within one run. '
            'At c8, TPOT includes scheduling interference from other requests and is not an isolated decode step.\n']
    out.append(table(['Input', 'Clients', 'TTFT ms', 'TPOT ms'],
                     [[b.upper(), c, cell(values(SECOND, 'qwen-38-bf16', b, c, 'ttft_p50_ms'), 0),
                       cell(values(SECOND, 'qwen-38-bf16', b, c, 'tpot_p50_ms'), 2)] for b, c in POINTS]))
    out += ['## Live state memory at 64K, B=1\n',
            'Approximation from usage gauge × per-rank pool. GiB and KiB are binary units. '
            'Eight-rank bytes include replication and allocation rounding; these are not measured maximum request capacities.\n']
    memrows = []
    for mk, name, s in MODELS:
        r, = [r for r in data[s]['memory'] if (r['model'], r['bucket'], r['live_sequences']) == (mk, '64k', '1')]
        memrows.append([name, 'Second' if s == SECOND else 'First'] + [f'{float(r[k]):.2f}' for k in
                       ('live_kib_per_token_per_rank', 'live_gib_8_ranks', 'pool_gib_per_rank')])
    out.append(table(['Model', 'Node', 'Live KiB/token/GPU', 'Live GiB/eight GPUs', 'Reserved pool GiB/GPU'], memrows))
    out += ['## Trace components\n',
            'GPU kernel milliseconds per engine step, averaged across all eight ranks. '
            'Prefill uses the last full 8192-token chunk. Kernels can overlap; sums are not elapsed latency. '
            'Dense GEMM mixes projection work and must not be counted as an exact attention total. '
            'Qwen all-reduce includes profiler-related rank waiting; it is not isolated communication cost.\n']
    columns = [('attention_core_ms', 'Attention core'), ('indexer_topk_ms', 'Indexer'),
               ('recurrent_attention_ms', 'Recurrent'), ('qkv_rope_kvcache_ms', 'KV preparation'),
               ('dense_gemm_ms', 'Dense GEMM'), ('moe_expert_gemm_ms', 'Experts'),
               ('moe_route_combine_ms', 'Routing/combine'), ('allreduce_ms', 'All-reduce')]
    for capture in ('prefill16k', 'prefill64k', 'decode1k_B1', 'decode1k_B8', 'decode64k_B1', 'decode64k_B8'):
        rows = []
        for mk, name, s in MODELS:
            group = 'last_full_chunk' if capture.startswith('prefill') else 'decode_steps'
            rs = [r for r in data[s]['components'] if (r['model'], r['capture'], r['group']) == (mk, capture, group)]
            assert len(rs) == 8, (mk, capture, len(rs))
            rows.append([name] + [f'{st.mean(float(r[k]) for r in rs):.2f}' if all(r[k] for r in rs) else 'Unavailable' for k, _ in columns])
        out += [f'### {capture}\n', table(['Model'] + [n + ' ms' for _, n in columns], rows)]
    out += ['HBM byte counters and expert-routing statistics were not collected. '
            'The traces identify executed kernels and scaling patterns; they do not isolate causal optimization gains.\n',
            '## Individual timing blocks\n', 'No slow first block is dropped. Full precision and additional metrics are in [serving.csv](serving.csv).\n']
    out.append(table(['Model', 'Input', 'Clients', 'Block 1 tok/s', 'Block 2 tok/s', 'Block 3 tok/s', 'Median tok/s'],
                     [[name, b.upper(), c] + [f'{v:.3f}' for v in values(s, mk, b, c, 'output_tok_s')] +
                      [f'{st.median(values(s, mk, b, c, "output_tok_s")):.3f}'] for mk, name, s in MODELS for b, c in POINTS]))
    out += ['## Evidence\n',
            '- First node: [timing](../blog-architecture-h100-v1/serving.csv), [components](../blog-architecture-h100-v1/components.csv), '
            '[memory](../blog-architecture-h100-v1/memory.csv), [raw evidence](../blog-architecture-h100-v1/data/).\n'
            '- Second node: [timing](../qwen-bf16-h100-v1/serving.csv), [components](../qwen-bf16-h100-v1/components.csv), '
            '[memory](../qwen-bf16-h100-v1/memory.csv), [raw evidence](../qwen-bf16-h100-v1/data/).\n'
            '- [Provenance and sanitization](../../docs/provenance.md); [earlier experiments](../../docs/previous-experiments.md).\n']
    output['reports/five-model/results.md'] = '\n'.join(out)
    for s, title in ((FIRST, 'First-node evidence: 0731, V4.1, MiMo and GLM'), (SECOND, 'Second-node evidence: Qwen and MiMo control')):
        output[f'reports/{s}/findings.md'] = (f'# {title}\n\n{intro}\n'
            'This branch uses a selected evidence view. Read the [five-model results](../five-model/results.md), '
            '[experiment protocol](../../docs/experiments.md) and [per-model guides](../../reproduce/README.md).\n\n'
            'Per-run evidence: [data/](data/). Per-rank diagnostics: [components.csv](components.csv). '
            'Snapshots: [memory.csv](memory.csv). Timing: [serving.csv](serving.csv). '
            'Node inventory: [study/_env/](study/_env/). Input hashes: [manifest](study/_inputs/manifest.json).\n\n'
            'Original frozen plans and full source narratives remain at the immutable parent linked in '
            '[provenance](../../docs/provenance.md). This summary does not change their completion ledgers.\n')
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    errors = []
    output = generate()
    for rel, body in output.items():
        p = ROOT / rel
        if args.check:
            if not p.is_file() or p.read_text(encoding='utf-8') != body:
                errors.append(rel)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding='utf-8', newline='\n')
    if errors:
        raise SystemExit('Stale generated tables: ' + ', '.join(errors))
    print(('Checked' if args.check else 'Generated') + f' {len(output)} files: 90 timing runs and the separate MiMo control.')


if __name__ == '__main__':
    main()
