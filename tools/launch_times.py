"""Print how long each recorded launch took and what memory it reported; no GPU required.

Reads the saved server logs of the two October studies (reports/<study>/data/<key>/_server/) and
the live-memory CSVs. It writes nothing. The budget tables in reproduce/README.md come from here.

  python tools/launch_times.py

`ready` is the launch command to "Application startup complete". `up` is the launch command to the
server's last log line, so it includes the measurements and the shutdown. A launch with no `ready`
did not start: it is kept, because a failed first launch is part of the time a session needs.
"""
import csv
import datetime as dt
import gzip
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STUDIES = (('blog-architecture-h100-v1', 'first'), ('qwen-bf16-h100-v1', 'second'))
STAMP = re.compile(r'\b(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b')
LINES = dict(weights=r'Model loading took ([\d.]+) GiB', pool=r'Available KV cache memory: ([\d.]+) GiB')


def lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        return f.read().splitlines()


def launches():
    for study, node in STUDIES:
        for path in sorted((ROOT / 'reports' / study / 'data').glob('*/_server/serve-*')):
            config, day, clock = re.search(r'serve-(off(?:-profidle)?)-(\d{8})-(\d{6})', path.name).groups()
            start = dt.datetime.strptime(day + clock, '%Y%m%d%H%M%S')

            def stamp(line):
                found = STAMP.search(line)
                try:
                    return dt.datetime(start.year, *map(int, found.groups())) if found else None
                except ValueError:
                    return None

            body = lines(path)
            stamps = [t for t in map(stamp, body) if t and t >= start - dt.timedelta(minutes=1)]
            done = next((i for i, line in enumerate(body) if 'Application startup complete' in line), None)
            ready = None if done is None else next((t for t in map(stamp, reversed(body[:done + 1])) if t), None)
            minutes = lambda t: None if t is None else (t - start).total_seconds() / 60
            row = dict(node=node, key=path.parts[-3], kind='diagnostics' if config.endswith('profidle') else 'timing',
                       start=start, ready=minutes(ready), up=minutes(max(stamps)) if stamps else None,
                       failed=any('CUDA error' in line for line in body) or done is None)
            for name, pattern in LINES.items():
                row[name] = next((float(m.group(1)) for m in map(re.compile(pattern).search, body) if m), None)
            yield row


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(out + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    runs = sorted(launches(), key=lambda r: (r['node'], r['start']))
    show = lambda v, digits=1: 'n/a' if v is None else f'{v:.{digits}f}'
    print('## Every recorded launch\n')
    print(table(['Node', 'Started', 'Key', 'Step', 'Ready min', 'Up min', 'Result'],
                [[r['node'], f'{r["start"]:%m-%d %H:%M}', r['key'], r['kind'], show(r['ready']), show(r['up']),
                  'failed' if r['failed'] else 'ok'] for r in runs]))
    print('\n## Per model and node (successful launches)\n')
    rows = []
    for node, key in dict.fromkeys((r['node'], r['key']) for r in runs):
        ok = [r for r in runs if (r['node'], r['key']) == (node, key) and not r['failed']]
        timing = [r['up'] for r in ok if r['kind'] == 'timing']
        diag = [r['up'] for r in ok if r['kind'] == 'diagnostics']
        first = min(ok, key=lambda r: r['start'])
        again = [r['ready'] for r in ok if r is not first]
        rows.append([node, key, show(first['ready']), ' / '.join(show(v) for v in again) or 'n/a',
                     ' / '.join(show(v) for v in diag) or 'n/a', ' / '.join(show(v) for v in timing) or 'n/a',
                     show(sum(timing)) if timing else 'n/a', show(first['weights'], 2), show(first['pool'], 2)])
    print(table(['Node', 'Key', 'First launch ready', 'Relaunch ready', 'Diagnostics up', 'Timing blocks up',
                 'Timing total', 'Weights GiB/GPU', 'KV pool GiB/GPU'], rows))
    print('\n## GPU memory in use during the 64K, one-request snapshot (nvidia-smi)\n')
    rows = []
    for study, node in STUDIES:
        with (ROOT / 'reports' / study / 'memory.csv').open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                if (r['bucket'], r['live_sequences']) == ('64k', '1'):
                    rows.append([node, r['model'], f'{int(r["gpu_mem_mib_rank0"]) / 1024:.1f}',
                                 f'{int(r["gpu_mem_mib_sum"]) / 1024:.0f}'])
    print(table(['Node', 'Key', 'GPU 0 GiB', 'Eight GPUs GiB'], rows))
    for node in ('first', 'second'):
        mine = [r for r in runs if r['node'] == node and r['up'] is not None]
        last = max(mine, key=lambda r: r['start'])
        elapsed = (last['start'] - min(r['start'] for r in runs if r['node'] == node)).total_seconds() / 3600 + last['up'] / 60
        print(f'\n{node.capitalize()} node: {len(mine)} launches with a log, server up {sum(r["up"] for r in mine) / 60:.1f} h, '
              f'{elapsed:.1f} h from the first launch to the last shutdown.')


if __name__ == '__main__':
    main()
