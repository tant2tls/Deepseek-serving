"""Score the twelve natural-ending answers of each model in the rerun session.

  python tools/check_natural_answers.py          # write reports/five-model-rerun-h100-v1/natural_outcomes.csv

Standard library only. It reads the published functional.json of each model and applies fixed checks:
math answers against a frozen number, code answers against frozen unit tests, chat answers against a
structural rule taken from the prompt. **Code answers are executed**, each in its own short-lived
subprocess in an empty temporary directory with CPU, memory and file-size limits and, where the system
allows it, without network. Run it on a machine where that is acceptable; the audit never runs it.

A small regression check: twelve tasks do not rank the models' answer quality.
"""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'five-model-rerun-h100-v1'
BASE = ROOT / 'reports' / STUDY
EVALUATOR = 'check_natural_answers.py v1 (2026-10-10)'
MATH = {'average_speed': 50, 'divisors_360': 24, 'linear_equation': 15, 'sum_multiples': 9368}
TESTS = {
    'merge_intervals': '''
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([]) == []
assert merge_intervals([[5, 7], [1, 2], [2, 3]]) == [[1, 3], [5, 7]]
assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
''',
    'lru_cache': '''
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c.put(3, 30)
assert c.get(3) == 30
''',
    'to_roman': '''
for n, r in ((1, 'I'), (4, 'IV'), (9, 'IX'), (14, 'XIV'), (40, 'XL'), (90, 'XC'), (400, 'CD'), (1994, 'MCMXCIV'), (2024, 'MMXXIV'), (3999, 'MMMCMXCIX')):
    assert to_roman(n) == r, (n, to_roman(n))
''',
    'top_k_words': '''
assert [tuple(x) for x in top_k_words("the cat and the hat and the bat", 2)] == [("the", 3), ("and", 2)]
assert [tuple(x) for x in top_k_words("B a b A c", 2)] == [("a", 2), ("b", 2)]
assert [tuple(x) for x in top_k_words("x-ray x_ray x", 1)] == [("x", 3)]
assert list(top_k_words("", 3)) == []
''',
}


def code_of(text):
    """The answer's Python source: the fenced blocks if there are any, else the whole text."""
    blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', text, flags=re.S)
    return '\n\n'.join(blocks) if blocks else text


def limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run_code(task, text):
    work = tempfile.mkdtemp(prefix='natural-check-')
    try:
        (Path(work) / 'answer_test.py').write_text(code_of(text) + '\n\n' + TESTS[task] + '\nprint("PASS")\n', encoding='utf-8')
        cmd = [sys.executable, '-I', '-S', 'answer_test.py']
        if shutil.which('unshare') and subprocess.run(['unshare', '-n', 'true'], capture_output=True).returncode == 0:
            cmd = ['unshare', '-n'] + cmd  # no network inside the check
        try:
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=30, preexec_fn=limits,
                               env={'PATH': os.environ.get('PATH', '')})
        except subprocess.TimeoutExpired:
            return False, 'timeout', cmd[0] == 'unshare'
        ok = r.returncode == 0 and r.stdout.strip().endswith('PASS')
        return ok, '' if ok else (r.stderr.strip().splitlines() or ['no output'])[-1][:160], cmd[0] == 'unshare'
    finally:
        shutil.rmtree(work, ignore_errors=True)


def chat_rule(task, text):
    t = text.strip()
    if task == 'hash_table':
        words = len(t.split())
        return t.splitlines()[-1].strip() == 'DONE' and 200 <= words <= 420, f'{words} words; last line {t.splitlines()[-1].strip()[:20]!r}'
    if task == 'interview_steps':
        steps = re.findall(r'(?m)^\s*(?:\*\*)?(?:Step\s+)?(\d+)[.):]', t)
        return steps == ['1', '2', '3', '4', '5'], f'numbered lines {steps}'
    if task == 'cars_essay':
        paras = [p for p in re.split(r'\n\s*\n', t) if len(p.split()) >= 15]
        return len(paras) == 4, f'{len(paras)} paragraphs of 15 words or more'
    if task == 'ww1_bullets':
        bullets = [l for l in t.splitlines() if l.startswith('- ')]
        other = [l for l in t.splitlines() if l.strip() and not l.startswith('- ')]
        return len(bullets) == 3 and not other, f'{len(bullets)} bullets, {len(other)} other lines'
    return None, 'no rule'


def repetition(text):
    """Share of the answer's word 8-grams that repeat an earlier one (0 = none)."""
    w = text.split()
    grams = [' '.join(w[i:i + 8]) for i in range(max(0, len(w) - 7))]
    return round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0


def main():
    rows = []
    for p in sorted(BASE.glob('data/*/_functional/functional.json')):
        doc = json.loads(p.read_text(encoding='utf-8'))
        for c in doc['checks']:
            if c['kind'] != 'natural_eos':
                continue
            text, task = c.get('text') or '', c['task']
            sandbox = ''
            if c['domain'] == 'math':
                last = [l for l in text.strip().splitlines() if l.strip()][-1:] or ['']
                found = re.findall(r'-?\d+(?:\.\d+)?', last[0].replace(',', ''))
                ok, detail = bool(found) and float(found[-1]) == MATH[task], f'last line {last[0].strip()[:60]!r}; expected {MATH[task]}'
            elif c['domain'] == 'code':
                ok, detail, isolated = run_code(task, text)
                sandbox = 'subprocess, limits, no network' if isolated else 'subprocess, limits'
            else:
                ok, detail = chat_rule(task, text)
            usage = c.get('usage') or {}
            rows.append(dict(study=STUDY, model=doc['model_key'], domain=c['domain'], task=task, passed=ok, detail=detail,
                             finish_reason=c.get('finish_reason'), truncated=c.get('finish_reason') == 'length',
                             completion_tokens=usage.get('completion_tokens'),
                             reasoning_chars=len(c.get('reasoning') or ''), answer_chars=len(text),
                             repeated_8gram_share=repetition(text), sandbox=sandbox, evaluator=EVALUATOR,
                             evidence=p.relative_to(ROOT).as_posix()))
    out = BASE / 'natural_outcomes.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    print('wrote', out.relative_to(ROOT), len(rows), 'rows')
    for mk in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == mk]
        print(f'  {mk}: {sum(bool(r["passed"]) for r in rs)}/{len(rs)} passed;',
              ', '.join(f'{r["task"]}: {r["detail"]}' for r in rs if not r['passed']))


if __name__ == '__main__':
    main()
