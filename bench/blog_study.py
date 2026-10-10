#!/usr/bin/env python3
"""Chain for the architecture-blog protocol (target.md, docs/experiments.md): AR only, prefix caching off.

The study is chosen with BLOG_STUDY (default: the completed blog-architecture-h100-v1).
Study qwen-bf16-h100-v1 (2026-10-08) measures the original BF16 Qwen checkpoint with the same
protocol and inputs, plus one MiMo timing block as a node control; see STUDIES below.

  python bench/blog_study.py env                      # record node/runtime
  python bench/blog_study.py plan --counts '{...}'    # freeze plan.json after the pilots
  python bench/blog_study.py dry-run                  # validate plan, inputs, pins, paths
  python bench/blog_study.py chain diag:v4-0731 timing:v4-0731:1 ...

`diag:<model>` launches the idle-profiler server and runs functional checks, a disjoint
pilot, live-KV snapshots and trace captures. `timing:<model>:<block>` launches the plain
server and runs the six unprofiled points of one repeat block. Each step owns its server:
it is started in its own process group, readiness is bounded, and the group is stopped
and GPU memory checked on every exit path. Historical chains are never invoked.
"""
import argparse, datetime, hashlib, json, os, shutil, signal, subprocess, sys, threading, time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm

ROOT = rm.ROOT
BLOG = "blog-architecture-h100-v1"
BLOG_RUNTIME = "0.31.1rc1.dev50+g554340f3d"
# The 15-hour session of the rerun plan (target.md#the-15-hour-session). Its build was resolved
# from the nightly index on 2026-10-10 and frozen before the first launch.
RERUN = "five-model-rerun-h100-v1"
RERUN_RUNTIME = "0.31.1rc1.dev260+ga98247ab4"
RERUN_COMMIT = "a98247ab4db686ee03c66d5feb3c761e52a2f8ab"
_FIVE = ("v4-0731", "mimo-v26", "v41", "glm-53", "qwen-38-bf16")
# Per study: models that may launch, models with a diagnostic launch, timing blocks, where the
# revisions are pinned, and the frozen inputs it must reproduce byte for byte.
STUDIES = {
    # Selected first-node deployments; completed source order is documented in docs/experiments.md.
    BLOG: dict(allowed=("v4-0731", "v41", "mimo-v26", "glm-53"),
               diag=("v4-0731", "v41", "mimo-v26", "glm-53"),
               blocks={"1": ["v4-0731", "mimo-v26", "v41", "glm-53"], "2": ["mimo-v26", "v41", "v4-0731", "glm-53"],
                       "3": ["v41", "v4-0731", "mimo-v26", "glm-53"]},
               pins=("target.md",), inputs_reference=f"reports/{BLOG}/study/_inputs/manifest.json"),
    # Original BF16 Qwen (reproduce/qwen-38-bf16.md). MiMo runs one timing block as a node
    # control against its three blocks of 2026-10-07, and (plan_addendum_mimo_trace_control.json)
    # one diagnostic launch so trace components can be compared across the two nodes as well.
    "qwen-bf16-h100-v1": dict(allowed=("qwen-38-bf16", "mimo-v26"), diag=("qwen-38-bf16", "mimo-v26"),
                              blocks={"1": ["qwen-38-bf16", "mimo-v26"], "2": ["qwen-38-bf16"],
                                      "3": ["qwen-38-bf16"]},
                              pins=("target.md", "reproduce/qwen-38-bf16.md"),
                              inputs_reference=f"reports/{BLOG}/study/_inputs/manifest.json"),
    # All five on one node and one build: 2,048 forced output tokens, 1 and 16 clients, contexts
    # up to 128K in the diagnostics. Each block starts two models later than the one before.
    RERUN: dict(allowed=_FIVE, diag=_FIVE,
                blocks={"1": list(_FIVE), "2": list(_FIVE[2:] + _FIVE[:2]), "3": list(_FIVE[4:] + _FIVE[:4])},
                pins=("target.md",), inputs_reference=None, runtime=RERUN_RUNTIME, osl=2048, loads=(1, 16),
                # 64K has nine auxiliary requests, so its 16-client pilot holds nine at once.
                pilot_n={("1k", 1): 3, ("1k", 16): 16, ("16k", 1): 3, ("16k", 16): 16, ("64k", 1): 3, ("64k", 16): 9},
                warm_n={"1k": 16, "16k": 16, "64k": 3}),
}
RETIRED = ()
STUDY = os.environ.get("BLOG_STUDY", BLOG)
SPEC = STUDIES[STUDY]
SDIR = ROOT / "results" / STUDY
INP = SDIR / "_inputs"
LOGS = SDIR / "_logs"
BASE = rm.BASE
ALLOWED = SPEC["allowed"]
CONFIGS = {"off": "timing", "off-profidle": "diagnostic"}  # both pass --no-enable-prefix-caching
if STUDY == RERUN:  # routed-experts capture switched on; prefix caching still off; never a timing launch
    CONFIGS["off-routed"] = "instrumented"
SERVE_ARGS = ["--max-num-batched-tokens", "8192"]  # explicit chunk budget
BUCKETS = ("16k", "1k", "64k")  # 16K first: the priority comparison
SESSION = STUDY == RERUN        # the rerun session adds contexts, intervals and counter-based progress
OSL = SPEC.get("osl", rm.OSL)   # forced output tokens per request (256 in the completed studies)
LOADS = SPEC.get("loads", (1, 8))
RUNTIME = SPEC.get("runtime", BLOG_RUNTIME)
READY_TIMEOUT_S = 1800
PILOT_N = SPEC.get("pilot_n", {("1k", 1): 6, ("1k", 8): 24, ("16k", 1): 3, ("16k", 8): 12, ("64k", 1): 3, ("64k", 8): 9})
WARM_N = SPEC.get("warm_n", {"1k": 16, "16k": 8, "64k": 3})
CAPTURE_BUCKETS = ("1k", "16k", "64k", "128k")            # session diagnostics: prefill and decode at B=1/8
INTERVALS = (("1k", 8), ("16k", 8), ("64k", 8), ("128k", 8), ("128k", 1))  # session timing launches
INTERVAL_STEPS = 512


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timeline(**rec):
    """One line per step start and end, in UTC; published with the results of each model."""
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "timeline.jsonl", "a") as f:
        f.write(json.dumps(dict(utc=utc(), **rec)) + "\n")


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] CHAIN {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "chain.log", "a") as f:
        f.write(line + "\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def runtime():
    venv = os.environ.get("VLLM_VENV", "/root/vllm")
    out = sh(f"{venv}/bin/python -c \"import vllm,torch;print(vllm.__version__);print(torch.__version__)\"").split()
    return dict(venv=venv, vllm=out[-2] if len(out) >= 2 else None, torch=out[-1] if out else None)


def gpu_mem():
    return [int(x) for x in sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()]


def post(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def manifest():
    return json.loads((INP / "manifest.json").read_text())


def ctok(req, mk):
    """Client-side content tokens of one manifest request under a model's tokenizer. The frozen
    manifest holds the first three models; later models are in _inputs/extra_tokens.json."""
    if mk in req["content_tokens"]:
        return req["content_tokens"][mk]
    return json.loads((INP / "extra_tokens.json").read_text())[mk][req["request_id"]]


def load_plan():
    plan = json.loads((SDIR / "plan.json").read_text())
    for f in sorted(SDIR.glob("plan_addendum_*.json")):
        add = json.loads(f.read_text())
        for blk, models in add.get("blocks", {}).items():
            plan["blocks"][blk] = plan["blocks"][blk] + models
        # Session checkpoint (target.md): points dropped before block 1, for all three blocks.
        plan["dropped_points"] = plan.get("dropped_points", []) + add.get("dropped_points", [])
    return plan


def split_rows(bucket, split):
    return [json.loads(l) for l in open(INP / bucket / f"{split}.jsonl")]


def chat_body(mk, prompt, **kw):
    return dict(model=rm.MODELS[mk]["model"], messages=[{"role": "user", "content": prompt}],
                temperature=0, **rm.extra_body(mk), **kw)


# ---------------------------------------------------------------- owned server
class Server:
    def __init__(self, mk, cfg):
        assert mk in ALLOWED, f"model {mk} is outside the study"
        assert mk not in RETIRED, f"{mk} is a retired key (reproduce/qwen-38-bf16.md)"
        assert cfg in CONFIGS, f"config {cfg} is not a no-prefix AR config"
        self.mk, self.cfg, self.proc = mk, cfg, None
        self.pdir = SDIR / mk / "profiles"
        self.log = SDIR / mk / "_server" / f"serve-{cfg}-{datetime.datetime.now():%Y%m%d-%H%M%S}.log"

    def __enter__(self):
        if any(m > 2000 for m in gpu_mem()) or sh("pgrep -f '[v]llm serve'").strip():
            raise RuntimeError("GPUs are not free; refusing to launch over another server")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.pdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HOME=os.environ.get("HF_HOME", "/workspace/hf"), PROFILE_DIR=str(self.pdir))
        self.t0 = time.time()
        self.proc = subprocess.Popen(["bash", "bench/serve.sh", self.mk, self.cfg, str(self.log), *SERVE_ARGS],
                                     cwd=ROOT, env=env, start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"launched {self.mk} {self.cfg} pgid={self.proc.pid} log={self.log.relative_to(ROOT)}")
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def wait_ready(self):
        want = rm.MODELS[self.mk]["model"]
        while time.time() - self.t0 < READY_TIMEOUT_S:
            time.sleep(5)
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode} before readiness")
            try:
                if json.loads(rm.http_get("/v1/models", timeout=5))["data"][0]["id"] == want:
                    self.ready_s = time.time() - self.t0
                    text = self.log.read_text(errors="replace")
                    if "'enable_prefix_caching': False" not in text:
                        raise RuntimeError("server log does not show prefix caching disabled")
                    log(f"ready {self.mk} {self.cfg} after {self.ready_s:.0f}s")
                    return
            except (OSError, ValueError, KeyError, IndexError):
                pass
        raise RuntimeError("readiness timeout")

    def stop(self):
        if self.proc is None:
            return
        # Grace periods are short: results are already on disk when stop() runs and the node is rented.
        def alive():  # poll() reaps the group leader; an unreaped leader would look alive to pgrep
            self.proc.poll()
            return sh(f"pgrep -g {self.proc.pid}").strip()

        for sig, wait in ((signal.SIGINT, 25), (signal.SIGTERM, 10), (signal.SIGKILL, 20)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            t = time.time()
            while time.time() - t < wait and alive():
                time.sleep(1)
            if not alive():
                break
        t = time.time()
        while time.time() - t < 120 and max(gpu_mem()) > 500:
            time.sleep(3)
        rec = dict(model=self.mk, config=self.cfg, stopped=datetime.datetime.now().isoformat(),
                   up_s=round(time.time() - self.t0), gpu_mem_mib=gpu_mem(),
                   leftover=alive().split())
        with open(LOGS / "shutdown.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"stopped {self.mk} {self.cfg} up={rec['up_s']}s gpu_mem_max={max(rec['gpu_mem_mib'])}MiB "
            f"leftover={len(rec['leftover'])}")
        self.proc = None

    def __exit__(self, *a):
        self.stop()


# ---------------------------------------------------------------- timing points
def run_bench(mk, workload, cfg, bucket, c, n, rep, split, timeout):
    """One `vllm bench serve` run through run_matrix.run_point, then per-request validation."""
    man = manifest()["buckets"][bucket]["splits"][split]
    reqs = man["requests"][:n]
    assert len(reqs) == n, f"{bucket}/{split} has fewer than {n} requests"
    point = f"ctx{bucket}_osl{OSL}_c{c}"
    pdir = SDIR / mk / workload / cfg / point
    summary = rm.run_point(mk, STUDY, workload, cfg, point, rep, c=c, n=n, seed=0,
                           dataset_args=rm.custom_args(INP / bucket / f"{split}.jsonl"),
                           expect_prompt_tokens=sum(ctok(r, mk) for r in reqs), timeout=timeout, osl=OSL)
    rdir = max(pdir.glob(f"repeat-{rep}*"), key=lambda p: p.stat().st_mtime)
    problems = list(summary["problems"])
    out_lens = in_lens = None
    try:
        res = json.loads((rdir / "bench.json").read_text())
        out_lens, in_lens = res.get("output_lens"), res.get("input_lens")
        if out_lens is None or len(out_lens) != n or any(x != OSL for x in out_lens):
            problems.append(f"per-request output length is not {OSL} for every request")
    except Exception as e:
        problems.append(f"bench.json unreadable: {e!r}")
    poll = [json.loads(l) for l in open(rdir / "telemetry" / "poll.jsonl")]
    running = [r.get("vllm:num_requests_running") for r in poll if r.get("vllm:num_requests_running") is not None]
    side = dict(study=STUDY, model_key=mk, runtime=runtime(), bucket=bucket, split=split, concurrency=c, n=n,
                block=rep, prefix_caching=False, speculation=False, output_tokens=OSL,
                list_file_sha256=man["file_sha256"],
                request_ids=[r["request_id"] for r in reqs], request_sha256=[r["sha256"] for r in reqs],
                domains=[r["domain"] for r in reqs], client_content_tokens=[ctok(r, mk) for r in reqs],
                output_lens=out_lens, input_lens_client=in_lens,
                running_requests_1hz=running, valid=not problems, problems=problems)
    (rdir / "blog_validation.json").write_text(json.dumps(side))
    log(f"{workload} {mk} {point} block={rep} n={n} valid={side['valid']} dur={summary.get('duration_s')} "
        f"out_tps={summary.get('output_throughput')} ttft_p50={summary.get('median_ttft_ms')} "
        f"tpot_p50={summary.get('median_tpot_ms')} {problems or ''}")
    return summary


def timing_step(mk, block):
    plan = load_plan()
    order = [(b, c) for b in BUCKETS for c in LOADS if f"{b}:c{c}" not in plan.get("dropped_points", ())]
    if block % 2 == 0:
        order.reverse()
    with Server(mk, "off") as srv:
        for b in BUCKETS:  # warm every shape with disjoint requests before the timed points
            if SESSION:  # same auxiliary prompts with a short output: the shapes, not 2,048 tokens each
                rm.warmup(mk, c=WARM_N[b], n=WARM_N[b], seed=0, dataset_args=rm.custom_args(INP / b / "warm.jsonl"))
            else:
                rm.warmup(mk, c=min(8, WARM_N[b]), n=WARM_N[b], seed=0,
                          dataset_args=rm.custom_args(INP / b / "aux.jsonl"))
        for b, c in order:
            run_bench(mk, "serving", "off", b, c, plan["counts"][f"{b}:c{c}"], block, "timing",
                      plan["run_timeout_s"])
        if SESSION:  # after the serving points, so a capacity limit at 128K cannot disturb them
            for b, B in INTERVALS:
                try:
                    decode_interval(srv, b, B, block)
                except Exception as e:
                    log(f"FAILED decode interval {mk} {b} B={B}: {e!r}")


# ---------------------------------------------------------------- diagnostics
def functional(mk):
    out = SDIR / mk / "_functional"; out.mkdir(parents=True, exist_ok=True)
    rec = dict(model_key=mk, runtime=runtime(), checks=[])
    row = split_rows("1k", "aux")[0]
    meta = manifest()["buckets"]["1k"]["splits"]["aux"]["requests"][0]
    try:  # rendered request: server-side token count of the templated chat prompt
        tk = post("/tokenize", dict(model=rm.MODELS[mk]["model"], messages=[{"role": "user", "content": row["prompt"]}],
                                    add_generation_prompt=True, **rm.extra_body(mk)))
        rec["rendered"] = dict(server_prompt_tokens=tk.get("count"), client_content_tokens=ctok(meta, mk),
                               template_overhead=tk.get("count") - ctok(meta, mk))
    except Exception as e:
        rec["rendered"] = dict(error=repr(e))
    natural = {"code": "Write a Python function that returns the n-th Fibonacci number iteratively. Reply with the code only.",
               "math": "What is 17 * 23? Give the final number on the last line.",
               "chat": "In two sentences, explain why the sky is blue."}
    for d, q in natural.items():  # natural EOS, thinking off
        r = post("/v1/chat/completions", chat_body(mk, q, max_tokens=512))
        m = r["choices"][0]
        rec["checks"].append(dict(kind="natural_eos", domain=d, prompt=q, text=m["message"].get("content"),
                                  reasoning=m["message"].get("reasoning_content") or m["message"].get("reasoning"),
                                  finish_reason=m["finish_reason"], usage=r["usage"]))
    r = post("/v1/chat/completions", chat_body(mk, row["prompt"], max_tokens=rm.OSL, ignore_eos=True))
    rec["checks"].append(dict(kind="forced_256", request_id=meta["request_id"], usage=r["usage"],
                              finish_reason=r["choices"][0]["finish_reason"]))
    nat = [c for c in rec["checks"] if c["kind"] == "natural_eos"]
    rec["ok"] = (all(c["finish_reason"] == "stop" and not c["reasoning"] and c["text"] for c in nat)
                 and rec["checks"][-1]["usage"]["completion_tokens"] == rm.OSL)
    (out / "functional.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log(f"functional {mk} ok={rec['ok']} rendered={rec['rendered']}")
    return rec["ok"]


class Stream(threading.Thread):
    """One streaming chat request; counts content chunks until closed."""

    def __init__(self, mk, prompt, max_tokens=4000, usage=False):
        super().__init__(daemon=True)
        self.body = chat_body(mk, prompt, max_tokens=max_tokens, ignore_eos=True, stream=True)
        if usage:  # session: every chunk carries the server's token counts for this request
            self.body["stream_options"] = dict(include_usage=True, continuous_usage_stats=True)
        self.n, self.t_first, self.err, self.resp = 0, None, None, None
        self.t_start, self.usage, self.done, self.status = None, None, False, None

    def run(self):
        import requests
        try:
            self.t_start = time.time()
            self.resp = requests.post(BASE + "/v1/chat/completions", json=self.body, stream=True, timeout=3600)
            self.status = self.resp.status_code
            for line in self.resp.iter_lines():
                if line.startswith(b"data: {") and b'"content"' in line:
                    self.n += 1
                    self.t_first = self.t_first or time.time()
                if line.startswith(b"data: {") and b'"usage"' in line:
                    u = json.loads(line[6:]).get("usage")
                    if u:
                        self.usage = u
                        if u.get("completion_tokens"):
                            self.t_first = self.t_first or time.time()
            self.done = True
        except Exception as e:  # closing the response to abort lands here
            self.err = repr(e)

    def close(self):
        if self.resp is not None:
            self.resp.close()


def kv_gauges():
    txt = rm.http_get("/metrics")
    return {g: rm.prom_max(txt, g) for g in rm.GAUGES}


def counters():
    """Engine progress from the server's own counters (never from streamed text chunks)."""
    t0 = time.time()
    txt = rm.http_get("/metrics")
    f = lambda name: rm.prom_sum(txt, name)
    return dict(t=(t0 + time.time()) / 2, gen=f("vllm:generation_tokens_total"), prompt=f("vllm:prompt_tokens_total"),
                first=f("vllm:time_to_first_token_seconds_count"), steps=f("vllm:iteration_tokens_total_count"),
                step_tokens=f("vllm:iteration_tokens_total_sum"), preempt=f("vllm:num_preemptions_total"),
                running=rm.prom_max(txt, "vllm:num_requests_running"),
                waiting=rm.prom_max(txt, "vllm:num_requests_waiting"), kv=rm.prom_max(txt, "vllm:kv_cache_usage_perc"))


class Live:
    """B requests held in the engine at one context, all admitted together and none added later."""

    def __init__(self, mk, bucket, B, max_tokens):
        self.mk, self.bucket, self.B = mk, bucket, B
        self.rows = split_rows(bucket, "aux")[:B]
        self.metas = manifest()["buckets"][bucket]["splits"]["aux"]["requests"][:B]
        assert len(self.rows) == B, f"{bucket}/aux has fewer than {B} requests"
        self.streams = [Stream(mk, r["prompt"], max_tokens=max_tokens, usage=True) for r in self.rows]

    def __enter__(self):
        self.idle = counters()
        self.t0 = time.time()
        for s in self.streams:
            s.start()
        return self

    def __exit__(self, *a):
        for s in self.streams:
            s.close()
        t = time.time()  # wait for finished or aborted requests to leave the engine
        while time.time() - t < 180 and (counters()["running"] or 0) > 0:
            time.sleep(0.5)

    def wait_decoding(self, timeout):
        """Return once every request has produced its first token (server-side TTFT count)."""
        while True:
            c = counters()
            if (c["first"] or 0) - (self.idle["first"] or 0) >= self.B:
                self.t_all = time.time() - self.t0
                return c
            bad = [s.err or s.status for s in self.streams if s.err or (s.status and s.status != 200)]
            if bad or time.time() - self.t0 > timeout:
                raise RuntimeError(f"requests did not all reach decode within {timeout}s: {bad}")
            time.sleep(0.1)

    def identity(self):
        return dict(model_key=self.mk, runtime=runtime(), bucket=self.bucket, B=self.B,
                    request_ids=[m["request_id"] for m in self.metas],
                    request_sha256=[m["sha256"] for m in self.metas],
                    client_content_tokens=[ctok(m, self.mk) for m in self.metas])

    def progress(self):
        return dict(usage=[s.usage for s in self.streams], chunks=[s.n for s in self.streams],
                    client_ttft_s=[(s.t_first - s.t_start) if s.t_first and s.t_start else None for s in self.streams],
                    finished=[s.done for s in self.streams], errors=[s.err for s in self.streams])

    def window(self, steps, cap_s, poll_s=0.1):
        """Decode-only interval until every request has committed `steps` more tokens (or the cap).

        Step time is elapsed time over engine steps, both from counter readings at the two ends;
        the samples in between show that all B requests stayed running and nothing was queued."""
        a = counters()
        samples = [a]
        while True:
            time.sleep(poll_s)
            c = counters()
            samples.append(c)
            if c["gen"] - a["gen"] >= steps * self.B or c["t"] - a["t"] > cap_s or (c["running"] or 0) < self.B:
                break
        b = samples[-1]
        full = [x for x in samples if (x["running"] or 0) >= self.B]
        b = full[-1] if len(full) >= 2 else b  # last reading with all B still running
        tokens, dt = b["gen"] - a["gen"], b["t"] - a["t"]
        n_steps = (b["steps"] - a["steps"]) if b["steps"] is not None and a["steps"] is not None else None
        out = dict(window_s=dt, tokens=tokens, engine_steps=n_steps,
                   step_tokens=(b["step_tokens"] - a["step_tokens"]) if n_steps is not None else None,
                   ms_per_token_step=(1e3 * dt * self.B / tokens) if tokens else None,
                   ms_per_engine_step=(1e3 * dt / n_steps) if n_steps else None,
                   tokens_per_engine_step=(tokens / n_steps) if n_steps else None,
                   preemptions=(b["preempt"] - a["preempt"]) if b["preempt"] is not None else None,
                   running_min=min((x["running"] or 0) for x in samples[:samples.index(b) + 1]),
                   waiting_max=max((x["waiting"] or 0) for x in samples), kv_usage_start=a["kv"], kv_usage_end=b["kv"],
                   samples=[[round(x["t"] - a["t"], 4), x["gen"] - a["gen"], x["running"], x["waiting"]] for x in samples])
        return out


def live_state(live, at):
    """State held by B live requests, right after the last of them produced its first token."""
    return dict(kind="live_kv", **live.identity(),
                idle_gauges={"vllm:kv_cache_usage_perc": live.idle["kv"]},
                live_gauges={"vllm:kv_cache_usage_perc": at["kv"], "vllm:num_requests_running": at["running"],
                             "vllm:num_requests_waiting": at["waiting"]},
                kv_usage_idle=live.idle["kv"], kv_usage_live=at["kv"],
                server_prompt_tokens_total=at["prompt"] - live.idle["prompt"],
                generated_tokens_total=at["gen"] - live.idle["gen"],
                preemptions=(at["preempt"] - live.idle["preempt"]) if at["preempt"] is not None else None,
                running=at["running"], waiting=at["waiting"], gpu_mem_mib=gpu_mem(),
                time_to_all_decoding_s=live.t_all, progress=live.progress(), recorded_utc=utc())


def session_capture(srv, bucket, B):
    """Session diagnostics at one decode condition: state snapshot, unprofiled window, profiled window."""
    mk, label = srv.mk, f"decode{bucket}_B{B}"
    kdir = SDIR / mk / "_kv"; kdir.mkdir(parents=True, exist_ok=True)
    meta = None
    with Live(mk, bucket, B, max_tokens=4000) as live:
        at = live.wait_decoding(1500)
        snap = live_state(live, at)
        # A preempted or queued request means the pool cannot hold this batch: a capacity limit,
        # recorded as such and kept out of the compute-scaling captures.
        snap["capacity_limit"] = bool(snap["preemptions"]) or (at["waiting"] or 0) > 0 or (at["running"] or 0) < B
        (kdir / f"live_{bucket}_B{B}.json").write_text(json.dumps(snap, indent=1))
        log(f"kv {mk} {bucket} B={B} usage={at['kv']} running={at['running']} waiting={at['waiting']} "
            f"prompt_tokens={snap['server_prompt_tokens_total']} all_decoding_after={live.t_all:.1f}s "
            f"capacity_limit={snap['capacity_limit']}")
        if snap["capacity_limit"]:
            return
        plain = live.window(128, 30)
        before = {p for p in srv.pdir.rglob("*") if p.is_file()}
        post("/start_profile")
        prof = live.window(48, 8, poll_s=0.05)
        post("/stop_profile")
        meta = dict(kind="decode", **live.identity(), unprofiled_window=plain, profiled_window=prof,
                    window_s=prof["window_s"], tokens_in_window=[prof["tokens"] / B] * B,
                    server_prompt_tokens_total=snap["server_prompt_tokens_total"],
                    kv_usage=at["kv"], progress=live.progress())
    collect_trace(srv, label, before, meta)


def decode_interval(srv, bucket, B, block):
    """Unprofiled decode interval on the plain launch: INTERVAL_STEPS tokens per request at actual B."""
    mk = srv.mk
    rdir = rm.unique_dir(SDIR / mk / "decode_interval" / "off" / f"ctx{bucket}_B{B}" / f"repeat-{block}")
    rdir.mkdir(parents=True)
    rec = dict(kind="decode_interval", study=STUDY, block=block, steps_requested=INTERVAL_STEPS, valid=False,
               problems=[], started_utc=utc())
    with Live(mk, bucket, B, max_tokens=4000) as live:
        rec.update(live.identity())
        try:
            at = live.wait_decoding(1500)
            rec.update(live_state(live, at))
            rec["kind"] = "decode_interval"
            if rec["preemptions"] or (at["waiting"] or 0) > 0:
                rec["problems"].append("capacity limit: a request was preempted or queued before the interval")
            else:
                live.window(16, 10)  # lead-in: every request is past its first tokens
                w = live.window(INTERVAL_STEPS, 600)
                rec["interval"] = w
                if w["tokens"] < INTERVAL_STEPS * B:
                    rec["problems"].append(f"interval ended at {w['tokens']} of {INTERVAL_STEPS * B} tokens")
                if w["preemptions"] or w["waiting_max"] or w["running_min"] < B:
                    rec["problems"].append("batch changed inside the interval (preemption, queueing or a finished request)")
            rec["progress_end"] = live.progress()
        except Exception as e:
            rec["problems"].append(repr(e))
    rec.update(valid=not rec["problems"], finished_utc=utc())
    (rdir / "interval.json").write_text(json.dumps(rec, indent=1))
    w = rec.get("interval") or {}
    log(f"interval {mk} {bucket} B={B} block={block} valid={rec['valid']} ms_per_step={w.get('ms_per_token_step')} "
        f"engine_step_ms={w.get('ms_per_engine_step')} window_s={w.get('window_s')} "
        f"all_decoding_after={rec.get('time_to_all_decoding_s')} {rec['problems'] or ''}")


# ---------------------------------------------------------------- expert-routing statistics
def routed_request(mk, prompt, max_tokens):
    """One request on the routed-experts launch: (response, expert IDs [tokens - 1, layers, k])."""
    import base64, io
    import numpy as np
    r = post("/v1/chat/completions", chat_body(mk, prompt, max_tokens=max_tokens, ignore_eos=True), timeout=3000)
    ch = r["choices"][0]
    b64 = ch.pop("routed_experts", None) or ch["message"].pop("routed_experts", None)
    return r, (np.load(io.BytesIO(base64.b64decode(b64))) if b64 else None)


def routing_summary(groups, n_experts, ranks=8):
    """Statistics of one condition. `groups` is a list of engine steps, each an int array
    [tokens in the step, layers, k] of selected expert IDs."""
    import numpy as np
    layers, k = groups[0].shape[1], groups[0].shape[2]
    counts = np.zeros((layers, n_experts), dtype=np.int64)       # tokens per expert, whole condition
    distinct = np.zeros((len(groups), layers), dtype=np.int64)   # experts used in one step
    rank_peak = np.zeros((len(groups), layers))                  # busiest GPU / mean GPU, per step
    per = n_experts // ranks
    for g, a in enumerate(groups):
        for l in range(layers):
            c = np.bincount(a[:, l, :].reshape(-1).astype(np.int64), minlength=n_experts)[:n_experts]
            counts[l] += c
            distinct[g, l] = int((c > 0).sum())
            load = c[:per * ranks].reshape(ranks, per).sum(1)
            rank_peak[g, l] = load.max() / load.mean() if load.sum() else 0
    # A layer whose every slot is expert 0 for every token has no routed experts (dense FFN) or no capture.
    routed = [l for l in range(layers) if not (counts[l, 0] == counts[l].sum())]
    sel = distinct[:, routed] if routed else distinct
    tok = [int(a.shape[0]) for a in groups]
    used = counts[routed] if routed else counts
    mean_load = used.sum(1, keepdims=True) / n_experts
    return dict(
        steps=len(groups), tokens_per_step_mean=float(np.mean(tok)), tokens_per_step_min=min(tok), tokens_per_step_max=max(tok),
        layers=layers, routed_layers=routed, k=k, n_experts=n_experts, nominal_share=k / n_experts,
        distinct_per_step_mean=float(sel.mean()), distinct_per_step_min=int(sel.min()), distinct_per_step_max=int(sel.max()),
        share_of_experts_per_step_mean=float(sel.mean() / n_experts),
        distinct_per_step_by_layer=[round(float(x), 3) for x in distinct.mean(0)],
        coverage_by_layer=[int((counts[l] > 0).sum()) for l in range(layers)],
        tokens_per_expert=dict(max_over_mean_by_layer=[round(float(x), 3) for x in (used.max(1) / mean_load[:, 0])],
                               cv_by_layer=[round(float(x), 3) for x in (used.std(1) / mean_load[:, 0])],
                               unused_experts_by_layer=[int((used[i] == 0).sum()) for i in range(len(used))]),
        gpu_load_peak_over_mean=dict(placement=f"assumed linear: GPU r holds experts r*{per} to (r+1)*{per}-1",
                                     per_step_mean=float(rank_peak[:, routed].mean()) if routed else None,
                                     whole_condition_by_layer=[round(float(x), 3) for x in (
                                         used[:, :per * ranks].reshape(len(used), ranks, per).sum(2).max(1)
                                         / used[:, :per * ranks].reshape(len(used), ranks, per).sum(2).mean(1))]),
        tokens_per_expert_counts=counts.tolist())


def routing_step(mk):
    """Read the router's selected experts on the frozen prompts (no speed claim is made here)."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    out = SDIR / mk / "_routing"; out.mkdir(parents=True, exist_ok=True)
    raw = SDIR / mk / "_routing_raw"; raw.mkdir(parents=True, exist_ok=True)  # large arrays stay on the node
    snap = Path(os.environ.get("HF_HOME", "/workspace/hf")) / "hub" / ("models--" + rm.MODELS[mk]["model"].replace("/", "--")) \
        / "snapshots" / rm.MODELS[mk]["revision"] / "config.json"
    cfg = json.loads(snap.read_text())
    cfg = {**cfg, **(cfg.get("text_config") or {})}
    n_cfg = next((cfg[x] for x in ("n_routed_experts", "num_experts", "num_local_experts") if cfg.get(x)), None)
    rec = dict(kind="expert_routing", study=STUDY, model_key=mk, started_utc=utc(), config_experts=n_cfg,
               config_k=next((cfg[x] for x in ("num_experts_per_tok", "n_activated_experts", "moe_topk") if cfg.get(x)), None),
               conditions={}, problems=[])
    with Server(mk, "off-routed"):
        rec["runtime"] = runtime()
        aux1k = split_rows("1k", "aux")
        tail1k = split_rows("1k", "timing")[-64:]  # never used by a timing run (those take the first 48)
        # Agreement with the plain launch: same forced request as the functional check.
        r, a = routed_request(mk, aux1k[0]["prompt"], OSL)
        if a is None:
            raise RuntimeError("the server returned no routed_experts: capture is not supported for this model")
        n_exp = n_cfg or int(a.max()) + 1
        text = r["choices"][0]["message"].get("content") or ""
        plain = json.loads((SDIR / mk / "_functional" / "functional.json").read_text()) \
            if (SDIR / mk / "_functional" / "functional.json").exists() else {}
        ref = next((c for c in plain.get("checks", []) if c["kind"] == f"forced_{OSL}" and "text_sha256" in c), None)
        same = hashlib.sha256(text.encode()).hexdigest() == ref["text_sha256"] if ref else None
        first_diff = next((i for i, (x, y) in enumerate(zip(text, ref["text"])) if x != y), None) if ref and not same else None
        rec["agreement_with_plain_launch"] = dict(request_id="1k-aux-000", output_tokens=r["usage"]["completion_tokens"],
                                                 same_text=same, first_differing_char=first_diff,
                                                 chars=len(text), reference="_functional/functional.json forced request")
        pt = r["usage"]["prompt_tokens"]
        np.savez_compressed(out / "decode1k_B1.npz", routed_experts=a, prompt_tokens=pt)
        rec["conditions"]["decode1k_B1"] = dict(prompt_tokens=pt, rows=int(a.shape[0]),
                                                **routing_summary([a[j:j + 1] for j in range(pt, a.shape[0])], n_exp))
        rec["conditions"]["prefill1k_one_step"] = dict(prompt_tokens=pt, **routing_summary([a[:pt]], n_exp))
        # One 16K prompt: its prefill runs in 8,192-token chunks.
        r, a = routed_request(mk, split_rows("16k", "aux")[0]["prompt"], 8)
        pt = r["usage"]["prompt_tokens"]
        np.savez_compressed(out / "prefill16k.npz", routed_experts=a, prompt_tokens=pt)
        chunks = [a[i:min(i + 8192, pt)] for i in range(0, pt, 8192)]
        rec["conditions"]["prefill16k_chunks"] = dict(prompt_tokens=pt, chunk_tokens=[int(c.shape[0]) for c in chunks],
                                                      **routing_summary(chunks, n_exp))
        rec["conditions"]["prefill16k_first_chunk"] = dict(prompt_tokens=pt, **routing_summary(chunks[:1], n_exp))
        # Decode at B=8 and B=64: one token of every request forms a step (requests are admitted together).
        for B, rows in ((8, aux1k[:8]), (64, tail1k)):
            with ThreadPoolExecutor(B) as ex:
                res = list(ex.map(lambda row: routed_request(mk, row["prompt"], OSL), rows))
            arrs = [x[1][x[0]["usage"]["prompt_tokens"]:] for x in res]
            n = min(x.shape[0] for x in arrs)
            np.savez_compressed(raw / f"decode1k_B{B}.npz", **{f"r{i}": x for i, x in enumerate(arrs)})
            steps = [np.concatenate([x[j:j + 1] for x in arrs]) for j in range(n)]
            rec["conditions"][f"decode1k_B{B}"] = dict(
                requests=B, generated_rows_per_request=n, domains=[("code", "math", "chat")[i % 3] for i in range(B)],
                output_tokens=[x[0]["usage"]["completion_tokens"] for x in res], **routing_summary(steps, n_exp))
    rec["finished_utc"] = utc()
    (out / "routing.json").write_text(json.dumps(rec))
    c = rec["conditions"]
    log(f"routing {mk} experts={n_exp} k={c['decode1k_B1']['k']} routed_layers={len(c['decode1k_B1']['routed_layers'])} "
        f"share B1={c['decode1k_B1']['share_of_experts_per_step_mean']:.3f} B8={c['decode1k_B8']['share_of_experts_per_step_mean']:.3f} "
        f"B64={c['decode1k_B64']['share_of_experts_per_step_mean']:.3f} "
        f"chunk8192={c['prefill16k_first_chunk']['share_of_experts_per_step_mean']:.3f} "
        f"agrees_with_plain={rec['agreement_with_plain_launch']['same_text']}")


# Twelve natural-ending prompts, four per domain, that ask for a sustained answer (2K cap).
NATURAL = [
    ("code", "merge_intervals", "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals given as a list of [start, end] pairs and returns the merged list sorted by start. Include a docstring. Reply with the code only."),
    ("code", "lru_cache", "Write a Python class `LRUCache` whose constructor takes a capacity and which has `get(key)` and `put(key, value)` methods; `get` returns -1 for a missing key and both count as a use. Reply with the code only."),
    ("code", "to_roman", "Write a Python function `to_roman(n)` that converts an integer from 1 to 3999 to a Roman numeral string. Reply with the code only."),
    ("code", "top_k_words", "Write a Python function `top_k_words(text, k)` that returns the k most frequent lowercase words in `text` as a list of (word, count) tuples, most frequent first, ties broken alphabetically. Words are maximal runs of ASCII letters. Reply with the code only."),
    ("math", "average_speed", "A train travels 180 km at 60 km/h and then 120 km at 40 km/h. What is its average speed for the whole trip in km/h? Explain step by step, then give the final number on the last line in the form `Answer: <number>`."),
    ("math", "divisors_360", "How many positive divisors does 360 have? Explain step by step, then give the final number on the last line in the form `Answer: <number>`."),
    ("math", "linear_equation", "Solve for x: 3x + 7 = 2(x - 4) + 30. Explain step by step, then give the final number on the last line in the form `Answer: <number>`."),
    ("math", "sum_multiples", "What is the sum of all integers from 1 to 200 that are divisible by 3 or by 5? Explain step by step, then give the final number on the last line in the form `Answer: <number>`."),
    ("chat", "hash_table", "Explain how a hash table works to a first-year computer science student in about 300 words. End your reply with the single word DONE on its own line."),
    ("chat", "interview_steps", "Write a guide with exactly five numbered steps for preparing for a job interview. Each step should be two or three sentences."),
    ("chat", "cars_essay", "Compare electric cars and petrol cars in a short essay of four paragraphs: cost, range, maintenance and environmental impact, in that order."),
    ("chat", "ww1_bullets", "Summarise the causes of the First World War in exactly three bullet points, each starting with '- '."),
]


def functional_session(mk):
    """Rendered request, twelve natural-ending answers and one forced-length request."""
    out = SDIR / mk / "_functional"; out.mkdir(parents=True, exist_ok=True)
    rec = dict(model_key=mk, runtime=runtime(), output_tokens=OSL, natural_cap=2048, checks=[], recorded_utc=utc())
    row = split_rows("1k", "aux")[0]
    meta = manifest()["buckets"]["1k"]["splits"]["aux"]["requests"][0]
    try:
        tk = post("/tokenize", dict(model=rm.MODELS[mk]["model"], messages=[{"role": "user", "content": row["prompt"]}],
                                    add_generation_prompt=True, **rm.extra_body(mk)))
        rec["rendered"] = dict(server_prompt_tokens=tk.get("count"), client_content_tokens=ctok(meta, mk),
                               template_overhead=tk.get("count") - ctok(meta, mk))
    except Exception as e:
        rec["rendered"] = dict(error=repr(e))
    for d, task, q in NATURAL:
        t0 = time.time()
        try:
            r = post("/v1/chat/completions", chat_body(mk, q, max_tokens=2048))
            m = r["choices"][0]
            rec["checks"].append(dict(kind="natural_eos", domain=d, task=task, prompt=q, text=m["message"].get("content"),
                                      reasoning=m["message"].get("reasoning_content") or m["message"].get("reasoning"),
                                      finish_reason=m["finish_reason"], usage=r["usage"], wall_s=time.time() - t0))
        except Exception as e:
            rec["checks"].append(dict(kind="natural_eos", domain=d, task=task, prompt=q, error=repr(e)))
    nat = [c for c in rec["checks"] if c["kind"] == "natural_eos"]
    try:
        r = post("/v1/chat/completions", chat_body(mk, row["prompt"], max_tokens=OSL, ignore_eos=True), timeout=1500)
        text = r["choices"][0]["message"].get("content") or ""
        # The text is kept so that an instrumented launch can be checked against this plain output.
        rec["checks"].append(dict(kind=f"forced_{OSL}", request_id=meta["request_id"], usage=r["usage"],
                                  finish_reason=r["choices"][0]["finish_reason"], text=text,
                                  reasoning=r["choices"][0]["message"].get("reasoning_content")
                                  or r["choices"][0]["message"].get("reasoning"),
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        forced = r["usage"]["completion_tokens"] == OSL
    except Exception as e:
        rec["checks"].append(dict(kind=f"forced_{OSL}", error=repr(e)))
        forced = False
    rec["natural_stopped"] = sum(c.get("finish_reason") == "stop" for c in nat)
    rec["natural_with_reasoning"] = sum(bool(c.get("reasoning")) for c in nat)
    rec["forced_ok"] = forced
    rec["ok"] = forced and all(c.get("finish_reason") == "stop" and not c.get("reasoning") and c.get("text") for c in nat)
    (out / "functional.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log(f"functional {mk} ok={rec['ok']} forced={forced} natural_stopped={rec['natural_stopped']}/12 "
        f"with_reasoning={rec['natural_with_reasoning']} rendered={rec['rendered']}")
    return rec["ok"]


def collect_trace(srv, label, before, meta):
    """Wait for the rank traces written since `before` and file them under profiles/<label>/."""
    new, last = [], -1
    for _ in range(180):
        new = sorted(p for p in srv.pdir.rglob("*") if p.is_file() and p not in before and p.parent == srv.pdir)
        size = sum(p.stat().st_size for p in new)
        if len(new) >= 8 and size == last:
            break
        last = size
        time.sleep(5)
    dst = srv.pdir / label; dst.mkdir(exist_ok=True)
    for p in new:
        shutil.move(str(p), dst / p.name)
    meta.update(label=label, files=sorted(f.name for f in dst.iterdir()), n_rank_files=len(new),
                note="diagnostic profiled capture; not a timing result")
    (srv.pdir / f"{label}.manifest.json").write_text(json.dumps(meta, indent=1))
    log(f"trace {srv.mk} {label} files={len(new)} "
        f"{ {k: meta[k] for k in ('B', 'window_s', 'tokens_in_window', 'request_wall_s') if k in meta} }")


def prefill_capture(srv, bucket):
    mk = srv.mk
    row = split_rows(bucket, "aux")[0]
    meta = manifest()["buckets"][bucket]["splits"]["aux"]["requests"][0]
    before = {p for p in srv.pdir.rglob("*") if p.is_file()}
    post("/start_profile")
    t0 = time.time()
    r = post("/v1/chat/completions", chat_body(mk, row["prompt"], max_tokens=8, ignore_eos=True))
    wall = time.time() - t0
    post("/stop_profile")
    collect_trace(srv, f"prefill{bucket}", before,
                  dict(kind="prefill", model_key=mk, runtime=runtime(), bucket=bucket, request_id=meta["request_id"],
                       request_sha256=meta["sha256"], usage=r["usage"], max_tokens=8, request_wall_s=wall))


def decode_capture(srv, bucket, B, profile):
    """B live sequences, all past prefill, no new admissions: KV snapshot and (optionally) a trace window."""
    mk = srv.mk
    rows = split_rows(bucket, "aux")[:B]
    metas = manifest()["buckets"][bucket]["splits"]["aux"]["requests"][:B]
    idle = kv_gauges()
    streams = [Stream(mk, r["prompt"]) for r in rows]
    for s in streams:
        s.start()
    t0 = time.time()
    try:
        # All B requests must be past prefill. A stream can stop yielding text once the model
        # passes its natural end (ignore_eos keeps generating special tokens), so progress is
        # judged on the busiest stream, and the engine batch is read from the trace itself.
        while min(s.n for s in streams) < 1 or max(s.n for s in streams) < 8:
            if time.time() - t0 > 600 or any(s.err for s in streams):
                raise RuntimeError(f"decode capture did not reach steady decode: {[s.err for s in streams]}")
            time.sleep(0.05)
        snap = dict(kind="live_kv", model_key=mk, runtime=runtime(), bucket=bucket, B=B,
                    request_ids=[m["request_id"] for m in metas],
                    client_content_tokens=[ctok(m, mk) for m in metas],
                    generated_at_snapshot=[s.n for s in streams], idle_gauges=idle, live_gauges=kv_gauges(),
                    gpu_mem_mib=gpu_mem(), time_to_all_decoding_s=time.time() - t0)
        kdir = SDIR / mk / "_kv"; kdir.mkdir(parents=True, exist_ok=True)
        (kdir / f"live_{bucket}_B{B}.json").write_text(json.dumps(snap, indent=1))
        log(f"kv {mk} {bucket} B={B} usage={snap['live_gauges'].get('vllm:kv_cache_usage_perc')} "
            f"running={snap['live_gauges'].get('vllm:num_requests_running')}")
        # Plain (unprofiled) decode window first: step time at this exact B and context.
        n0, t1 = [s.n for s in streams], time.time()
        time.sleep(1.0)
        n1, t2 = [s.n for s in streams], time.time()
        plain = dict(window_s=t2 - t1, tokens=[b - a for a, b in zip(n0, n1)])
        if profile:
            before = {p for p in srv.pdir.rglob("*") if p.is_file()}
            post("/start_profile")
            a, ta = [s.n for s in streams], time.time()
            while max(s.n - x for s, x in zip(streams, a)) < 48 and time.time() - ta < 5:
                time.sleep(0.02)
            b, tb = [s.n for s in streams], time.time()
            post("/stop_profile")
            meta = dict(kind="decode", model_key=mk, runtime=runtime(), bucket=bucket, B=B,
                        request_ids=snap["request_ids"], client_content_tokens=snap["client_content_tokens"],
                        generated_before_window=a, tokens_in_window=[y - x for x, y in zip(a, b)],
                        window_s=tb - ta, unprofiled_window=plain, live_gauges=snap["live_gauges"])
    finally:
        for s in streams:
            s.close()
    t = time.time()  # wait for the aborted requests to leave the engine
    while time.time() - t < 60 and (kv_gauges().get("vllm:num_requests_running") or 0) > 0:
        time.sleep(0.5)
    if profile:
        collect_trace(srv, f"decode{bucket}_B{B}", before, meta)


def diag_step(mk, parts):
    with Server(mk, "off-profidle") as srv:
        if SESSION:
            (SDIR / mk / "_server" / "metrics_idle.prom").write_text(rm.http_get("/metrics"))
        if "functional" in parts and not (functional_session(mk) if SESSION else functional(mk)):
            log(f"functional check FAILED for {mk}; continuing with diagnostics only")
        if "pilot" in parts:
            for b in BUCKETS:
                for c in LOADS:
                    run_bench(mk, "pilot", "off-profidle", b, c, PILOT_N[(b, c)], 1, "aux", 1500 if SESSION else 900)
        if SESSION:
            if "kv" in parts or "trace" in parts:
                for b in CAPTURE_BUCKETS:
                    for B in (1, 8):
                        try:
                            session_capture(srv, b, B)
                        except Exception as e:
                            log(f"FAILED decode capture {mk} {b} B={B}: {e!r}")
            if "trace" in parts:
                for b in CAPTURE_BUCKETS:
                    try:
                        prefill_capture(srv, b)
                    except Exception as e:
                        log(f"FAILED prefill capture {mk} {b}: {e!r}")
            return
        if "kv" in parts:
            for b in ("1k", "16k", "64k"):
                for B in (1, 8):
                    try:
                        decode_capture(srv, b, B, profile=("trace" in parts and b != "16k"))
                    except Exception as e:
                        log(f"FAILED decode capture {mk} {b} B={B}: {e!r}")
        if "trace" in parts:
            for b in ("16k", "64k"):
                try:
                    prefill_capture(srv, b)
                except Exception as e:
                    log(f"FAILED prefill capture {mk} {b}: {e!r}")


# ---------------------------------------------------------------- plan / env / dry-run
def write_env():
    out = SDIR / "_env"; out.mkdir(parents=True, exist_ok=True)
    venv = runtime()["venv"]
    for name, cmd in dict(nvidia_smi="nvidia-smi", gpu_query="nvidia-smi --query-gpu=index,name,memory.total,"
                          "clocks.sm,clocks.mem,power.limit,pci.bus_id --format=csv", topology="nvidia-smi topo -m",
                          nvlink="nvidia-smi nvlink -s", lscpu="lscpu", numa="numactl -H", mem="free -g",
                          cpufreq="cd /sys/devices/system/cpu/cpu0/cpufreq 2>/dev/null && grep . scaling_governor "
                                  "scaling_driver scaling_min_freq scaling_max_freq || echo 'no cpufreq interface'",
                          freeze=f"{venv}/bin/python -m pip freeze 2>/dev/null || {venv}/bin/uv pip freeze --python {venv}/bin/python").items():
        (out / f"{name}.txt").write_text(sh(cmd))
    extra = dict(commit=RERUN_COMMIT, recorded_utc=utc()) if SESSION else {}
    (out / "runtime.json").write_text(json.dumps(dict(runtime(), recorded=datetime.datetime.now().isoformat(),
                                                      serve_args=SERVE_ARGS, **extra), indent=1))
    log(f"env recorded: {runtime()}")


def write_session_plan(counts, note):
    """Frozen plan of the 15-hour session (target.md#the-15-hour-session)."""
    n_timing = sum(len(v) for v in SPEC["blocks"].values())
    plan = dict(
        study=STUDY, frozen_utc=utc(), runtime=runtime(), runtime_commit=RERUN_COMMIT,
        runtime_source="latest nightly wheel on 2026-10-10 (wheels.vllm.ai/nightly), frozen before the first launch",
        models={mk: {k: rm.MODELS[mk][k] for k in ("model", "revision")} for mk in ALLOWED},
        node="one 8xH100 80GB node for every model; inventory in _env/",
        deployment="TP8 + EP, utilization 0.90, max_model_len 262144, max_num_seqs 64, "
                   "max_num_batched_tokens 8192 (explicit), native precision, text only, prefix caching off, "
                   "speculation off",
        requests=f"public code/math/chat text, one user message, server-side chat template, temperature 0, thinking "
                 f"off (GLM: reasoning_effort=low), ignore_eos with {OSL} output tokens",
        inputs="same corpus and construction as blog-architecture-h100-v1; 1K and 16K request texts identical to it, "
               "64K timing split extended from 45 to 48, 128K auxiliary requests added for the context diagnostics",
        inputs_manifest_sha256=hashlib.sha256((INP / "manifest.json").read_bytes()).hexdigest(),
        output_tokens=OSL, loads=list(LOADS), counts=counts,
        count_rule="12 requests at one client and 48 at sixteen (4 and 16 per domain), the first n of the timing "
                   "split for every model; below the max(24, 4c) starting rule so that three blocks fit the session",
        blocks=SPEC["blocks"], block_rule="every published serving number has three launch-separated blocks; "
                                          "each block starts two models later in the same cyclic order",
        diagnostic_models=list(SPEC["diag"]),
        point_order="16k c1, 16k c16, 1k c1, 1k c16, 64k c1, 64k c16 in odd blocks; reversed in even blocks; "
                    "then the decode intervals in fixed order",
        warmup="auxiliary split (disjoint from timing) at every bucket after each launch, 128 output tokens",
        decode_intervals=dict(conditions=[f"{b}:B{B}" for b, B in INTERVALS], tokens_per_request=INTERVAL_STEPS,
                              rule="B auxiliary requests admitted together, interval starts 16 tokens after the "
                                   "last first token, no admissions inside it; progress from server counters"),
        run_timeout_s=3000, ready_timeout_s=READY_TIMEOUT_S,
        validity=f"all requests complete, every request returns exactly {OSL} output tokens, server prompt tokens "
                 "at least 98% of the client content tokens, no failed requests",
        rerun_rule="an invalid run is kept; one rerun in a new repeat-N-rerunK directory is paired by manifest "
                   "identity; the fastest attempt is never selected",
        diagnostics="per model, one idle-profiler launch: architecture and support check from the server log; "
                    "12 natural-ending prompts (2K cap) and one forced request; pilot of the six serving points "
                    f"with {dict((f'{b}:c{c}', n) for (b, c), n in PILOT_N.items())} auxiliary requests; state "
                    "snapshot, unprofiled window and trace at 1k/16k/64k/128k x B=1/8; prefill traces at "
                    "1k/16k/64k/128k",
        capacity_rule="a preempted or queued request in a diagnostic batch is recorded as a capacity limit and "
                      "gets no trace or interval value",
        timing_runs=len(BUCKETS) * len(LOADS) * n_timing, decode_interval_runs=len(INTERVALS) * n_timing,
        trace_captures=12 * len(SPEC["diag"]), live_state_snapshots=8 * len(SPEC["diag"]),
        session=dict(hours=15, order=["setup", "diagnostics", "expert-routing statistics (gated)",
                                      "HBM counters (gated)", "timing blocks 1-3"],
                     checkpoint="before block 1: if less than 7.5 h remain drop 16k:c16, then all c16; recorded "
                                "in plan_addendum_matrix.json as dropped_points and kept for all three blocks",
                     not_run="speculative decoding; loads 4, 32, 64; held-out blocks; the 60-task set; drift "
                             "reference runs; the five conditional extensions; prefix caching (never measured)"),
        note=note)
    (SDIR / "plan.json").write_text(json.dumps(plan, indent=1))
    log(f"session plan frozen: {counts}")


def write_plan(counts, note):
    if SESSION:
        return write_session_plan(counts, note)
    plan = dict(
        study=STUDY, frozen=datetime.datetime.now().isoformat(), runtime=runtime(),
        models={mk: {k: rm.MODELS[mk][k] for k in ("model", "revision")} for mk in ALLOWED},
        inputs_reference=SPEC["inputs_reference"],
        deployment="TP8 + EP, utilization 0.90, max_model_len 262144, max_num_seqs 64, "
                   "max_num_batched_tokens 8192 (explicit), native precision, prefix caching off, speculation off",
        requests="public code/math/chat text, one user message, server-side chat template, temperature 0, "
                 "thinking off, ignore_eos with 256 output tokens",
        inputs_manifest_sha256=hashlib.sha256((INP / "manifest.json").read_bytes()).hexdigest(),
        counts=counts, count_rule="start at max(16, 4c), multiple of 3 for domain balance, raised until the "
                                  "fastest piloted model has a window of at least 60 s; same first-n requests "
                                  "of the timing split for every model",
        blocks=SPEC["blocks"], diagnostic_models=list(SPEC["diag"]),
        point_order="16k c1, 16k c8, 1k c1, 1k c8, 64k c1, 64k c8 in odd blocks; reversed in even blocks",
        warmup="aux split (disjoint from timing) at every bucket after each launch",
        run_timeout_s=1500, ready_timeout_s=READY_TIMEOUT_S,
        validity="all requests complete, every request returns exactly 256 output tokens, server prompt "
                 "tokens at least 98% of the client content tokens, no failed requests",
        rerun_rule="an invalid run is kept; one rerun in a new repeat-N-rerunK directory is paired by "
                   "manifest identity; the fastest attempt is never selected",
        diagnostics="per model: functional checks, disjoint pilot, live-KV snapshots at 1k/16k/64k x B=1/8, "
                    "traces prefill16k, prefill64k, decode1k/64k x B=1/8 on the idle-profiler launch",
        timing_runs=6 * sum(len(v) for v in SPEC["blocks"].values()), trace_captures=6 * len(SPEC["diag"]),
        live_kv_snapshots=6 * len(SPEC["diag"]), note=note)
    (SDIR / "plan.json").write_text(json.dumps(plan, indent=1))
    log(f"plan frozen: {counts}")


def dry_run(steps):
    plan = load_plan()
    man = manifest()
    problems = []
    if hashlib.sha256((INP / "manifest.json").read_bytes()).hexdigest() != plan["inputs_manifest_sha256"]:
        problems.append("inputs manifest changed after the plan was frozen")
    if runtime()["vllm"] != RUNTIME:
        problems.append(f"runtime {runtime()['vllm']} is not the study's build {RUNTIME}")
    if SPEC["inputs_reference"]:  # same request lists as the completed blog study, byte for byte
        ref = json.loads((ROOT / SPEC["inputs_reference"]).read_text())
        for b, e in ref["buckets"].items():
            for split, s in e["splits"].items():
                mine = man["buckets"].get(b, {}).get("splits", {}).get(split, {})
                if mine.get("file_sha256") != s["file_sha256"] or \
                        [r["sha256"] for r in mine.get("requests", [])] != [r["sha256"] for r in s["requests"]]:
                    problems.append(f"{b}/{split} differs from the reference inputs")
    serve = (ROOT / "bench" / "serve.sh").read_text()
    for mk in ALLOWED:
        pinned_doc = "".join((ROOT / f).read_text() for f in SPEC["pins"]) + \
            "".join(f.read_text() for f in SDIR.glob("plan_addendum_*.json"))
        if rm.MODELS[mk]["revision"] not in serve or rm.MODELS[mk]["revision"] not in pinned_doc:
            problems.append(f"{mk} revision is not pinned consistently")
    for b in man["buckets"]:
        for split, s in man["buckets"][b]["splits"].items():
            if hashlib.sha256((INP / s["file"]).read_bytes()).hexdigest() != s["file_sha256"]:
                problems.append(f"{s['file']} hash mismatch")
        for c in LOADS:
            if b not in BUCKETS:
                continue  # diagnostics-only context (128K in the rerun session)
            n = plan["counts"][f"{b}:c{c}"]
            if n > man["buckets"][b]["splits"]["timing"]["n"] or n % 3:
                problems.append(f"{b}:c{c} count {n} unavailable or not domain balanced")
    seen = set()
    for st in steps:
        kind, mk, *rest = st.split(":")
        if kind == "routing" and SESSION and mk in ALLOWED:
            continue
        if mk not in ALLOWED or mk in RETIRED or kind not in ("diag", "timing") or \
                (kind == "diag" and mk not in SPEC["diag"]):
            problems.append(f"step {st} not allowed")
            continue
        if kind == "timing":
            blk = rest[0]
            if (mk, blk) in seen or mk not in plan["blocks"].get(blk, []):
                problems.append(f"step {st} duplicates or is outside the declared blocks")
            seen.add((mk, blk))
            for b in BUCKETS:
                for c in LOADS:
                    d = SDIR / mk / "serving" / "off" / f"ctx{b}_osl{OSL}_c{c}" / f"repeat-{blk}"
                    print(f"  would write {d.relative_to(ROOT)}{' (exists: rerun dir)' if d.exists() else ''}")
    print("configs:", CONFIGS, "serve args:", SERVE_ARGS, "runtime:", runtime())
    print("DRY RUN", "FAILED: " + "; ".join(problems) if problems else "OK", f"({len(steps)} steps)")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["env", "plan", "dry-run", "chain"])
    ap.add_argument("steps", nargs="*")
    ap.add_argument("--counts"); ap.add_argument("--note", default="")
    ap.add_argument("--diag-parts", default="functional,pilot,kv,trace")
    ap.add_argument("--publish", action="store_true", help="session: curate, commit and push after every step")
    a = ap.parse_args()
    if a.cmd == "env":
        return write_env()
    if a.cmd == "plan":
        return write_plan(json.loads(a.counts), a.note)
    if a.cmd == "dry-run":
        sys.exit(0 if dry_run(a.steps) else 1)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally/exit paths on termination
    for st in a.steps:
        kind, mk, *rest = st.split(":")
        outcome = "failed"
        for attempt in (1, 2):
            log(f"step {st} start" + (" (retry after a failed launch)" if attempt == 2 else ""))
            if SESSION:
                timeline(step=st, model=mk, event="start", attempt=attempt)
            try:
                if kind == "diag":
                    diag_step(mk, a.diag_parts.split(","))
                elif kind == "timing":
                    timing_step(mk, int(rest[0]))
                elif kind == "routing" and SESSION:
                    routing_step(mk)
                else:
                    raise SystemExit(f"unknown step {st}")
                log(f"step {st} done")
                outcome = "done"
                break
            except SystemExit:
                raise
            except Exception as e:
                log(f"step {st} FAILED: {e!r}")
                # A first launch can die while the ranks JIT-build one module; one relaunch is allowed.
                if "before readiness" not in repr(e):
                    break
        if SESSION:  # the server is stopped: record the time, then publish and push this model's results
            timeline(step=st, model=mk, event="end", outcome=outcome)
            if a.publish:
                r = subprocess.run([sys.executable, str(ROOT / "bench" / "session_publish.py"), st], cwd=ROOT,
                                   capture_output=True, text=True)
                log(f"publish {st} rc={r.returncode} {(r.stdout + r.stderr).strip().splitlines()[-1:] or ''}")
    log("chain finished; no server left running" if not sh("pgrep -f '[v]llm serve'").strip()
        else "chain finished; WARNING a vllm serve process is still present")


if __name__ == "__main__":
    main()
