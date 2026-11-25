from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Callable

import torch

from ..constants import SEED

WARMUP = 1
REPS = 1
BLOCKS = ("dense", "dense", "method", "dense")


class _CpuEvent:
    def __init__(self, enable_timing=True):
        self.t = None

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, end) -> float:
        return (end.t - self.t) * 1e3


def _event():
    return torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else _CpuEvent()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CallTimer:
    def __init__(self):
        self.rec, self.skip = [], []

    @contextlib.contextmanager
    def __call__(self, tag: str):
        a, b = _event(), _event()
        a.record()
        yield
        b.record()
        self.rec.append((tag, a, b))

    @contextlib.contextmanager
    def excluded(self):
        _sync()
        a, b = _event(), _event()
        a.record()
        yield
        _sync()
        b.record()
        self.skip.append((a, b, len(self.rec)))

    def result(self) -> tuple[list, float, float]:
        _sync()
        calls = [(t, a.elapsed_time(b)) for t, a, b in self.rec]
        skip = sum(a.elapsed_time(b) for a, b, n in self.skip if 0 < n < len(self.rec))
        return calls, (self.rec[0][1].elapsed_time(self.rec[-1][2]) - skip if self.rec else 0.0), skip


def _stats(hook):
    s = getattr(hook, "summary", None)
    return s() if callable(s) else getattr(hook, "stats", None)


def _prepare(hook) -> None:
    if hook is None:
        return
    for name in ("reset", "build_all"):
        f = getattr(hook, name, None)
        if callable(f):
            f()


def generate_timed(adapter, hook, clip: str, seed: int = SEED) -> dict:
    _prepare(hook)
    timer, whole = CallTimer(), CallTimer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync()
    adapter.timer = timer
    try:
        with whole("generate"):
            adapter.generate(clip, hook, seed)
    finally:
        adapter.timer = None
    per_call = bool(timer.rec)
    calls, ms, skip = (timer if per_call else whole).result()
    region = "dit_calls" if per_call else "generate"
    if per_call and getattr(adapter, "latency", "span") == "calls":
        ms, region = sum(c for _, c in calls), "dit_call_sum"
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    return {"latency_ms": ms, "call_ms": calls, "excluded_ms": skip, "peak_GiB": peak, "region": region}


def time_block(adapter, make_hook: Callable, clip: str, seed: int = SEED, reps: int = REPS, warmup: int = WARMUP,
               name: str = "method", log=print) -> dict:
    _sync()
    b0, b1 = _event(), _event()
    b0.record()
    hook = make_hook()
    b1.record()
    for _ in range(warmup):
        generate_timed(adapter, hook, clip, seed)
    runs = [generate_timed(adapter, hook, clip, seed) for _ in range(reps)]
    lat = [r["latency_ms"] for r in runs]
    out = {"row": name, "latency_ms": statistics.fmean(lat), "runs_ms": lat, "build_ms": b0.elapsed_time(b1),
           "call_ms": runs[-1]["call_ms"], "excluded_ms": [r["excluded_ms"] for r in runs], "region": runs[-1]["region"],
           "peak_GiB": max((r["peak_GiB"] for r in runs if r["peak_GiB"] is not None), default=None),
           "stats": _stats(hook)}
    log(f"timing {name}: {out['latency_ms']:.0f} ms (mean of {reps}; build {out['build_ms']:.0f} ms)")
    del hook
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarize_blocks(blocks: list[dict]) -> dict:
    d = [b["latency_ms"] for b in blocks if b["row"] == "dense"]
    dense = statistics.fmean(d)
    method = next(b["latency_ms"] for b in blocks if b["row"] == "method")
    return {"dense_ms": dense, "method_ms": method, "speedup": dense / method, "dense_gap": (max(d) - min(d)) / dense}


def time_model(adapter, make_hook: Callable, clip: str, seed: int = SEED, reps: int = REPS, warmup: int = WARMUP,
               log=print) -> dict:
    makers = {"dense": lambda: None, "method": make_hook}
    blocks = [time_block(adapter, makers[name], clip, seed, reps, warmup, name, log) for name in BLOCKS]
    return {"clip": clip, "seed": seed, "reps": reps, "warmup": warmup,
            "protocol": f"one process: {', '.join(BLOCKS)}; {warmup} untimed warm-up + {reps} timed generation(s) per "
                        "block; speedup = mean of the three dense latencies / method latency; latency = first DiT call "
                        "start -> last DiT call end (CUDA events); the hook is built once per block, outside the "
                        "latency; plan building is excluded from the latency",
            "rows": blocks, **summarize_blocks(blocks),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
