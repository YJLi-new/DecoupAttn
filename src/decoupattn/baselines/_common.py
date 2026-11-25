from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import ClassVar

import torch

from ..config import REPO_ROOT, data_root
from ..constants import SEED

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SETTINGS_DIR = os.path.join(REPO_ROOT, "configs", "baselines")
SOURCES_DIR = os.path.join(REPO_ROOT, "third_party", "baselines")

DENSE_STEPS = 1
DENSE_LAYERS = 1
DENSITY_BAND = (0.195, 0.205)
LN2 = 0.6931471805599453


def load_settings(method: str, model: str, path: str | None = None) -> dict:
    with open(path or os.path.join(SETTINGS_DIR, f"{method}.toml"), "rb") as f:
        cfg = tomllib.load(f)
    return {**cfg.get("default", {}), **cfg.get(model, {})}


def load_source(name: str) -> dict:
    with open(os.path.join(SOURCES_DIR, name, "source.toml"), "rb") as f:
        return tomllib.load(f)


def upstream_path(name: str, *parts: str) -> str:
    root = os.path.join(data_root(), "baselines", load_source(name).get("dest", name))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} is missing: run `python scripts/fetch_baselines.py --name {name}`")
    return os.path.join(root, *parts)


def add_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


class Baseline:
    name = ""
    label = ""
    upstream: tuple[str, ...] = ()
    standalone = False
    reference = "dense"
    timing_settings: ClassVar[dict] = {}

    def __init__(self, model: str, **overrides):
        self.model = model
        self.settings = {**load_settings(self.name, model), **overrides}

    def work_dir(self, run_dir: str) -> str:
        return os.path.join(run_dir, "baselines", self.name)

    def prepare(self, adapter, run_dir: str, device: str = "cuda", log=print) -> None:
        pass

    def hook(self, adapter, run_dir: str):
        raise NotImplementedError

    def timing_hook(self, adapter, run_dir: str | None = None):
        saved = self.settings
        self.settings = {**saved, **self.timing_settings}
        try:
            return self.hook(adapter, run_dir)
        finally:
            self.settings = saved

    def generate(self, adapter, clips: list[str], out_dir: str, seed: int, dense: bool = False, log=print) -> None:
        raise NotImplementedError(f"{self.name} runs through its hook")

    def time(self, adapter, clip: str, seed: int, log=print) -> dict:
        raise NotImplementedError(f"{self.name} runs through its hook")


def bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def flat(o: torch.Tensor) -> torch.Tensor:
    return o.permute(0, 2, 1, 3).flatten(2, 3)


def unpermute_rows(x_p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x_p)
    out.scatter_(-1, idx.long(), x_p)
    return out


@contextlib.contextmanager
def seeded(device: torch.device, layer: int, step: int):
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(1000 * layer + step)
        yield


class Sections:
    def __init__(self):
        self.ev = []

    @contextlib.contextmanager
    def __call__(self, name: str):
        cuda = torch.cuda.is_available()
        a = torch.cuda.Event(enable_timing=True) if cuda else time.perf_counter()
        if cuda:
            a.record()
        try:
            yield
        finally:
            b = torch.cuda.Event(enable_timing=True) if cuda else time.perf_counter()
            if cuda:
                b.record()
            self.ev.append((name, a, b))

    def ms(self) -> dict[str, float]:
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        out: dict[str, float] = {}
        for n, a, b in self.ev:
            out[n] = out.get(n, 0.0) + (a.elapsed_time(b) if cuda else (b - a) * 1e3)
        return out


def _mean(x: list) -> float | None:
    return float(sum(x) / len(x)) if x else None


class BaselineHook:
    name = ""
    wants_text = False

    def __init__(self, dense_steps: int = DENSE_STEPS, dense_layers: int = DENSE_LAYERS, lse: bool = False):
        self.ds, self.dl, self.lse = int(dense_steps), int(dense_layers), bool(lse)
        self.seg = Sections()
        self.calls = {"sparse": 0, "dense": 0}
        self._acc: list = []

    def reset(self) -> None:
        self.seg = Sections()
        self.calls = {"sparse": 0, "dense": 0}
        self._acc = []

    def is_dense(self, ctx) -> bool:
        if ctx.step < self.ds or ctx.layer < self.dl:
            self.calls["dense"] += 1
            return True
        return False

    def record(self, real, executed, total, pseudo=0.0) -> None:
        self.calls["sparse"] += 1
        self._acc.append((real, executed, pseudo, total))

    def summary(self) -> dict:
        real = [float(r) / n for r, _, _, n in self._acc]
        exe = [float(e) / n for _, e, _, n in self._acc]
        pseudo = [float(p) / n for _, _, p, n in self._acc]
        return {"method": self.name, "sparse_calls": self.calls["sparse"], "dense_calls": self.calls["dense"],
                "exec_density": _mean(exe), "real_density": _mean(real), "pseudo_density": _mean(pseudo),
                "sections_ms": self.seg.ms()}


class PerBranch:
    def __init__(self, factory, branches=("cond", "uncond")):
        self.m = {b: factory() for b in branches}
        first = next(iter(self.m.values()))
        self.name, self.wants_text = first.name, first.wants_text

    def __call__(self, q, k, v, ctx):
        return self.m[ctx.branch](q, k, v, ctx)

    def reset(self) -> None:
        for m in self.m.values():
            m.reset()

    def summary(self) -> dict:
        parts = [m.summary() for m in self.m.values()]
        n = [p["sparse_calls"] for p in parts]
        out = {"method": self.name, "sparse_calls": sum(n), "dense_calls": sum(p["dense_calls"] for p in parts),
               "per_branch": True}
        for key in ("exec_density", "real_density", "pseudo_density"):
            vals = [(p[key], w) for p, w in zip(parts, n) if p[key] is not None and w]
            out[key] = sum(x * w for x, w in vals) / sum(w for _, w in vals) if vals else None
        sec: dict[str, float] = {}
        for p in parts:
            for s, ms in p["sections_ms"].items():
                sec[s] = sec.get(s, 0.0) + ms
        out["sections_ms"] = sec
        return out


def next_point(tried: list, target: float, lo: float, hi: float, start: float, sign: int) -> float:
    if not tried:
        return start
    below = [t for t in tried if t[1] < target]
    above = [t for t in tried if t[1] > target]
    if below and above:
        q0, e0 = min(below, key=lambda t: abs(t[1] - target))
        q1, e1 = min(above, key=lambda t: abs(t[1] - target))
        return q0 + (target - e0) * (q1 - q0) / (e1 - e0)
    if above:
        q = min(above, key=lambda t: t[1])[0]
        return q - sign * 0.5 * abs(q - (lo if sign > 0 else hi))
    q = max(below, key=lambda t: t[1])[0]
    return q + sign * 0.5 * abs((hi if sign > 0 else lo) - q)


def bisect_point(tried: list, target: float, lo: float, hi: float, sign: int) -> float:
    for p, d in tried:
        if (d > target) == (sign > 0):
            hi = p
        else:
            lo = p
    return (lo + hi) / 2


def match_density(evaluate, lo: float, hi: float, start: float, sign: int = 1, iters: int = 6, target: float = 0.200,
                  band: tuple[float, float] = DENSITY_BAND, tried: list | None = None, save=None,
                  scheme: str = "interp") -> dict:
    tried = [tuple(t) for t in (tried or [])]
    for _ in range(iters - len(tried)):
        if any(band[0] <= d <= band[1] for _, d in tried):
            break
        if scheme == "bisect":
            nxt = bisect_point(tried, target, lo, hi, sign)
        else:
            nxt = next_point(tried, target, lo, hi, start, sign)
        p = round(min(hi, max(lo, nxt)), 4)
        if any(abs(p - q) < 5e-5 for q, _ in tried):
            break
        tried.append((p, float(evaluate(p))))
        if save is not None:
            save({"tried": tried, "chosen": None})
    best = min(tried, key=lambda t: abs(t[1] - target))
    return {"tried": tried, "chosen": best[0], "exec_density": best[1], "in_band": band[0] <= best[1] <= band[1]}


class MatchedBaseline(Baseline):
    knob = ""
    knob_sign = 1

    def make_hook(self, adapter, settings: dict):
        raise NotImplementedError

    def pmatch_file(self, run_dir: str) -> str:
        return os.path.join(self.work_dir(run_dir), "pmatch.json")

    def matched(self, run_dir: str | None) -> float | None:
        if run_dir and os.path.exists(self.pmatch_file(run_dir)):
            with open(self.pmatch_file(run_dir)) as f:
                return json.load(f).get("chosen")
        return None

    def hook(self, adapter, run_dir: str | None = None):
        s = dict(self.settings)
        m = self.matched(run_dir)
        if m is not None:
            s[self.knob] = m
        return self.make_hook(adapter, s)

    def prepare(self, adapter, run_dir: str, device: str = "cuda", log=print) -> None:
        if self.settings.get("rematch"):
            self.rematch(adapter, run_dir, log=log)

    def rematch(self, adapter, run_dir: str, seed: int = SEED, log=print) -> dict:
        s, path = self.settings, self.pmatch_file(run_dir)
        state = {"tried": []}
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            if state.get("chosen") is not None:
                return state
        clips = adapter.clips("calib")[: int(s.get("match_clips", 2))]
        getattr(adapter, "ensure_loaded", lambda: None)()

        def evaluate(value: float) -> float:
            hook = self.make_hook(adapter, {**s, self.knob: value})
            dens = []
            for c in clips:
                hook.reset()
                adapter.generate(c, hook, seed)
                dens.append(hook.summary()["exec_density"])
                log(f"{self.name} {self.knob}={value:.4f} {c}: executed density {dens[-1]:.4f}")
            return sum(dens) / len(dens)

        def save(st: dict) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                json.dump(st, f, indent=1)
            os.replace(path + ".tmp", path)

        lo, hi = s["match_range"]
        res = match_density(evaluate, lo, hi, s.get("match_start", s[self.knob]), self.knob_sign,
                            int(s.get("match_iters", 6)), band=tuple(s.get("match_band", DENSITY_BAND)),
                            tried=state.get("tried"), save=save, scheme=s.get("match_scheme", "interp"))
        res.update(param=self.knob, clips=clips)
        save(res)
        log(f"{self.name}: {self.knob} = {res['chosen']} (executed density {res['exec_density']:.4f})")
        return res
