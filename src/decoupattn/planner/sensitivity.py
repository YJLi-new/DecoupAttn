from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..config import repo_path
from ..constants import DISTORTION_CLIPS, OPTIONS, SEED, SENS_REF_LEVEL
from ..plan import KIND_DENSE, Plan
from .build import build_plan
from .units import Units, step_rows, thirds

SENS_CLIPS = DISTORTION_CLIPS
SPACE = "latent"
FLOOR = "dense"
I_REF = OPTIONS.index(f"Static{round(100 * SENS_REF_LEVEL):02d}")


def step_row_steps(units: Units) -> np.ndarray:
    if units.layout == "steps":
        return np.arange(1, units.steps)
    return np.asarray(units.rows)[step_rows(units.steps, units.rows)[1:]]


def group_rows(units: Units, s0: int, s1: int) -> list[int]:
    return [r for r in units.rows if s0 <= r <= s1]


def steps_of_rows(units: Units, rows) -> list[int]:
    return [t + 1 for t, r in enumerate(step_row_steps(units)) if r in rows]


def default_step_groups(units: Units) -> list[tuple[int, int]]:
    return [(min(s), max(s)) for s in (steps_of_rows(units, [r]) for r in units.rows)]


def aligned_step_groups(units: Units, step_groups) -> list[tuple[int, int]]:
    out = []
    for s0, s1 in step_groups:
        s = steps_of_rows(units, group_rows(units, s0, s1))
        if not s:
            raise ValueError(f"sensitivity step group {[s0, s1]} holds no plan row of {units.rows}")
        out.append((min(s), max(s)))
    return out


def check_step_groups(units: Units, step_groups) -> None:
    for (s0, s1), (a0, a1) in zip(step_groups, aligned_step_groups(units, step_groups)):
        if (max(s0, 1), s1) != (a0, a1):
            raise ValueError(f"sensitivity step group {[s0, s1]}: its rows {group_rows(units, s0, s1)} run on steps "
                             f"{a0}..{a1}, so the injected steps would differ from the steps of the denominator; "
                             f"use {[a0, a1]}")


def groups(units: Units, step_groups) -> list[tuple[str, int, int, int, int]]:
    check_step_groups(units, step_groups)
    return [(f"g{gi}_t{ti}", int(s0), int(s1), l0, l1)
            for gi, (s0, s1) in enumerate(step_groups) for ti, (l0, l1) in enumerate(thirds(units.n_layers))]


def _expected(units: Units, step_groups) -> dict:
    return {FLOOR: ([0, 0], [0, 0]), **{g[0]: ([g[1], g[2]], [g[3], g[4]]) for g in groups(units, step_groups)}}


def _current(recs, expected) -> list[dict]:
    return [x for x in recs if x["group"] in expected and (x.get("steps"), x.get("layers")) == expected[x["group"]]]


def group_plan(units: Units, D, block_shape, group, reassign, extra: dict | None = None) -> Plan:
    _, s0, s1, l0, l1 = group
    ch = np.full(len(units), I_REF, np.int64)
    p = build_plan(units, D, block_shape, ch, np.ones(units.n_calls, np.int64), reassign, extra)
    t = np.arange(p.n_steps)
    on = (t >= max(s0, 1)) & (t <= s1)
    if units.layout == "steps":
        on &= np.isin(t, units.rows)
    lay = np.arange(p.n_layers)
    sel = on[:, None] & ((lay >= l0) & (lay <= l1))[None, :]
    p.kind = np.where(sel[:, :, None], p.kind, KIND_DENSE).astype(np.int8)
    return p


def _records(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f]


def _write_records(path, recs) -> None:
    with open(path + ".tmp", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in recs)
    os.replace(path + ".tmp", path)


def latent_records(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    recs = _records(path)
    return recs if all(x.get("space") == SPACE for x in recs) else []


def pending(adapter, path, step_groups=None, n_clips: int = SENS_CLIPS) -> list[str]:
    units = Units.of(adapter)
    expected = _expected(units, step_groups or default_step_groups(units))
    done = {(x["group"], x["clip"]) for x in _current(latent_records(path), expected)}
    return [c for c in adapter.clips("calib")[:n_clips] if any((g, c) not in done for g in expected)]


def _mse(a, ref) -> float:
    return float(((a.float() - ref) ** 2).mean())


def measure(adapter, D, block_shape, reassign, out_dir: str, step_groups=None, n_clips: int = SENS_CLIPS,
            tied: bool = False, extra: dict | None = None) -> str:
    from ..runtime import runtime_for

    out = f"{out_dir}/J_s{SEED}.jsonl"
    units = Units.of(adapter)
    step_groups = step_groups or default_step_groups(units)
    G = groups(units, step_groups)
    os.makedirs(out_dir, exist_ok=True)
    recs = latent_records(out)

    def latent(clip, g):
        hook = None
        if g is not None:
            hook = runtime_for(adapter, group_plan(units, D, block_shape, g, reassign, extra), fast=False, tied=tied)
        return adapter.generate(clip, hook, seed=SEED).float()

    every = [(FLOOR, 0, 0, 0, 0)] + G
    for c in pending(adapter, out, step_groups, n_clips):
        recs = [x for x in recs if x["clip"] != c]
        _write_records(out, recs)
        ref = latent(c, None)
        rows = []
        for g in every:
            lat = latent(c, g if g[0] != FLOOR else None)
            rows.append({"group": g[0], "clip": c, "space": SPACE, "mse": _mse(lat, ref), "steps": [g[1], g[2]],
                         "layers": [g[3], g[4]]})
            torch.cuda.empty_cache()
        recs += rows
        _write_records(out, recs)
    return out


def from_injection(units: Units, path: str, E_full: np.ndarray, step_groups=None,
                   clips=None) -> tuple[np.ndarray, np.ndarray]:
    step_groups = step_groups or default_step_groups(units)
    U, wu = units.units, units.weights
    expected = _expected(units, step_groups)
    recs = [x for x in _current(latent_records(path), expected) if clips is None or x["clip"] in clips]
    if not recs:
        raise ValueError(f"{path}: no final-latent sensitivity records of these groups (the sensitivity stage "
                         "measures them)")
    by = {}
    for x in recs:
        by.setdefault(x["group"], []).append(x)
    missing = [g for g in expected if g not in by]
    if missing:
        raise ValueError(f"{path} has no records of the groups {missing} with their current steps and layers")
    fl = float(np.mean([x["mse"] for x in by[FLOOR]]))
    st = units.unit_steps
    J = np.zeros(len(U))
    jh = np.zeros((len(step_groups), 3))
    for gi, (s0, s1) in enumerate(step_groups):
        rows = group_rows(units, s0, s1)
        for ti, (l0, l1) in enumerate(thirds(units.n_layers)):
            m = np.isin(st, rows) & (U[:, 1] >= l0) & (U[:, 1] <= l1)
            mse = float(np.mean([x["mse"] for x in by[f"g{gi}_t{ti}"]]))
            jh[gi, ti] = max(mse - fl, 0.0) / (wu[m] * E_full[m, I_REF]).sum()
            J[m] = jh[gi, ti]
    assert (J > 0).all(), f"a sensitivity group has no positive excess over the floor: {jh}"
    return J / J.mean(), jh


def uniform(units: Units) -> np.ndarray:
    return np.ones(len(units))


def load(units: Units, source: dict, E_full: np.ndarray | None = None) -> np.ndarray:
    kind = source.get("kind", "injection")
    if kind == "uniform":
        return uniform(units)
    if kind != "injection":
        raise ValueError(f"unknown sensitivity source {kind!r}")
    sg = [tuple(x) for x in source["step_groups"]] if source.get("step_groups") else None
    return from_injection(units, repo_path(source["path"]), E_full, sg, source.get("clips"))[0]
