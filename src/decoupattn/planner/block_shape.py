from __future__ import annotations

import numpy as np
import torch

from ..constants import BLOCK_SHAPE_MARGIN_DB, BLOCK_SHAPE_MASS_LEVEL, BLOCK_SHAPES, SEED, TARGET_DENSITY
from ..plan import KIND_DENSE, KIND_STATIC, Plan
from .units import Units

BLOCK_SHAPE_CLIPS = 8
BLOCK_SHAPE_VAL_CLIPS = 8


DEFAULT_CANDIDATES = ((1, 8, 16), (1, 16, 8), (2, 8, 8), (2, 4, 16), (4, 4, 8))


def best_tile(F, H, Wd):
    def pad(t):
        return (-(-F // t[0]) * t[0]) * (-(-H // t[1]) * t[1]) * (-(-Wd // t[2]) * t[2])

    return min(DEFAULT_CANDIDATES, key=pad)


def shape_name(t) -> str:
    return "x".join(str(int(x)) for x in t)


def calib_at_exec(bm, til, target=TARGET_DENSITY):
    from ..kernels.block_mass import build_calib
    from ..layout import exec_density

    lo, hi = 1e-4, 1.0
    best = None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        mk = build_calib(bm[None], til.n_real, mid)[0][0].to(bm.device)
        ed = float(exec_density(mk, til))
        if ed <= target:
            lo, best = mid, (mk, ed, mid)
        else:
            hi = mid
    return best


class MassHook:
    def __init__(self, units: Units, tilings: dict):
        self.units, self.tilings = units, tilings
        self.acc = {t: {} for t in tilings}
        self.on = False

    @torch.no_grad()
    def __call__(self, q, k, v, ctx):
        from ..kernels.block_mass import block_mass

        if self.on and ctx.branch == "cond" and ctx.step in self.units.rows and ctx.layer >= 1:
            key = (ctx.step, ctx.layer)
            for t, til in self.tilings.items():
                bm = block_mass(q, k, til).float().cpu()
                if key in self.acc[t]:
                    self.acc[t][key] += bm
                else:
                    self.acc[t][key] = bm


def probe(adapter, n_clips=BLOCK_SHAPE_CLIPS, shapes=BLOCK_SHAPES, device="cuda"):
    clips = list(adapter.clips("calib")[:n_clips])
    if len(clips) < n_clips:
        raise ValueError(f"block shape: {n_clips} calibration clips requested, the split has {len(clips)}")
    from ..kernels.block_mass import build_calib
    from ..layout import Tiling

    units = Units.of(adapter)
    A = tuple(best_tile(*adapter.latent_grid))
    cands = [A] + [tuple(t) for t in shapes if tuple(t) != A]
    tils = {t: Tiling(*adapter.latent_grid, t, device=device) for t in cands}
    hook = MassHook(units, tils)
    for c in clips:
        hook.on = True
        adapter.generate(c, hook, seed=SEED, steps=units.observed_steps)
        hook.on = False
    L, H = units.n_layers, units.n_heads
    steps = units.layout == "steps"
    nslot = 4 if steps else units.n_rows
    ret, means = {}, {}
    for t in cands:
        til = tils[t]
        nq = til.n_real.double()
        mean = torch.zeros(nslot, L, H, til.nt, til.nt, dtype=torch.float16)
        rr = []
        for (s, l), sm in hook.acc[t].items():
            m = sm / len(clips)
            mean[s if steps else units.rows.index(s), l] = m.half()
            mk = build_calib(m.to(device)[None], til.n_real, BLOCK_SHAPE_MASS_LEVEL)[0][0].double().cpu()
            rr.append(float((((m.double() * mk).sum(-1) * nq.cpu()).sum(-1) / nq.sum().cpu()).mean()))
        hook.acc[t] = None
        ret[shape_name(t)] = float(np.mean(rr))
        means[t] = mean
    B = max((t for t in cands if t != A), key=lambda t: ret[shape_name(t)])
    rec = {"default": shape_name(A), "challenger": shape_name(B), "kept_mass": ret,
           "padding": {shape_name(t): tils[t].S_pad / tils[t].S - 1 for t in cands}, "clips": clips, "seed": SEED}
    return rec, {A: means[A], B: means[B]}, {A: tils[A], B: tils[B]}


def val_plan(units: Units, mean, til, device="cuda") -> tuple[Plan, float, float]:
    L, H, nt = units.n_layers, units.n_heads, til.nt
    steps = units.layout == "steps"
    P = 4 if steps else units.n_rows
    slots = list(units.rows) if steps else range(units.n_rows)
    mask = np.ones((P, L, H, nt, nt), bool)
    ed, rd = np.zeros((P, L)), np.zeros((P, L))
    for r in slots:
        for l in range(1, L):
            mk, e, d = calib_at_exec(mean[r, l].to(device).float(), til)
            mask[r, l] = mk.cpu().numpy()
            ed[r, l], rd[r, l] = e, d
    si = np.arange(4) if steps else units.step_row
    kind_r = np.full((P, L, H), KIND_STATIC, np.int8)
    kind_r[:, 0] = KIND_DENSE
    if steps:
        kind_r[0] = KIND_DENSE
    kind = kind_r[si].copy()
    kind[0] = KIND_DENSE
    K = mask.sum(-1).max(-1)[si].astype(np.int16)
    est = np.ones(kind.shape[:2], np.int8)
    plan = Plan(mask=mask, step_row=si, block_shape=tuple(til.tile), kind=kind, k_per_head=K, estimator=est,
                rows=None if steps else np.array(units.rows), row_weight=None if steps else units.row_weight,
                dynamic_blocks=np.zeros(kind.shape, np.int16), extra={"density_per_row_layer": rd})
    e_, r_ = (ed[1:, 1:], rd[1:, 1:]) if steps else (ed[:, 1:], rd[:, 1:])
    return plan, float(e_.mean()), float(r_.mean())


def paired(psnr_a, psnr_b) -> dict:
    a, b = np.asarray(psnr_a, float), np.asarray(psnr_b, float)
    g = b - a
    return {"psnr_A": float(a.mean()), "psnr_B": float(b.mean()), "gain": float(g.mean()),
            "gain_se": float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None, "wins_B": f"{int((g > 0).sum())}/{len(g)}",
            "n": len(g)}


def decide(A: str, B: str, cmp: dict, ms: dict | None = None) -> tuple[dict, bool]:
    d = cmp["gain"]
    out = {"A": A, "B": B, **cmp}
    if d >= BLOCK_SHAPE_MARGIN_DB:
        out.update(shape=B, reason=f"B - A >= {BLOCK_SHAPE_MARGIN_DB} dB")
    elif d <= -BLOCK_SHAPE_MARGIN_DB:
        out.update(shape=A, reason=f"A - B >= {BLOCK_SHAPE_MARGIN_DB} dB")
    else:
        if ms is None:
            return out, True
        out.update(ms_A=ms[A], ms_B=ms[B], shape=A if ms[A] <= ms[B] else B, reason="tie -> faster")
    return out, False


def parse(name: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in str(name).split("x"))
