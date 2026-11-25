from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from ..constants import BUILD_STEP, CLUSTERS, KMEANS_ITERS, OPTIONS, REASSIGN_THRESHOLD_DB, SEED, SPEED_THRESHOLD
from .units import Units

STALE_CLIPS = 2
STALE_MASS_LEVEL = 0.20


def rmse(mse) -> float:
    a = np.asarray(list(mse), dtype=np.float64)
    return float(np.sqrt(a.mean())) if a.size else float("nan")


def db(a: float, b: float) -> float:
    return 20 * math.log10(a / b) if a > 0 and b > 0 else float("nan")


def head_mse(x, o, H) -> list[float]:
    xf = x[0].float().view(x.shape[1], H, -1)
    of = o[0].float().view(o.shape[1], H, -1)
    return (xf - of).pow(2).mean((0, 2)).tolist()


def static_masks(mean, til, level=STALE_MASS_LEVEL):
    from ..kernels.block_mass import build_calib

    P, L = mean.shape[:2]
    out = torch.ones(mean.shape, dtype=torch.bool)
    for r in range(P):
        if not bool(mean[r, 1:].any()):
            continue
        for l in range(1, L):
            out[r, l] = build_calib(mean[r, l].to(til.n_real.device).float()[None], til.n_real, level)[0][0].cpu()
    return out


class StaleProbe:
    def __init__(self, adapter, units: Units, til, masks, steps, build_step=BUILD_STEP):
        assert build_step in steps
        self.adapter, self.units, self.til, self.masks = adapter, units, til, masks
        self.steps, self.b = sorted(steps), build_step
        self.slot = (lambda s: s) if units.layout == "steps" else (lambda s: int(units.step_row[s]))
        self.plans, self.built, self.rec = {}, {}, []
        self.clip = None

    def _plan(self, row, layer):
        from ..kernels.block_sparse import SpatialPlan

        p = self.plans.get((row, layer))
        if p is None:
            p = self.plans[(row, layer)] = SpatialPlan(self.masks[row, layer], self.til)
        return p

    @torch.no_grad()
    def __call__(self, q, k, v, ctx):
        from ..attention import decoup_cp_attn
        from ..clustering import build_clusters, reassign_labels

        step, layer, branch = ctx.step, ctx.layer, ctx.branch
        if self.clip is None or branch != "cond" or step not in self.steps or layer < 1:
            return
        til, H = self.til, q.shape[2]
        if step == self.b:
            self.built[(layer, branch)] = build_clusters(k, til, CLUSTERS, KMEANS_ITERS)
            return
        built = self.built.get((layer, branch))
        if built is None:
            return
        frozen, centroids = built
        row = self.slot(step)
        Ms = self.masks[row, layer].to(q.device)
        sp = self._plan(row, layer)
        o = self.adapter.dense(q, k, v)
        rec = {"clip": self.clip, "step": step, "layer": layer, "branch": branch,
               "frozen": head_mse(decoup_cp_attn(q, k, v, sp, frozen, Ms, til, True, True), o, H)}
        rec["reassign"] = head_mse(decoup_cp_attn(q, k, v, sp, reassign_labels(k, centroids, til, CLUSTERS), Ms, til,
                                                  True, True), o, H)
        self.rec.append(rec)


def stale_db(records) -> float:
    return db(rmse(x for r in records for x in r["frozen"]), rmse(x for r in records for x in r["reassign"]))


def measure_stale(adapter, til, mean, out_dir: str, steps=None, n_clips: int = STALE_CLIPS) -> float:
    units = Units.of(adapter)
    steps = list(steps or units.rows)
    if BUILD_STEP not in steps:
        steps = [BUILD_STEP] + steps
    clips = list(adapter.clips("calib")[:n_clips])
    os.makedirs(out_dir, exist_ok=True)
    probe = StaleProbe(adapter, units, til, static_masks(mean, til), steps)
    recs = []
    for c in clips:
        path = f"{out_dir}/stale_{c}.jsonl"
        if not os.path.exists(path):
            probe.clip, probe.built, probe.rec = c, {}, []
            adapter.generate(c, probe, seed=SEED, steps=max(steps) + 1)
            probe.clip, probe.built = None, {}
            with open(path + ".tmp", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in probe.rec)
            os.replace(path + ".tmp", path)
        with open(path) as f:
            recs += [json.loads(x) for x in f]
    return stale_db(recs)


def reassign_rule(delta_stale_db: float) -> bool:
    return delta_stale_db > REASSIGN_THRESHOLD_DB


def speed_check(smax: float | None) -> str:
    if not smax:
        return "pending"
    return "sparse" if smax >= SPEED_THRESHOLD else "dense"


def smax_plan(units: Units, D, block_shape, reassign):
    from .build import build_plan

    ch = np.full(len(units), OPTIONS.index("Static05"), dtype=np.int64)
    est = np.ones(units.n_calls, dtype=np.int64)
    return build_plan(units, D, block_shape, ch, est, reassign)
