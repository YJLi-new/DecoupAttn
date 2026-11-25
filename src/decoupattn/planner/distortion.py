from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..constants import DISTORTION_CLIPS, ESTIMATORS, LEVELS, OPTIONS, SEED
from ..layout import exec_density_heads, own_mask, pair_weights, real_density_heads
from .dtable import rank_of
from .units import Units, cluster_cache, k_blocks

COLS = OPTIONS[1:]
FLAGS = {"Full": (True, True), "Centroid": (False, False)}


class DistortionHook:
    def __init__(self, adapter, units: Units, til, D, labeler, estimators=ESTIMATORS):
        self.adapter, self.units, self.til, self.D, self.labeler = adapter, units, til, D, labeler
        self.estimators = tuple(estimators)
        self.Ks = [k_blocks(r, til.nt) for r in LEVELS] + [1]
        self.own = own_mask(til)
        self.nr = til.n_real.double().to(self.own.device)
        self.w = pair_weights(til).to(self.own.device)
        self.gram = {}
        self.r = None
        self.t_unit = []

    def begin(self):
        L, H, R = self.units.n_layers, self.units.n_heads, self.units.n_rows
        nk, ne = len(self.Ks), len(self.estimators)

        def z(*s):
            return np.zeros((R, L) + s, np.float64)

        self.r = {"pd2": z(ne, nk, H), "d2": z(ne, nk, H), "xd": z(nk, H), "rdn": z(nk, H), "pon2": z(H), "cnt": np.zeros((R, L))}
        self.t_unit = []
        self.labeler.reset()

    def _gram(self, layer, step, H):
        key = (layer, self.adapter.expert(step))
        if key not in self.gram:
            Wo = self.adapter.out_proj(layer, step).float()
            Wh = Wo.view(Wo.shape[0], H, -1).permute(1, 0, 2)
            self.gram[key] = torch.bmm(Wh.transpose(1, 2), Wh).contiguous()
        return self.gram[key]

    @torch.no_grad()
    def __call__(self, q, k, v, ctx):
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        from ..attention import decoup_cp_attn, spatial_attn
        from ..kernels.block_sparse import SpatialPlan

        if self.r is None or ctx.layer < 1:
            return
        r0 = self.units.table_row(ctx.step, ctx.branch)
        if r0 is None:
            return
        t0 = time.time()
        step, layer, til, r = ctx.step, ctx.layer, self.til, self.r
        lab = None if self.estimators == ("none",) else self.labeler.get(k, step, layer, ctx.branch)
        S, Hh, Dd = q.shape[1], q.shape[2], q.shape[3]
        G = self._gram(layer, step, Hh)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            O = F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in (q, k, v)))[0]
        torch.backends.cuda.matmul.allow_tf32 = prev
        r["pon2"][r0, layer] += (G * torch.bmm(O.transpose(1, 2), O)).sum((1, 2)).double().cpu().numpy()
        r["cnt"][r0, layer] += 1
        rk = rank_of(torch.from_numpy(self.D[r0, layer]).to(q.device), self.own)
        for li, K in enumerate(self.Ks):
            m = rk < K
            r["xd"][r0, layer, li] = exec_density_heads(m, self.nr, til.S).cpu().numpy()
            r["rdn"][r0, layer, li] = real_density_heads(m, self.w, til.S).cpu().numpy()
            spn = SpatialPlan(m, til)
            for ei, est in enumerate(self.estimators):
                if est == "none":
                    o = spatial_attn(q, k, v, spn, til, False)
                else:
                    wl, pisa = FLAGS[est]
                    o = decoup_cp_attn(q, k, v, spn, lab, m, til, wl, pisa)
                d = o[0].view(S, Hh, Dd).transpose(0, 1).float() - O
                r["d2"][r0, layer, ei, li] += (d.double() ** 2).sum((1, 2)).cpu().numpy()
                r["pd2"][r0, layer, ei, li] += (G * torch.bmm(d.transpose(1, 2), d)).sum((1, 2)).double().cpu().numpy()
                del o, d
            del spn, m
        del O
        self.t_unit.append(time.time() - t0)

    def end(self) -> dict:
        r, self.r = self.r, None
        cnt = r.pop("cnt")
        nb = len(self.units.measured_branches)
        assert (cnt[:, 1:] == nb).all(), f"calls per (row, layer) {set(cnt[:, 1:].ravel())} != {nb}"
        n = np.maximum(cnt, 1)
        for kk in ("pd2", "d2"):
            r[kk] = r[kk] / n[:, :, None, None, None]
        r["pon2"] = r["pon2"] / n[:, :, None]
        r["pd2"][:, 0] = np.nan
        return r


def measure(adapter, til, D, reassign, out_dir: str, n_clips: int = DISTORTION_CLIPS, estimators=ESTIMATORS,
            tied: bool = False) -> list[str]:
    units = Units.of(adapter)
    clips = list(adapter.clips("calib")[:n_clips])
    os.makedirs(out_dir, exist_ok=True)
    hook = DistortionHook(adapter, units, til, D, cluster_cache(til, reassign, tied), estimators)
    files = []
    for c in clips:
        path = f"{out_dir}/{c}_s{SEED}.npz"
        files.append(path)
        if os.path.exists(path):
            continue
        hook.begin()
        adapter.generate(c, hook, seed=SEED, steps=units.observed_steps)
        r = hook.end()
        np.savez(path[:-4] + ".tmp.npz", cols=np.array(COLS), comps=np.array(hook.estimators), levels=np.array(LEVELS),
                 K=np.array(hook.Ks), seed=SEED, tile=np.array(til.tile), refresh=np.array(bool(reassign)),
                 rows=np.array(units.rows), branches=np.array(units.measured_branches), t_unit=np.array(hook.t_unit), **r)
        os.replace(path[:-4] + ".tmp.npz", path)
    return files


def tables_from(units: Units, Z) -> tuple[dict, np.ndarray, np.ndarray]:
    U = units.units
    nO = len(OPTIONS)
    E = {c: np.zeros((len(U), nO)) for c in ESTIMATORS}
    X = np.ones((len(U), nO))
    RD = np.ones((len(U), nO))
    pd2 = np.mean([z["pd2"] for z in Z], 0)
    if "comps" in getattr(Z[0], "files", Z[0]) and len(Z[0]["comps"]) == 1:
        pd2 = np.repeat(pd2, len(ESTIMATORS), axis=2)
    xd = np.mean([z["xd"] for z in Z], 0)
    rdn = np.mean([z["rdn"] for z in Z], 0)
    r, l, h = U[:, 0], U[:, 1], U[:, 2]
    for j in range(1, nO):
        for ci, c in enumerate(ESTIMATORS):
            E[c][:, j] = pd2[r, l, ci, j - 1, h]
        X[:, j] = xd[r, l, j - 1, h]
        RD[:, j] = rdn[r, l, j - 1, h]
    assert all(np.isfinite(E[c]).all() for c in ESTIMATORS) and np.isfinite(X).all()
    return E, X, RD


def load_tables(units: Units, files: list[str]):
    return tables_from(units, [np.load(f) for f in sorted(files)])
