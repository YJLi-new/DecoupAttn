from __future__ import annotations

import math
import os

import numpy as np
import torch

from ..compensation import head_stats
from ..constants import CLUSTERS, DTABLE_CLIPS, SEED
from .units import Units, cluster_cache

NEG = float("-inf")


def rank_of(Dm: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
    b = Dm.float().masked_fill(own[None], float("inf"))
    order = torch.argsort(b, dim=-1, descending=True)
    rk = torch.empty_like(order)
    rk.scatter_(-1, order, torch.arange(b.shape[-1], device=b.device).expand_as(order).contiguous())
    return rk


@torch.no_grad()
def residual_map(q, k, v, til, lab, C=CLUSTERS, budget=48e6):
    H, D = q.shape[2], q.shape[3]
    NT, Sp = til.nt, til.S_pad
    dev = q.device
    sc = 1.0 / math.sqrt(D)
    src = til.slot_src.clamp(min=0)
    realb = til.real
    realf = realb.float()
    ntc = max(1, int(budget // (128 * Sp)))
    out = torch.zeros(H, NT, NT, device=dev)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for h in range(H):
            kh, vh = k[0, :, h].float(), v[0, :, h].float()
            lb = lab[h].long()
            _, km, vm, var, hm, _ = head_stats(kh, vh, lb, C)
            kb, vb, kv = km.bfloat16().float(), vm.bfloat16().float(), var.bfloat16().float()
            hms = (hm * sc).T.contiguous()
            Kt, Vt, Lt = kh[src] * realf[:, None], vh[src] * realf[:, None], lb[src]
            Vbt = vb[Lt] * realf[:, None]
            Vt3, Vbt3 = Vt.view(NT, 128, D), Vbt.view(NT, 128, D)
            Qall = q[0, :, h].float()[src] * realf[:, None]
            for t0 in range(0, NT, ntc):
                t1 = min(NT, t0 + ntc)
                n = t1 - t0
                R = n * 128
                rs = slice(t0 * 128, t1 * 128)
                Q, qv = Qall[rs], realf[rs]
                qcount = qv.view(n, 128).sum(1).clamp(min=1)
                s = (Q @ Kt.T) * sc
                s.masked_fill_(~realb[None], NEG)
                lse = torch.logsumexp(s, 1)
                P = s.sub_(lse[:, None]).exp_()
                y = P @ Vt
                yy = y.pow(2).sum(-1).clamp(min=1e-30)
                pcl = ((Q @ kb.T) * sc - lse[:, None]).exp_()
                A = P.view(R, NT, 128)
                bZ = A.sum(-1)
                NV = torch.bmm(A.transpose(0, 1), Vt3)
                Wc = pcl[:, Lt] * realf[None]
                gmass = Wc.view(R, NT, 128).sum(-1)
                hq = Q @ hms
                wcl = pcl * (1.0 + ((Q * Q) @ kv.T) * (0.5 / D))
                B = (wcl[:, Lt] * realf[None]).view(R, NT, 128)
                zH = B.sum(-1)
                NH = torch.bmm(B.transpose(0, 1), Vbt3)
                del B, Wc
                NH += gmass.T[:, :, None] * hq[None]
                dZ = bZ - zH
                dN = NV - NH
                d0 = dN - y[None] * dZ.T[:, :, None]
                x = d0.pow(2).sum(-1) / yy[None]
                out[h, t0:t1] = (x.T.reshape(n, 128, NT) * qv.view(n, 128, 1)).sum(1) / qcount[:, None]
                del d0, x, P, s, A, NV, NH, dN
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    out.diagonal(dim1=1, dim2=2).zero_()
    return out


class DTableHook:
    def __init__(self, units: Units, til, labeler):
        self.units, self.til, self.labeler = units, til, labeler
        self.sum, self.n = {}, {}
        self.on = False

    def __call__(self, q, k, v, ctx):
        if not self.on or ctx.layer < 1:
            return
        r = self.units.table_row(ctx.step, ctx.branch)
        if r is None:
            return
        lab = self.labeler.get(k, ctx.step, ctx.layer, ctx.branch)
        D = residual_map(q, k, v, self.til, lab.lab, lab.C).float().cpu()
        key = (r, ctx.layer)
        if key in self.sum:
            self.sum[key] += D
            self.n[key] += 1
        else:
            self.sum[key], self.n[key] = D, 1


def measure(adapter, til, reassign: bool, out_dir: str, n_clips: int = DTABLE_CLIPS) -> str:
    units = Units.of(adapter)
    clips = list(adapter.clips("calib")[:n_clips])
    os.makedirs(out_dir, exist_ok=True)
    apath = f"{out_dir}/D_acc.pt"
    st = torch.load(apath) if os.path.exists(apath) else {"d": {}, "n": {}, "done": []}
    hook = DTableHook(units, til, cluster_cache(til, reassign))
    hook.sum, hook.n = st["d"], st["n"]
    for c in clips:
        if c in st["done"]:
            continue
        hook.labeler.reset()
        hook.on = True
        adapter.generate(c, hook, seed=SEED, steps=units.observed_steps)
        hook.on = False
        st["done"].append(c)
        torch.save(st, apath + ".tmp")
        os.replace(apath + ".tmp", apath)
    return finish(units, st["d"], st["n"], st["done"], clips, til, reassign, out_dir)


def finish(units: Units, sums, counts, done, clips, til, reassign, out_dir) -> str:
    L, H, R = units.n_layers, units.n_heads, units.n_rows
    D = np.zeros((R, L, H, til.nt, til.nt), np.float32)
    for (r, l), sm in sums.items():
        D[r, l] = (sm / counts[(r, l)]).numpy()
    nb = len(units.measured_branches)
    assert len(counts) == R * (L - 1) and set(counts.values()) == {len(clips) * nb}, (len(counts), set(counts.values()))
    path = f"{out_dir}/D.npz"
    np.savez(path[:-4] + ".tmp.npz", D=D, clips=np.array(done), tile=np.array(til.tile), refresh=np.array(bool(reassign)),
             rows=np.array(units.rows), branches=np.array(units.measured_branches))
    os.replace(path[:-4] + ".tmp.npz", path)
    return path


def load(path: str) -> np.ndarray:
    return np.load(path)["D"]
