from __future__ import annotations

import torch

from ._triton import jit, tl


@jit
def _assign(X, CEN, C2, OUT, S, G, D: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BN + tl.arange(0, BN)
    rm = rows < S
    d = tl.arange(0, D)
    x = tl.load(X + (h * S + rows[:, None]).to(tl.int64) * D + d[None, :], mask=rm[:, None], other=0.0)
    best = tl.full([BN], float("inf"), tl.float32)
    bi = tl.zeros([BN], tl.int32)
    for k0 in range(0, G, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < G
        c = tl.load(CEN + (h * G + ks[None, :]) * D + d[:, None], mask=km[None, :], other=0.0)
        c2 = tl.load(C2 + h * G + ks, mask=km, other=float("inf"))
        dist = c2[None, :] - 2.0 * tl.dot(x, c)
        cm = tl.min(dist, 1)
        ci = tl.argmin(dist, 1)
        upd = cm < best
        best = tl.where(upd, cm, best)
        bi = tl.where(upd, k0 + ci, bi)
    tl.store(OUT + h * S + rows, bi, mask=rm)


@jit
def _seg_stats(X, IDX, OFF, OLD, MEAN, VAR, S, G, D: tl.constexpr, BR: tl.constexpr, WITH_VAR: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    o0 = tl.load(OFF + h * (G + 1) + c)
    o1 = tl.load(OFF + h * (G + 1) + c + 1)
    d = tl.arange(0, D)
    s1 = tl.zeros([D], tl.float32)
    s2 = tl.zeros([D], tl.float32)
    for r0 in range(o0, o1, BR):
        rows = r0 + tl.arange(0, BR)
        m = rows < o1
        src = tl.load(IDX + h * S + rows, mask=m, other=0).to(tl.int64)
        x = tl.load(X + (h * S + src[:, None]) * D + d[None, :], mask=m[:, None], other=0.0)
        s1 += tl.sum(x, 0)
        if WITH_VAR:
            s2 += tl.sum(x * x, 0)
    n = (o1 - o0).to(tl.float32)
    inv = 1.0 / tl.maximum(n, 1.0)
    mean = s1 * inv
    o = (h * G + c) * D + d
    if WITH_VAR:
        tl.store(MEAN + o, mean)
        tl.store(VAR + o, tl.maximum(s2 * inv - mean * mean, 0.0))
    else:
        old = tl.load(OLD + o)
        tl.store(MEAN + o, tl.where(n > 0, mean, old))


def assign(xa, cen, lab32):
    H, S, D = xa.shape
    G = cen.shape[1]
    ca = cen.bfloat16() if xa.dtype == torch.bfloat16 else cen
    _assign[(-(-S // 64), H)](xa, ca, (cen * cen).sum(-1).contiguous(), lab32, S, G, D=D, BN=64, BK=64, num_warps=4,
                              num_stages=2)


def seg_stats(x, lab, G, old=None, with_var=False):
    H, S, D = x.shape
    lab = lab.long()
    idx = torch.sort(lab, dim=1).indices.to(torch.int32).contiguous()
    cnt = torch.zeros(H, G, dtype=torch.long, device=x.device).scatter_add_(1, lab, torch.ones_like(lab))
    off = torch.cat([cnt.new_zeros(H, 1), torch.cumsum(cnt, 1)], 1).to(torch.int32).contiguous()
    mean = torch.empty(H, G, D, device=x.device)
    var = torch.empty(H, G, D, device=x.device) if with_var else mean
    _seg_stats[(G, H)](x, idx, off, old if old is not None else mean, mean, var, S, G, D=D, BR=32, WITH_VAR=with_var,
                       num_warps=4)
    return mean, (var if with_var else None), cnt
