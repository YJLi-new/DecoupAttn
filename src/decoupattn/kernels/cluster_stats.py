from __future__ import annotations

import math

import torch

from ._triton import jit, tl


@jit
def _cluster_means(K, V, KB, VB, IDX, OFF, sx_s, sx_h, S, C, D: tl.constexpr, BR: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    o0 = tl.load(OFF + h * (C + 1) + c)
    o1 = tl.load(OFF + h * (C + 1) + c + 1)
    d = tl.arange(0, D)
    ks = tl.zeros([D], tl.float32)
    vs = tl.zeros([D], tl.float32)
    for r0 in range(o0, o1, BR):
        rows = r0 + tl.arange(0, BR)
        m = rows < o1
        src = tl.load(IDX + h * S + rows, mask=m, other=0).to(tl.int64)
        soff = src[:, None] * sx_s + h * sx_h + d[None, :]
        ks += tl.sum(tl.load(K + soff, mask=m[:, None], other=0.0).to(tl.float32), 0)
        vs += tl.sum(tl.load(V + soff, mask=m[:, None], other=0.0).to(tl.float32), 0)
    inv = 1.0 / tl.maximum((o1 - o0).to(tl.float32), 1.0)
    o = (h * C + c) * D + d
    tl.store(KB + o, (ks * inv).to(KB.dtype.element_ty))
    tl.store(VB + o, (vs * inv).to(VB.dtype.element_ty))


def cluster_means(k3, v3, cl):
    H, D, C, S = k3.shape[1], k3.shape[2], cl.C, k3.shape[0]
    kb = torch.empty(H, C, D, dtype=torch.float32, device=k3.device)
    vb = torch.empty_like(kb)
    _cluster_means[(C, H)](k3, v3, kb, vb, cl.k_idx, cl.k_off, k3.stride(0), k3.stride(1), S, C, D=D, BR=32,
                           num_warps=4)
    return kb, vb


@jit
def _stats_fused(K, V, KBF, VBF, KB, VB, KV, IDX, OFF, sx_s, sx_h, S, C, D: tl.constexpr, BR: tl.constexpr,
                 VAR: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    o0 = tl.load(OFF + h * (C + 1) + c)
    o1 = tl.load(OFF + h * (C + 1) + c + 1)
    d = tl.arange(0, D)
    ks = tl.zeros([D], tl.float32)
    vs = tl.zeros([D], tl.float32)
    for r0 in range(o0, o1, BR):
        rows = r0 + tl.arange(0, BR)
        m = rows < o1
        src = tl.load(IDX + h * S + rows, mask=m, other=0).to(tl.int64)
        soff = src[:, None] * sx_s + h * sx_h + d[None, :]
        ks += tl.sum(tl.load(K + soff, mask=m[:, None], other=0.0).to(tl.float32), 0)
        vs += tl.sum(tl.load(V + soff, mask=m[:, None], other=0.0).to(tl.float32), 0)
    inv = 1.0 / tl.maximum((o1 - o0).to(tl.float32), 1.0)
    kb = ks * inv
    vb = vs * inv
    o = (h * C + c) * D + d
    tl.store(KBF + o, kb)
    tl.store(VBF + o, vb)
    tl.store(KB + o, kb.to(tl.bfloat16))
    tl.store(VB + o, vb.to(tl.bfloat16))
    if VAR:
        s2 = tl.zeros([D], tl.float32)
        for r0 in range(o0, o1, BR):
            rows = r0 + tl.arange(0, BR)
            m = rows < o1
            src = tl.load(IDX + h * S + rows, mask=m, other=0).to(tl.int64)
            x = tl.load(K + src[:, None] * sx_s + h * sx_h + d[None, :], mask=m[:, None], other=0.0).to(tl.float32)
            dx = tl.where(m[:, None], x - kb[None, :], 0.0)
            s2 += tl.sum(dx * dx, 0)
        tl.store(KV + o, (s2 * inv).to(tl.bfloat16))


@jit
def _vk_partial(K, V, P, sx_s, sx_h, S, CH, D: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr):
    bd = tl.program_id(0)
    sp = tl.program_id(1)
    h = tl.program_id(2)
    nsp = tl.num_programs(1)
    d = tl.arange(0, D)
    e = bd * BD + tl.arange(0, BD)
    acc = tl.zeros([BD, D], tl.float32)
    s0 = sp * CH
    s1 = tl.minimum(s0 + CH, S)
    for r0 in range(s0, s1, BS):
        rows = r0 + tl.arange(0, BS)
        m = rows < s1
        base = rows.to(tl.int64)[:, None] * sx_s + h * sx_h
        k = tl.load(K + base + d[None, :], mask=m[:, None], other=0.0)
        v = tl.load(V + base + e[None, :], mask=m[:, None], other=0.0)
        acc = tl.dot(tl.trans(v), k, acc)
    po = ((h * nsp + sp) * D + e[:, None]).to(tl.int64) * D + d[None, :]
    tl.store(P + po, acc)


_BUF: dict = {}


def _buf(name, shape, dtype, dev):
    key = (name, dtype, str(dev))
    t = _BUF.get(key)
    n = math.prod(shape)
    if t is None or t.numel() < n:
        t = _BUF[key] = torch.empty(n, dtype=dtype, device=dev)
    return t[:n].view(*shape)


def fast_stats(k3, v3, k_idx, k_off, cnt, C, wl=True, pisa=True):
    S, H, D = k3.shape
    dev = k3.device
    f32, b16 = torch.float32, torch.bfloat16
    r = {"kbf": torch.empty(H, C, D, dtype=f32, device=dev), "vbf": torch.empty(H, C, D, dtype=f32, device=dev),
         "kb": _buf("kb", (H, C, D), b16, dev), "vb": _buf("vb", (H, C, D), b16, dev), "kv": _buf("kv", (H, C, D), b16, dev)}
    _stats_fused[(C, H)](k3, v3, r["kbf"], r["vbf"], r["kb"], r["vb"], r["kv"], k_idx, k_off, k3.stride(0), k3.stride(1),
                         S, C, D=D, BR=32, VAR=wl, num_warps=4)
    r["ht"] = None
    if pisa:
        nsp = max(1, min(64, S // 1024))
        ch = -(-S // nsp)
        P = _buf("vkp", (H, nsp, D, D), f32, dev)
        _vk_partial[(D // 64, nsp, H)](k3, v3, P, k3.stride(0), k3.stride(1), S, ch, D=D, BD=64, BS=64, num_warps=4,
                                       num_stages=3)
        vk = P.sum(1)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            between = torch.bmm((r["vbf"] * cnt.float()[..., None]).transpose(1, 2), r["kbf"])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        r["ht"] = ((vk - between) / (S * math.sqrt(D))).transpose(1, 2).to(b16).contiguous()
    return r
