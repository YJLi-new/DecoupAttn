from __future__ import annotations

import math
import os

import torch

from ._triton import jit, tl

FAST_CFG = {"RB": int(os.environ.get("OCF_RB", "128")), "BN": int(os.environ.get("OCF_BN", "64")),
            "W": int(os.environ.get("OCF_W", "8")), "ST": int(os.environ.get("OCF_ST", "3"))}


@jit
def _cluster_var(K, KB, KV, IDX, OFF, sx_s, sx_h, S, C, D: tl.constexpr, BR: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    o0 = tl.load(OFF + h * (C + 1) + c)
    o1 = tl.load(OFF + h * (C + 1) + c + 1)
    d = tl.arange(0, D)
    kb = tl.load(KB + (h * C + c) * D + d).to(tl.float32)
    s2 = tl.zeros([D], tl.float32)
    for r0 in range(o0, o1, BR):
        rows = r0 + tl.arange(0, BR)
        m = rows < o1
        src = tl.load(IDX + h * S + rows, mask=m, other=0).to(tl.int64)
        x = tl.load(K + src[:, None] * sx_s + h * sx_h + d[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        dx = tl.where(m[:, None], x - kb[None, :], 0.0)
        s2 += tl.sum(dx * dx, 0)
    inv = 1.0 / tl.maximum((o1 - o0).to(tl.float32), 1.0)
    tl.store(KV + (h * C + c) * D + d, s2 * inv)


@jit
def _gk_merge_cp(QP, OS, LSE, KB, VB, KV, HM, L2N, QT, PQ, OUT, LSEO, S, sout_s, slo_h, qk_scale_log2, half_s2, scale,
                 C, NT_: tl.constexpr, D: tl.constexpr, RB: tl.constexpr, BN: tl.constexpr, WL: tl.constexpr,
                 PISA: tl.constexpr, RET_LSE: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * RB + tl.arange(0, RB)
    i = tl.load(QT + pid * RB)
    d = tl.arange(0, D)
    qoff = (h * S + rows[:, None]).to(tl.int64) * D + d[None, :]
    q = tl.load(QP + qoff)
    qf = q.to(tl.float32)
    q2 = (qf * qf).to(tl.bfloat16)
    m_i = tl.full([RB], float("-inf"), tl.float32)
    l_i = tl.zeros([RB], tl.float32)
    g_i = tl.zeros([RB], tl.float32)
    acc = tl.zeros([RB, D], tl.float32)
    for j0 in range(0, C, BN):
        cols = j0 + tl.arange(0, BN)
        cv = cols < C
        koff = (h * C + cols[:, None]) * D + d[None, :]
        kb = tl.load(KB + koff, mask=cv[:, None], other=0.0).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(kb)) * qk_scale_log2
        l2n = tl.load(L2N + (h * NT_ + i) * C + cols, mask=cv, other=float("-inf"))
        s = tl.where((l2n > float("-inf"))[None, :], s + l2n[None, :], float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        if WL:
            kv = tl.load(KV + koff, mask=cv[:, None], other=0.0).to(tl.bfloat16)
            u = tl.dot(q2, tl.trans(kv)) * half_s2
            w = p * (1.0 + u)
        else:
            w = p
        l_i = l_i * alpha + tl.sum(w, 1)
        g_i = g_i * alpha + tl.sum(p, 1)
        vb = tl.load(VB + koff, mask=cv[:, None], other=0.0).to(tl.bfloat16)
        acc = tl.dot(w.to(tl.bfloat16), vb, acc * alpha[:, None])
        m_i = m_new
    if PISA:
        acc += tl.load(HM + qoff).to(tl.float32) * g_i[:, None]
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
    lse_c = tl.where(l_i > 0, m_safe + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    lse_s = tl.load(LSE + h * S + rows)
    mx = tl.maximum(lse_s, lse_c)
    w_s = tl.exp2(lse_s - mx)
    w_c = tl.exp2(lse_c - mx)
    o_s = tl.load(OS + qoff).to(tl.float32)
    o = (o_s * w_s[:, None] + acc * (w_c / tl.maximum(l_i, 1e-30))[:, None]) / (w_s + w_c)[:, None]
    dst = tl.load(PQ + rows)
    tl.store(OUT + dst[:, None] * sout_s + h * D + d[None, :], o.to(OUT.dtype.element_ty))
    if RET_LSE:
        tl.store(LSEO + h * slo_h + dst, (mx + tl.log2(w_s + w_c)) * 0.6931471805599453)


def pisa_h(k3, v3, kb, vb, cnt):
    S = k3.shape[0]
    kt = k3.transpose(0, 1)
    vt = v3.transpose(0, 1)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        vk = torch.bmm(vt.float().transpose(1, 2), kt.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    n = cnt.float()
    between = torch.bmm((vb.float() * n[..., None]).transpose(1, 2), kb.float())
    return ((vk - between) / S).contiguous()


def merge_cp(qp, os_, lse, kb, vb, kv, hm, l2n, qt, pq, out, nt, wl, pisa, lse_out=None):
    H, Sp, D = qp.shape
    C = kb.shape[1]
    ret = lse_out is not None
    _gk_merge_cp[(Sp // 64, H)](qp, os_, lse, kb, vb, kv, hm, l2n, qt, pq, out, lse_out if ret else out, Sp,
                                out.stride(0), lse_out.stride(0) if ret else 0, math.log2(math.e) / math.sqrt(D), 0.5 / D,
                                1.0 / math.sqrt(D), C, NT_=nt, D=D, RB=64, BN=64, WL=wl, PISA=pisa, RET_LSE=ret,
                                num_warps=4, num_stages=2)


@jit
def _merge_fast(QP, OS, LSE, KB, VB, KV, HT, L2N, QT, PQ, OUT, S, sout_s, qk_scale_log2, half_s2, C,
                NT_: tl.constexpr, D: tl.constexpr, RB: tl.constexpr, BN: tl.constexpr, WL: tl.constexpr,
                PISA: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * RB + tl.arange(0, RB)
    i = tl.load(QT + pid * RB)
    d = tl.arange(0, D)
    qoff = (h * S + rows[:, None]).to(tl.int64) * D + d[None, :]
    q = tl.load(QP + qoff)
    qf = q.to(tl.float32)
    q2 = (qf * qf).to(tl.bfloat16)
    m_i = tl.full([RB], float("-inf"), tl.float32)
    l_i = tl.zeros([RB], tl.float32)
    g_i = tl.zeros([RB], tl.float32)
    acc = tl.zeros([RB, D], tl.float32)
    for j0 in range(0, C, BN):
        cols = j0 + tl.arange(0, BN)
        cv = cols < C
        koff = (h * C + cols[:, None]) * D + d[None, :]
        kb = tl.load(KB + koff, mask=cv[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kb)) * qk_scale_log2
        l2n = tl.load(L2N + (h * NT_ + i) * C + cols, mask=cv, other=float("-inf"))
        s = tl.where((l2n > float("-inf"))[None, :], s + l2n[None, :], float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        if WL:
            kv = tl.load(KV + koff, mask=cv[:, None], other=0.0)
            u = tl.dot(q2, tl.trans(kv)) * half_s2
            w = p * (1.0 + u)
        else:
            w = p
        l_i = l_i * alpha + tl.sum(w, 1)
        g_i = g_i * alpha + tl.sum(p, 1)
        vb = tl.load(VB + koff, mask=cv[:, None], other=0.0)
        acc = tl.dot(w.to(tl.bfloat16), vb, acc * alpha[:, None])
        m_i = m_new
    if PISA:
        ht = tl.load(HT + (h * D + d[:, None]) * D + d[None, :])
        acc += tl.dot(q, ht) * g_i[:, None]
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
    lse_c = tl.where(l_i > 0, m_safe + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    lse_s = tl.load(LSE + h * S + rows)
    mx = tl.maximum(lse_s, lse_c)
    w_s = tl.exp2(lse_s - mx)
    w_c = tl.exp2(lse_c - mx)
    o_s = tl.load(OS + qoff).to(tl.float32)
    o = (o_s * w_s[:, None] + acc * (w_c / tl.maximum(l_i, 1e-30))[:, None]) / (w_s + w_c)[:, None]
    dst = tl.load(PQ + rows)
    tl.store(OUT + dst[:, None] * sout_s + h * D + d[None, :], o.to(OUT.dtype.element_ty))


def merge_fast(qp, os_, lse, st, l2n, qt, pq, out, nt, wl=True, pisa=True, rb=None, bn=None, nw=None, ns=None):
    H, Sp, D = qp.shape
    C = st["kb"].shape[1]
    rb = rb or FAST_CFG["RB"]
    _merge_fast[(Sp // rb, H)](qp, os_, lse, st["kb"], st["vb"], st["kv"], st["ht"] if pisa else qp, l2n, qt, pq, out,
                               Sp, out.stride(0), math.log2(math.e) / math.sqrt(D), 0.5 / D, C, NT_=nt, D=D, RB=rb,
                               BN=bn or FAST_CFG["BN"], WL=wl, PISA=pisa, num_warps=nw or FAST_CFG["W"],
                               num_stages=ns or FAST_CFG["ST"])
