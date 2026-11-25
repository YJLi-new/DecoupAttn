from __future__ import annotations

import math

import torch

from ._triton import jit, tl


@jit
def _block_mass_kernel(Q, K, VALID, BM, S, NT, qk_scale_log2, D: tl.constexpr, BL: tl.constexpr):
    i = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.arange(0, BL)
    d = tl.arange(0, D)
    qrow = i * BL + r
    q = tl.load(Q + (h * S + qrow[:, None]).to(tl.int64) * D + d[None, :])
    qv = tl.load(VALID + qrow).to(tl.float32)
    m = tl.full([BL], float("-inf"), tl.float32)
    l = tl.zeros([BL], tl.float32)
    for j in range(NT):
        krow = j * BL + r
        k = tl.load(K + (h * S + krow[:, None]).to(tl.int64) * D + d[None, :])
        kv = tl.load(VALID + krow)
        s = tl.dot(q, tl.trans(k)) * qk_scale_log2
        s = tl.where(kv[None, :] != 0, s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        l = l * tl.exp2(m - mn) + tl.sum(tl.exp2(s - mn[:, None]), 1)
        m = mn
    lse = m + tl.log2(l)
    nq = tl.maximum(tl.sum(qv, 0), 1.0)
    for j in range(NT):
        krow = j * BL + r
        k = tl.load(K + (h * S + krow[:, None]).to(tl.int64) * D + d[None, :])
        kv = tl.load(VALID + krow)
        s = tl.dot(q, tl.trans(k)) * qk_scale_log2
        p = tl.where(kv[None, :] != 0, tl.exp2(s - lse[:, None]), 0.0)
        tot = tl.sum(tl.sum(p, 1) * qv, 0) / nq
        tl.store(BM + (h * NT + i) * NT + j, tot)


@torch.no_grad()
def block_mass(q, k, til):
    H, D = q.shape[2], q.shape[3]
    idx = til.slot_src.clamp(min=0)
    qt = q[0].index_select(0, idx).mul_(til.real[:, None, None]).transpose(0, 1).contiguous()
    kt = k[0].index_select(0, idx).mul_(til.real[:, None, None]).transpose(0, 1).contiguous()
    bm = torch.empty(H, til.nt, til.nt, dtype=torch.float32, device=q.device)
    _block_mass_kernel[(til.nt, H)](qt, kt, til.real.to(torch.int8), bm, til.S_pad, til.nt,
                                    math.log2(math.e) / math.sqrt(D), D=D, BL=128, num_warps=8, num_stages=2)
    return bm


@torch.no_grad()
def block_mass_ref(q, k, til, heads, chunk=2048):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    out = []
    try:
        for h in heads:
            qh, kh = q[0, :, h].float(), k[0, :, h].float()
            acc = torch.zeros(til.nt, til.nt, device=q.device, dtype=torch.float64)
            for c0 in range(0, til.S, chunk):
                p = torch.softmax(qh[c0:c0 + chunk] @ kh.T / math.sqrt(qh.shape[1]), -1)
                pt = torch.zeros(p.shape[0], til.nt, device=q.device).index_add_(1, til.tok_tile, p)
                acc.index_add_(0, til.tok_tile[c0:c0 + chunk], pt.double())
            out.append((acc / til.n_real[:, None].double()).float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return torch.stack(out)


def build_calib(bm_mean, n_real, density):
    L, H, NT, _ = bm_mean.shape
    nr = n_real.double().to(bm_mean.device)
    w = (nr[:, None] * nr[None, :])
    S = float(nr.sum())
    masks = torch.zeros(L, H, NT, NT, dtype=torch.bool)
    dens = []
    eye = torch.eye(NT, dtype=torch.bool, device=bm_mean.device)
    for l in range(L):
        b = bm_mean[l].double().clone()
        m = eye[None].expand(H, NT, NT).clone()
        b[m] = float("inf")
        order = torch.argsort(b.flatten(), descending=True)
        cw = torch.cumsum(w.expand(H, NT, NT).flatten()[order], 0)
        n = int(torch.searchsorted(cw, density * H * S * S)) + 1
        sel = torch.zeros(H * NT * NT, dtype=torch.bool, device=b.device)
        sel[order[:max(n, int(m.sum()))]] = True
        sel = sel.view(H, NT, NT) | m
        masks[l] = sel.cpu()
        dens.append(float((sel.double() * w).sum() / (H * S * S)))
    return masks, dens
