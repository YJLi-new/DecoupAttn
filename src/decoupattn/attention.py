from __future__ import annotations

import math

import torch

from .compensation import FULL

_QT: dict = {}


def _qtile_rows(Sp, device):
    qt = _QT.get((Sp, device))
    if qt is None:
        qt = _QT[(Sp, device)] = (torch.arange(Sp, device=device) // 128).to(torch.int32)
    return qt


def oc_cp_attn(q, k, v, sp, cl, mask_hnn, til, kb, vb, wl=True, pisa=True, lse=False):
    from .kernels import block_sparse as BS
    from .kernels import merge as MG
    H, D = q.shape[2], q.shape[3]
    Sp, C = til.S_pad, cl.C
    plan = sp.get()
    k3, v3 = k[0], v[0]
    b = BS.exact_part_np(q[0], k3, v3, plan, til.S)
    kv = torch.zeros_like(kb)
    if wl:
        MG._cluster_var[(C, H)](k3, kb, kv, cl.k_idx, cl.k_off, k3.stride(0), k3.stride(1), til.S, C, D=D, BR=32,
                                num_warps=4)
    hm = b["qp"]
    if pisa:
        hmat = MG.pisa_h(k3, v3, kb, vb, cl.cnt)
        hm = torch.bmm(b["qp"], (hmat / math.sqrt(D)).transpose(1, 2).to(q.dtype))
    l2n = cl.l2n(mask_hnn)
    out = torch.empty(Sp, H * D, dtype=q.dtype, device=q.device)
    lse_out = torch.empty(H, Sp, dtype=torch.float32, device=q.device) if lse else None
    MG.merge_cp(b["qp"], b["os"], b["lse"], kb, vb, kv, hm, l2n, _qtile_rows(Sp, q.device), plan.q_idx[0].contiguous(),
                out, til.nt, wl, pisa, lse_out)
    if lse:
        return out[:til.S][None], lse_out[:, :til.S]
    return out[:til.S][None]


def decoup_cp_attn(q, k, v, sp, cl, mask_hnn, til, wl=True, pisa=True, lse=False):
    from .kernels.cluster_stats import cluster_means
    kb, vb = cluster_means(k[0], v[0], cl)
    return oc_cp_attn(q, k, v, sp, cl, mask_hnn, til, kb, vb, wl, pisa, lse)


def fast_cp_attn(q, k, v, sp, cl, mask_hnn, til, st, wl=True, pisa=True):
    from .kernels import block_sparse as BS
    from .kernels import merge as MG
    H, D = q.shape[2], q.shape[3]
    Sp = til.S_pad
    plan = sp.get()
    b = BS.exact_part_np(q[0], k[0], v[0], plan, til.S)
    l2n = cl.l2n(mask_hnn)
    out = torch.empty(Sp, H * D, dtype=q.dtype, device=q.device)
    MG.merge_fast(b["qp"], b["os"], b["lse"], st, l2n, _qtile_rows(Sp, q.device), plan.q_idx[0].contiguous(), out,
                  til.nt, wl, pisa)
    return out[:til.S][None]


def decoupled_attention(q, k, v, sp, cl, mask_hnn, til, estimator=FULL, fast=True, lse=False):
    f = estimator == FULL
    if fast and not lse:
        from .kernels.cluster_stats import fast_stats
        st = fast_stats(k[0], v[0], cl.k_idx, cl.k_off, cl.cnt, cl.C, f, f)
        return fast_cp_attn(q, k, v, sp, cl, mask_hnn, til, st, f, f)
    return decoup_cp_attn(q, k, v, sp, cl, mask_hnn, til, f, f, lse)


def spatial_attn(q, k, v, sp, til, compensate=True):
    from .kernels import block_sparse as BS
    return BS.qb_attn_np(q[0], k[0], v[0], sp.get(), til.S, compensate)[:til.S][None]
