from __future__ import annotations

import math

import torch

from ..compensation import CENTROID, FULL, decoup_estimate, head_stats


@torch.no_grad()
def dense_ref(q, k, v, heads, qc=1024, return_lse=False):
    S, D = q.shape[1], q.shape[-1]
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.empty(S, len(heads), D, device=q.device)
    lse = torch.empty(len(heads), S, device=q.device)
    try:
        for hi, h in enumerate(heads):
            qh, kh, vh = (x[0, :, h].float() for x in (q, k, v))
            for c0 in range(0, S, qc):
                s = (qh[c0:c0 + qc] @ kh.T) / math.sqrt(D)
                out[c0:c0 + qc, hi] = torch.softmax(s, -1) @ vh
                lse[hi, c0:c0 + qc] = torch.logsumexp(s, -1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return (out, lse) if return_lse else out


@torch.no_grad()
def spatial_ref(q, k, v, mask_hnn, til, heads, compensate=True, qc=1024):
    S, D = til.S, q.shape[-1]
    sc = 1.0 / math.sqrt(D)
    t = til.tok_tile.to(q.device)
    nr = til.n_real.float().to(q.device)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.empty(S, len(heads), D, device=q.device)
    try:
        for hi, h in enumerate(heads):
            qh, kh, vh = (x[0, :, h].float() for x in (q, k, v))
            kb = torch.zeros(til.nt, D, device=q.device).index_add_(0, t, kh) / nr[:, None]
            vb = torch.zeros(til.nt, D, device=q.device).index_add_(0, t, vh) / nr[:, None]
            M = mask_hnn[h].to(q.device)
            for c0 in range(0, S, qc):
                sl = slice(c0, min(S, c0 + qc))
                s = (qh[sl] @ kh.T) * sc
                s = s.masked_fill(~M[t[sl]][:, t], float("-inf"))
                if compensate:
                    s2 = (qh[sl] @ kb.T) * sc + torch.log(nr)[None]
                    s2 = s2.masked_fill(M[t[sl]], float("-inf"))
                    out[sl, hi] = torch.softmax(torch.cat([s, s2], 1), -1) @ torch.cat([vh, vb])
                else:
                    out[sl, hi] = torch.softmax(s, -1) @ vh
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


@torch.no_grad()
def decoup_ref(q, k, v, mask_hnn, lab_hs, C, til, heads, qc=1024, estimator=CENTROID, return_lse=False):
    S, D = til.S, q.shape[-1]
    t = til.tok_tile.to(q.device)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.empty(S, len(heads), D, device=q.device)
    lse = torch.empty(len(heads), S, device=q.device)
    try:
        for hi, h in enumerate(heads):
            qh, kh, vh = (x[0, :, h].float() for x in (q, k, v))
            lb = lab_hs[h].long().to(q.device)
            cnt, km, vm, var, hm, _ = head_stats(kh, vh, lb, C)
            pc = torch.zeros(til.nt, C, device=q.device).index_put_((t, lb), torch.ones(S, device=q.device),
                                                                    accumulate=True)
            M = mask_hnn[h].to(q.device)
            n = cnt[None] - M.float() @ pc
            full = estimator == FULL
            for c0 in range(0, S, qc):
                sl = slice(c0, min(S, c0 + qc))
                out[sl, hi], lse[hi, sl] = decoup_estimate(qh[sl], kh, vh, M[t[sl]][:, t], n[t[sl]], km, vm,
                                                           var if full else None, hm if full else None, True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return (out, lse) if return_lse else out
