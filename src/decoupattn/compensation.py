from __future__ import annotations

import math

import torch

FULL, CENTROID = "Full", "Centroid"


def head_stats(kh, vh, lb, C):
    S, D = kh.shape
    cnt = torch.bincount(lb, minlength=C).float()
    inv = 1.0 / cnt.clamp(min=1)
    km = torch.zeros(C, D, device=kh.device).index_add_(0, lb, kh) * inv[:, None]
    vm = torch.zeros(C, D, device=kh.device).index_add_(0, lb, vh) * inv[:, None]
    var = torch.zeros(C, D, device=kh.device).index_add_(0, lb, (kh - km[lb]).pow(2)) * inv[:, None]
    hm = (vh.T @ kh - (vm * cnt[:, None]).T @ km) / S
    tv = kh.var(0, unbiased=False).sum().clamp(min=1e-30)
    return cnt, km, vm, var, hm, var.sum(-1) / tv


def centroid_terms(q, km, var=None, hm=None):
    D = q.shape[-1]
    logphi = (q @ km.T) / math.sqrt(D)
    gamma = None if var is None else ((q * q) @ var.T) * (0.5 / D)
    hq = None if hm is None else (q @ hm.T) / math.sqrt(D)
    return logphi, gamma, hq


def decoup_estimate(q, k, v, exact, n, km, vm, var=None, hm=None, return_lse=False):
    D = q.shape[-1]
    s = ((q @ k.T) / math.sqrt(D)).masked_fill(~exact, float("-inf"))
    logphi, gamma, hq = centroid_terms(q, km, var, hm)
    t = (logphi + torch.log(n.clamp(min=1))).masked_fill(n < 0.5, float("-inf"))
    m = torch.maximum(s.amax(-1), t.amax(-1))
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    es = torch.exp(s - m[:, None])
    et = torch.exp(t - m[:, None])
    w = et if gamma is None else et * (1.0 + gamma)
    num = es @ v + w @ vm
    if hq is not None:
        num = num + et.sum(-1, keepdim=True) * hq
    den = es.sum(-1) + w.sum(-1)
    if return_lse:
        return num / den[:, None], m + torch.log(den)
    return num / den[:, None]
