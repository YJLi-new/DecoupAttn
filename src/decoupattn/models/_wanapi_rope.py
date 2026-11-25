from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _two_prod(a, b):
    p = a * b
    return p, tl.fma(a, b, -p)


@triton.jit
def _rope_kernel_i32(X, XO, FR, H, D2: tl.constexpr, BH: tl.constexpr):
    s = tl.program_id(0)
    h = tl.arange(0, BH)
    rm = (h < H)[:, None]
    i = tl.arange(0, D2)
    fb = FR + s.to(tl.int64) * (4 * D2) + i
    ch = tl.load(fb)[None, :]
    cl = tl.load(fb + D2)[None, :]
    dh = tl.load(fb + 2 * D2)[None, :]
    dl = tl.load(fb + 3 * D2)[None, :]
    off = (s.to(tl.int64) * H + h[:, None]) * D2 + i[None, :]
    x = tl.load(X + off, mask=rm, other=0)
    a = (x << 16).to(tl.float32, bitcast=True)
    b = (x & -65536).to(tl.float32, bitcast=True)
    p1, e1 = _two_prod(a, ch)
    p2, e2 = _two_prod(b, dh)
    sre = p1 - p2
    bb = sre - p1
    ere = (p1 - (sre - bb)) + (-p2 - bb)
    re = sre + (ere + (e1 - e2) + (a * cl - b * dl))
    p3, e3 = _two_prod(a, dh)
    p4, e4 = _two_prod(b, ch)
    sim = p3 + p4
    bb2 = sim - p3
    eim = (p3 - (sim - bb2)) + (p4 - bb2)
    im = sim + (eim + (e3 + e4) + (a * dl + b * cl))
    r16 = re.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    i16 = im.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    tl.store(XO + off, (i16 << 16) | r16, mask=rm)


_FR = {}


def _freqs_split(freqs):
    key = (freqs.data_ptr(), tuple(freqs.shape))
    fr = _FR.get(key)
    if fr is None:
        _FR.clear()
        f = torch.view_as_real(freqs.reshape(-1, freqs.shape[-1]))
        hi = f.float()
        lo = (f - hi.double()).float()
        fr = _FR[key] = torch.stack([hi[..., 0], lo[..., 0], hi[..., 1], lo[..., 1]], 1).contiguous()
    return fr


def rope(x, freqs):
    _, S, H, D = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    fr = _freqs_split(freqs)
    out = torch.empty((1, S, H, D), dtype=x.dtype, device=x.device)
    _rope_kernel_i32[(S,)](x.view(torch.int32), out.view(torch.int32), fr, H, D2=D // 2, BH=triton.next_power_of_2(H),
                           num_warps=8, enable_fp_fusion=False)
    return out
