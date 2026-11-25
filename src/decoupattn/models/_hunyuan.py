from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import re
import sys
import textwrap
import types

import numpy as np
import torch

from ._wanapi import dense_attn, dense_lse

MERGE_CHUNK = 8192


def dcm_path(repo: str) -> None:
    if repo not in sys.path:
        sys.path.insert(0, repo)

    def stub(n, **attrs):
        m = types.ModuleType(n)
        m.__spec__ = importlib.machinery.ModuleSpec(n, None)
        m.__dict__.update(attrs)
        sys.modules.setdefault(n, m)
    stub("fastvideo.models.flash_attn_no_pad", flash_attn_no_pad=None)
    if importlib.util.find_spec("liger_kernel") is None:
        stub("liger_kernel")
        stub("liger_kernel.ops")
        stub("liger_kernel.ops.swiglu", LigerSiLUMulFunction=None)


def merge_text(o_v, lse_v, q, k, v, nv, backend="fa3"):
    H, D = q.shape[2], q.shape[3]
    lb = "torch" if backend == "torch" else "fa3"
    o_t, lse_t = dense_lse(q[:, :nv], k[:, nv:], v[:, nv:], lb)
    out = torch.empty(1, q.shape[1], H * D, dtype=q.dtype, device=q.device)
    o_v = o_v.view(1, nv, H, D)
    for c0 in range(0, nv, MERGE_CHUNK):
        c1 = min(nv, c0 + MERGE_CHUNK)
        lv, lt = lse_v[:, c0:c1], lse_t[:, c0:c1]
        m = torch.maximum(lv, lt)
        a, b = torch.exp(lv - m), torch.exp(lt - m)
        w = (a / (a + b)).transpose(0, 1)[None, :, :, None]
        ov = o_v[:, c0:c1].float() * w + o_t[:, c0:c1].float() * (1 - w)
        out[:, c0:c1] = ov.to(q.dtype).flatten(2, 3)
    out[:, nv:] = dense_attn(q[:, nv:], k, v, backend).to(q.dtype)
    return out


def attend_joint(adapter, q, k, v, nv):
    o = None
    joint = getattr(adapter.hook, "joint", None)
    if joint is not None:
        o = joint(q, k, v, nv, adapter.ctx)
    elif adapter.hook is not None:
        r = adapter.hook(q[:, :nv], k[:, :nv], v[:, :nv], adapter.ctx)
        if r is not None:
            if not isinstance(r, tuple):
                raise TypeError("a hook on HunyuanVideo must return (o_v, lse_v) for the text merge")
            o = merge_text(r[0], r[1], q, k, v, nv, adapter.backend)
    if o is None:
        o = dense_attn(q, k, v, adapter.backend)
    return o.type_as(q)


class Joint:
    def __init__(self, adapter, nv):
        self.a, self.nv = adapter, nv

    def __call__(self, qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, **_):
        S, H, D = qkv.shape[1], qkv.shape[3], qkv.shape[4]
        n = int(key_padding_mask[0].sum())
        q, k, v = (t.contiguous() for t in qkv[:, :n].unbind(2))
        o = attend_joint(self.a, q, k, v, self.nv)
        out = qkv.new_zeros(1, S, H, D)
        out[:, :n] = o.view(1, n, H, D)
        return out


_MASK_ALLOC = re.compile(r"torch\.zeros\(\s*batch_size\s*,\s*sequence_length\s*,\s*sequence_length")
_MASK_FILL = re.compile(r"attention_mask\[\s*i\s*,\s*:\s*effective_sequence_length\[i\]\s*,\s*"
                        r":\s*effective_sequence_length\[i\]\s*\]")


def compact_mask(MH) -> None:
    cls = MH.HunyuanVideoTransformer3DModel
    if getattr(cls, "_compact_mask", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.forward))
    src, n1 = _MASK_ALLOC.subn("torch.zeros(batch_size, 1, sequence_length", src)
    src, n2 = _MASK_FILL.subn("attention_mask[i, :, : effective_sequence_length[i]]", src)
    if (n1, n2) != (1, 1):
        raise RuntimeError(f"compact_mask: the installed forward differs from the expected source ({n1}, {n2} "
                           "matches); set [sampler] compact_mask = false")
    ns = {}
    exec(compile(src, inspect.getsourcefile(cls.forward), "exec"), vars(MH), ns)
    cls.forward, cls._compact_mask = ns["forward"], True


def to_u8(x) -> np.ndarray:
    x = x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
    return (x * 255).round().clip(0, 255).astype(np.uint8)
