from __future__ import annotations

import json
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from ..config import repo_path
from ..constants import SEED
from .base import Hook, ModelAdapter

SPLITS = {"calib": "calib", "val": "val", "test": "test"}

_FA3_NONE = ("k_new", "v_new", "qv", "out", "cu_seqlens_q", "cu_seqlens_k", "cu_seqlens_k_new", "seqused_q", "seqused_k",
             "max_seqlen_q", "max_seqlen_k", "page_table", "kv_batch_idx", "leftpad_k", "rotary_cos", "rotary_sin",
             "seqlens_rotary", "q_descale", "k_descale", "v_descale")


def _expand_kv(q, k, v):
    rep = q.shape[2] // k.shape[2]
    if rep == 1:
        return k, v
    return k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)


def dense_attn(q, k, v, backend: str = "fa3", causal: bool = False) -> torch.Tensor:
    if backend == "fa3":
        from flash_attn_interface import flash_attn_func
        o = flash_attn_func(q, k, v, causal=causal)
        return (o[0] if isinstance(o, tuple) else o).flatten(2, 3)
    if backend == "fa2":
        import flash_attn
        return flash_attn.flash_attn_func(q, k, v, causal=causal).flatten(2, 3)
    if backend == "fi":
        import flashinfer
        o = flashinfer.single_prefill_with_kv_cache(q[0], k[0], v[0], causal=causal, backend="fa3")
        return o[None].flatten(2, 3)
    if backend == "torch":
        o, _ = _torch_attn(q, k, v, causal)
        return o.to(q.dtype).flatten(2, 3)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if backend == "cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ke, ve = _expand_kv(q, k, v)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = F.scaled_dot_product_attention(qt, ke.transpose(1, 2), ve.transpose(1, 2), is_causal=causal)
    elif backend == "sdpa":
        if k.shape[2] != q.shape[2]:
            o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    else:
        raise ValueError(f"unknown dense backend {backend}")
    return o.transpose(1, 2).flatten(2, 3)


def _torch_attn(q, k, v, causal=False):
    k, v = _expand_kv(q, k, v)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * q.shape[-1] ** -0.5
    if causal:
        s = s.masked_fill(torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v.float())
    return o, lse[0]


def dense_lse(q, k, v, backend: str = "fa3"):
    if backend == "fa3":
        import flash_attn_interface as FI
        out, lse, *_ = FI._flash_attn_forward(q, k, v, softmax_scale=q.shape[-1] ** -0.5, causal=False,
                                              **dict.fromkeys(_FA3_NONE))
        return out, lse[0].float()
    if backend == "torch":
        o, lse = _torch_attn(q, k, v)
        return o.to(q.dtype), lse
    raise ValueError(f"dense_lse has no backend {backend}")


class CallTimer:
    def __init__(self):
        self.rec = []

    @contextmanager
    def __call__(self, tag):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record()
        self.rec.append((tag, a, b))

    def result(self) -> list[tuple[str, float]]:
        torch.cuda.synchronize()
        return [(t, a.elapsed_time(b)) for t, a, b in self.rec]

    def span_ms(self) -> float:
        torch.cuda.synchronize()
        return self.rec[0][1].elapsed_time(self.rec[-1][2])


class Stop(Exception):
    pass


def load_split(path: str) -> dict:
    with open(repo_path(path)) as f:
        return json.load(f)


def read_prompts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    if "splits" in man:
        return {p["id"]: p["prompt"] for items in man["splits"].values() for p in items}
    return dict(man)


def to_uint8(video: torch.Tensor) -> np.ndarray:
    return ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


class VideoAdapter(ModelAdapter):
    family = "video"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.hook: Hook | None = None
        self.backend = cfg["model"]["dense_backend"]
        self.loaded = False
        self._split = None
        self._prompts = None

    def split_data(self) -> dict:
        if self._split is None:
            self._split = load_split(self.cfg["data"]["splits"])
        return self._split

    def clips(self, split: str) -> list[str]:
        return list(self.split_data()[SPLITS[split]])

    def prompt(self, clip: str) -> str:
        if self._prompts is None:
            self._prompts = read_prompts(self.cfg["data"]["clips"])
        return self._prompts[clip]

    def dense(self, q, k, v):
        return dense_attn(q, k, v, self.backend)

    def dense_lse(self, q, k, v):
        o, lse = dense_lse(q, k, v, "torch" if self.backend == "torch" else "fa3")
        return o.flatten(2, 3), lse

    def attend(self, q, k, v):
        o = self.hook(q, k, v, self.ctx) if self.hook is not None else None
        return self.dense(q, k, v) if o is None else o

    def timed(self, tag: str):
        return self.dit_timer(tag)

    @contextmanager
    def hooked(self, hook: Hook | None):
        self.hook = hook
        try:
            yield
        finally:
            self.hook = None

    def ensure_loaded(self):
        if not self.loaded:
            self.load()

    def path(self, key: str) -> str:
        return self.cfg["checkpoints"][key]

    def cache_file(self, sub: str, clip: str, ext: str = ".pt") -> str:
        d = os.path.join(self.cfg["data"]["work"], sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, clip + ext)


__all__ = ["SEED", "CallTimer", "Stop", "VideoAdapter", "dense_attn", "dense_lse", "load_split", "read_prompts",
           "to_uint8"]
