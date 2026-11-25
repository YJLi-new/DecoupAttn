from __future__ import annotations

import json
import os
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..config import repo_path
from ..context import CallContext
from .base import Hook, ModelAdapter

SPLITS = ("calib", "val", "test")


class Stop(Exception):
    pass


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str,
                    native: Callable | None = None) -> torch.Tensor:
    if backend == "fa3":
        from flash_attn_interface import flash_attn_func
        o = flash_attn_func(q, k, v, causal=False)
        return (o[0] if isinstance(o, tuple) else o).flatten(2, 3)
    if backend == "fa2":
        import flash_attn
        return flash_attn.flash_attn_func(q, k, v).flatten(2, 3)
    if backend == "fi":
        import flashinfer
        return flashinfer.single_prefill_with_kv_cache(q[0], k[0], v[0], causal=False, kv_layout="NHD",
                                                       backend="fa3")[None].flatten(2, 3)
    if backend == "cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).flatten(2, 3)
    if backend == "native":
        return native(q, k, v)
    if backend == "sdpa":
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).flatten(2, 3)
    raise ValueError(backend)


def nearest_frames(n_src: int, n_out: int) -> list[int]:
    return [round(i * (n_src - 1) / (n_out - 1)) for i in range(n_out)]


def to_uint8(video: torch.Tensor) -> np.ndarray:
    return ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


def caption(clip_dir: str) -> str:
    with open(os.path.join(clip_dir, "render.json")) as f:
        s = json.load(f)["spec"]
    objs = [o["model"].replace("_", " ") for o in s["objects"]]
    ground = str(s.get("ground", {}).get("texture", s.get("ground", {}).get("material", "plain"))).replace("_", " ")
    hdri = s["hdri"]["name"].replace("_2k.hdr", "").replace(".hdr", "").replace("_", " ")
    return (f"A photorealistic 3D rendered scene with {', '.join(objs)} standing on a {ground} ground, the camera moving "
            f"slowly around them; the scene is lit by natural image-based lighting of a {hdri} environment, with soft "
            f"realistic shadows and reflections.")


def read_pngs(paths: list[str]) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths]).astype(np.uint8)


def link(dst: str, src: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst):
        os.symlink(src, dst)


class RenderAdapter(ModelAdapter):
    step_clamp: int | None = None

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.hook: Hook | None = None
        self.stop_at: int | None = None
        self.loaded = False
        self.native: Callable | None = None
        self._splits = None

    @property
    def backend(self) -> str:
        return self.cfg["model"]["dense_backend"]

    def path(self, *keys: str) -> str:
        d = self.cfg
        for k in keys:
            d = d[k]
        return repo_path(d)

    def clip_dir(self, clip: str) -> str:
        return os.path.join(self.path("data", "clips"), clip)

    def work_dir(self) -> str:
        d = self.path("data", "work")
        os.makedirs(d, exist_ok=True)
        return d

    def split_lists(self) -> dict:
        if self._splits is None:
            with open(self.path("data", "splits")) as f:
                self._splits = json.load(f)
        return self._splits

    def clips(self, split: str) -> list[str]:
        sp = self.split_lists()
        drop = set(sp.get("dropped", []))
        return [c for c in sp[split] if c not in drop]

    def dense(self, q, k, v):
        return dense_attention(q, k, v, self.backend, self.native)

    def route(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        o = None
        if self.hook is not None:
            ctx = self.ctx
            if self.step_clamp is not None and ctx.step > self.step_clamp:
                s = ctx.step
                ctx.step = self.step_clamp
                try:
                    o = self.hook(q, k, v, ctx)
                finally:
                    ctx.step = s
            else:
                o = self.hook(q, k, v, ctx)
        if o is None:
            o = self.dense(q, k, v)
        return o.type_as(q)

    def dit_call(self, tag: str):
        if self.stop_at is not None and self.ctx.step >= self.stop_at:
            raise Stop()
        return self.dit_timer(tag)

    def begin(self, hook: Hook | None, steps: int | None = None) -> None:
        if not self.loaded:
            self.load()
        self.hook = hook
        self.ctx = CallContext()
        self.stop_at = steps if steps is not None and steps < self.steps else None

    def out_proj(self, layer: int, step: int = 0) -> torch.Tensor:
        return self._wo[layer]
