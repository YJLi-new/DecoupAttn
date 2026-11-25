from __future__ import annotations

import json
import os
import types

import numpy as np
import torch
import torch.nn.functional as F

from ._common import BaselineHook, add_path, upstream_path

EPSILON = {
    "wan50": (0.990, 0.842, 0.320),
    "wan50_720p": (0.990, 0.903, 0.320),
    "distilled4": (0.863, 0.763, 1.410),
}


def import_calibatt():
    add_path(upstream_path("ml-calibatt"))


def epsilon_of(settings: dict) -> tuple[float, float, float]:
    e = settings["epsilon"]
    return tuple(EPSILON[e]) if isinstance(e, str) else tuple(float(x) for x in e)


def anchor_row_counts(rows: int, anchors: int) -> list[int]:
    return [rows // anchors + (1 if i < rows % anchors else 0) for i in range(anchors)]


def fa3_with_lse(q, k, v, softmax_scale=None, causal=False, return_attn_probs=False, **kw):
    from flash_attn_interface import _flash_attn_forward, flash_attn_func
    if not return_attn_probs:
        return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal, **kw)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    o, lse, *_ = _flash_attn_forward(q, k, v, None, None, None, None, None, None, None, None, None, None, None, None,
                                     None, None, None, None, None, None, None, None, scale, causal=causal)
    return o, lse


def pad_energy(e: torch.Tensor, nq: int, nk: int) -> torch.Tensor:
    e = F.pad(e, (0, nk - e.shape[-1]))
    return torch.cat([e, e[:, :nq - e.shape[1]]], 1) if e.shape[1] < nq else e


def _processor_module(n_layers: int):
    from lite_attention import LiteAttention
    m = types.SimpleNamespace()
    m.install = lambda backend, pipeline: [LiteAttention(enable_skipping=True, max_batch_size=1) for _ in range(n_layers)]
    m.after_step = lambda backend: None
    return m


class HostSkipLists(dict):
    def __init__(self, d: dict, device):
        super().__init__({k: v.pin_memory() if torch.cuda.is_available() else v for k, v in d.items()})
        self.device = device

    def __getitem__(self, key):
        return dict.__getitem__(self, key).to(self.device, non_blocking=True)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def values(self):
        return (self[k] for k in self.keys())


def make_backend(geom: dict, settings: dict, mode: str, device: str = "cuda", capture_dir: str | None = None,
                 skip_lists: str | None = None, similarity_map: str | None = None, similarity: bool = False):
    import_calibatt()
    from CalibAtt.calibatt import CalibAttBackend
    from lite_attention import LiteAttention
    host = settings.get("skip_lists_device") == "cpu" and mode == "infer"
    mean = settings.get("cfg_energy") == "mean"
    on_cpu = settings.get("energy_device") == "cpu"

    F_, Hp, Wp = geom["grid"]
    fa = list(LiteAttention.get_MN(geom["head_dim"], torch.bfloat16))
    text = int(settings.get("max_text_tokens", 0))
    tiles = tuple(-(-(F_ * Hp * Wp + text) // b) for b in fa) if text else None

    class Backend(CalibAttBackend):
        energy = None

        def _calc_block_energy_from_qk(self, q, k, softmax_scale=None, causal=False):
            e = super()._calc_block_energy_from_qk(q, k, softmax_scale=softmax_scale, causal=causal)
            if self.energy is not None:
                key, x = (self.current_timestep, self.current_layer), e[0]
                x = x.float().cpu() if on_cpu else x
                x = pad_energy(x, *tiles) if tiles else x
                old = self.energy.get(key) if mean else None
                self.energy[key] = (x if old is None else (old.float() + x.float()) / 2).to(torch.float16)
            return e

        def load_skip_lists(self, load_path):
            if not host:
                return super().load_skip_lists(load_path)
            dev, self.device = self.device, torch.device("cpu")
            try:
                super().load_skip_lists(load_path)
            finally:
                self.device = dev
            self.skip_lists_dict = HostSkipLists(self.skip_lists_dict, dev)

    kw = {"pipeline": None, "processor_mod": _processor_module(geom["layers"]), "cfg": geom["cfg"],
          "num_layers": geom["layers"], "num_steps": geom["steps"], "num_heads": geom["heads"], "fa_block_size": fa,
          "num_frames": F_, "height_patches": Hp, "width_patches": Wp, "device": torch.device(device), "verbose": True,
          "repetitive_threshold": settings["repetitive_threshold"], "anchor_rows": settings["anchor_rows"],
          "repeat_anchor_strategy": settings["repeat_anchor_strategy"]}
    if text:
        import CalibAtt.calibatt as CA
        CA.flash_attn_func_fa3 = fa3_with_lse
        kw.update(max_text_tokens=text, text_tokens_position="end")
    if mode == "capture":
        e = epsilon_of(settings)
        kw.update(calibrate_masks=True, calibrate_similarity=similarity,
                  calibrate_similarity_num_samples=1 if similarity else None, epsilon_start=e[0], epsilon_floor=e[1],
                  epsilon_decay=e[2], output_dir=capture_dir)
        if settings.get("cache_device"):
            kw["cache_device"] = settings["cache_device"]
    else:
        kw.update(skip_lists_path=skip_lists, similarity_map_path=similarity_map)
    return Backend(**kw)


def geometry(adapter) -> dict:
    return {"grid": tuple(adapter.latent_grid), "layers": adapter.n_layers, "heads": adapter.n_heads,
            "head_dim": adapter.head_dim, "steps": adapter.steps, "cfg": len(adapter.branches) > 1}


class CalibAttHook(BaselineHook):
    name = "calibatt"

    def __init__(self, geom: dict, settings: dict, mode: str = "infer", lse: bool = False, dense_lse=None, **paths):
        n = 1 if settings["dense_first"] else 0
        super().__init__(n, n, lse)
        self.geom, self.settings, self.mode, self.dense_lse = geom, settings, mode, dense_lse
        self.text = int(settings.get("max_text_tokens", 0))
        self.be = make_backend(geom, settings, mode, **paths)
        F_, Hp, Wp = geom["grid"]
        self.S, self.rows = F_ * Hp * Wp, Hp
        self.fa = [self.be.fa_block_size_m, self.be.fa_block_size_n]
        self.dens: dict = {}
        if mode == "infer":
            self._densities()
        self.reset()

    def _densities(self) -> None:
        la = self.be.lite_attention_modules[min(1, len(self.be.lite_attention_modules) - 1)]
        if la._ktiles is None:
            la.initialize_ktiles(seq_len=self.S, head_dim=self.geom["head_dim"], dtype=torch.bfloat16)
        for key, t in self.be.skip_lists_dict.items():
            frac = la.calc_percentage_per_head(t).float().mean(dim=(0, 2))
            rep, _ = self.be.repetitive_heads_per_layer.get(key, ([], None))
            if rep:
                frac[rep] = self.be.anchor_rows / self.rows
            self.dens[key] = float(frac.mean())

    def reset(self) -> None:
        super().reset()
        if self.mode == "capture":
            self.be.reset_logs()
            self.be.energy = {}

    def __call__(self, q, k, v, ctx):
        if self.text:
            raise RuntimeError("CalibAtt on an MMDiT is calibrated on the joint sequence: the adapter must call hook.joint")
        if self.is_dense(ctx):
            return None
        be = self.be
        be.current_timestep, be.current_layer = ctx.step, ctx.layer
        B, S, H, D = q.shape
        if self.mode == "capture":
            out = be.compute_attention(q, k, v)
            self.record(H * S * S, H * S * S, H * S * S)
            return None if self.lse else out.reshape(B, S, H * D)
        d = self.dens[(ctx.step, ctx.layer)] * H * S * S
        self.record(d, d, H * S * S)
        if self.lse:
            out, lse = self._infer_lse(q, k, v)
            return out.reshape(B, S, H * D), lse
        return be.compute_attention(q, k, v).reshape(B, S, H * D)

    def joint(self, q, k, v, nv, ctx):
        if self.is_dense(ctx):
            return None
        be = self.be
        be.current_timestep, be.current_layer = ctx.step, ctx.layer
        B, S, H, D = q.shape
        out = be.compute_attention(q, k, v)
        d = (self.dens[(ctx.step, ctx.layer)] if self.mode == "infer" else 1.0) * H * nv * nv
        self.record(d, d, H * nv * nv)
        return out.reshape(B, S, H * D)

    def _infer_lse(self, query, key, value):
        be = self.be
        t, layer = be.current_timestep, be.current_layer
        lite = be.lite_attention_modules[layer]
        rep, _ = be.repetitive_heads_per_layer.get((t, layer), ([], None))
        lite._skip_list[0] = be.skip_lists_dict.get((t, layer))
        lite._use_external_skip_list = True
        lite._ktiles = (key.shape[1] + be.fa_block_size_n - 1) // be.fa_block_size_n
        out, lse = lite(query, key, value, return_softmax_lse=True, timestep=t)
        lse = lse[0].float()
        if rep:
            if be.repeat_anchor_strategy != "average":
                raise ValueError("the LSE path implements the anchor strategy of the paper's runs, 'average'")
            B, S, _, D = query.shape
            F_, Hp, Wp = self.geom["grid"]
            cnt = anchor_row_counts(Hp, be.anchor_rows)
            idx = torch.repeat_interleave(torch.arange(len(cnt), device=query.device),
                                          torch.tensor(cnt, device=query.device))
            qv = query[:, :, rep].view(B, F_, Hp, Wp, len(rep), D)
            qa = torch.stack([g.mean(dim=2) for g in torch.split(qv, cnt, dim=2)], dim=2)
            oa, la = self.dense_lse(qa.reshape(B, F_ * len(cnt) * Wp, len(rep), D), key[:, :, rep], value[:, :, rep])
            oa = oa.view(B, F_, len(cnt), Wp, len(rep), D).index_select(2, idx).reshape(B, S, len(rep), D)
            la = la.view(len(rep), F_, len(cnt), Wp).index_select(2, idx).reshape(len(rep), S)
            out[:, :, rep] = oa.to(out.dtype)
            lse[rep] = la.float()
        return out, lse

    def end_clip(self, d: str, first: bool) -> None:
        os.makedirs(d, exist_ok=True)
        if first:
            self.be.update_output_dir(d)
            self.be.save_deferred_masks()
        keys = sorted(self.be.energy)
        ts, ls = sorted({x[0] for x in keys}), sorted({x[1] for x in keys})
        E = np.empty((len(ts), len(ls)) + tuple(self.be.energy[keys[0]].shape), np.float16)
        for i, t in enumerate(ts):
            for j, layer in enumerate(ls):
                E[i, j] = self.be.energy[(t, layer)].cpu().numpy()
        np.save(os.path.join(d, "block_energy.tmp.npy"), E)
        os.replace(os.path.join(d, "block_energy.tmp.npy"), os.path.join(d, "block_energy.npy"))
        with open(os.path.join(d, "block_energy_index.json"), "w") as f:
            json.dump({"t": ts, "l": ls, "shape": list(E.shape), "fa_block": self.fa, "grid": list(self.geom["grid"]),
                       "S": self.S, "eps": list(epsilon_of(self.settings))}, f)
        self.be.energy = {}

    def summary(self) -> dict:
        return {**super().summary(), "mode": self.mode, "dense_first": bool(self.ds)}


def _lse_fn(adapter):
    if not adapter.needs_lse:
        return None
    from ..models._wanapi import dense_lse
    backend = "torch" if getattr(adapter, "backend", "fa3") == "torch" else "fa3"
    return lambda q, k, v: dense_lse(q, k, v, backend)


def infer_hook(adapter, settings: dict, agg_dir: str, device: str = "cuda") -> CalibAttHook:
    return CalibAttHook(geometry(adapter), settings, "infer", lse=adapter.needs_lse, dense_lse=_lse_fn(adapter),
                        device=device, skip_lists=os.path.join(agg_dir, "skip_lists.pt"),
                        similarity_map=os.path.join(agg_dir, "similarity_map.pt"))


def capture(adapter, settings: dict, cap: str, clips: list[str], seed: int, device: str = "cuda", log=print) -> None:
    os.makedirs(cap, exist_ok=True)
    stem = lambda c: os.path.join(cap, f"{c}_s{seed}")
    todo = [c for c in clips if not os.path.exists(os.path.join(stem(c), "block_energy.npy"))]
    if not todo:
        return
    first_sim = not any(os.path.exists(os.path.join(stem(c), "spatial_similarities.pt")) for c in clips)
    hook = CalibAttHook(geometry(adapter), settings, "capture", lse=adapter.needs_lse, device=device, capture_dir=cap,
                        similarity=first_sim)
    for i, c in enumerate(todo):
        hook.reset()
        adapter.generate(c, hook, seed)
        hook.end_clip(stem(c), first=(i == 0 and first_sim))
        log(f"calibatt capture {c} | {i + 1}/{len(todo)}")
