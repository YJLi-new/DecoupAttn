from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from . import _hunyuan as HJ
from ._wanapi import SEED, Stop, VideoAdapter


class HunyuanVideo(VideoAdapter):
    name = "hy"
    steps = 50
    layout = "rows"
    rows = (1, 4, 11, 22, 37)
    branches = ("cond",)
    n_layers = 60
    n_heads = 24
    head_dim = 128
    latent_grid = (33, 45, 80)
    block_shape = (1, 8, 16)
    needs_lse = True
    n_double = 20

    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["sampler"]
        self.h, self.w, self.frames = s["height"], s["width"], s["frames"]
        self.guidance = s["embedded_guidance"]
        self.pipe = self.tr = None
        self.st = {"call": 0, "stop": None}

    def _modules(self):
        HJ.dcm_path(self.cfg["model"]["repo"])
        import fastvideo.models.hunyuan_hf.modeling_hunyuan as MH
        from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
        return MH, HunyuanVideoPipeline

    def load(self, device: str = "cuda") -> None:
        from safetensors.torch import load_file
        MH, Pipe = self._modules()
        if self.cfg["sampler"].get("compact_mask", True):
            HJ.compact_mask(MH)
        td = self.path("transformer")
        with open(os.path.join(td, "config.json")) as f:
            tcfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        tr = MH.HunyuanVideoTransformer3DModel(**tcfg).to(torch.bfloat16)
        sd = {}
        for p in sorted(glob.glob(os.path.join(td, "*.safetensors"))):
            sd.update(load_file(p))
        tr.load_state_dict(sd, strict=True)
        del sd
        pipe = Pipe.from_pretrained(self.path("dcm_hy"), transformer=tr, text_encoder=None, text_encoder_2=None,
                                    tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
        pipe.enable_vae_tiling()
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.tr = pipe, tr
        self._install(tr, MH)
        self.loaded = True

    def _install(self, tr, MH):
        blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
        for i, b in enumerate(blocks):
            b.register_forward_pre_hook(lambda _m, _a, i=i: setattr(self.ctx, "layer", i))
        MH.flash_attn_no_pad = HJ.Joint(self, int(np.prod(self.latent_grid)))
        st, fwd = self.st, tr.forward

        def forward(*a, **kw):
            if st["stop"] is not None and st["call"] >= st["stop"]:
                raise Stop()
            self.ctx.at(step=st["call"], branch="cond")
            st["call"] += 1
            with self.timed(f"s{self.ctx.step}_cond"):
                return fwd(*a, **kw)
        tr.forward = forward

    def text(self, clip: str) -> dict:
        p = self.cache_file("text", clip)
        if not os.path.exists(p):
            self.encode_texts([clip])
        return torch.load(p, map_location="cuda", weights_only=False)

    @torch.no_grad()
    def encode_texts(self, clips: list[str]) -> None:
        todo = [c for c in clips if not os.path.exists(self.cache_file("text", c))]
        if not todo:
            return
        _, Pipe = self._modules()
        pipe = Pipe.from_pretrained(self.path("dcm_hy"), transformer=None, vae=None, torch_dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        pipe.text_encoder_2.to("cuda")
        for c in todo:
            p = self.prompt(c)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e, pooled, mask = pipe.encode_prompt(prompt=[p], prompt_2=[p], device=torch.device("cuda"))
            r = {"prompt_embeds": e.detach().cpu(), "pooled_prompt_embeds": pooled.detach().cpu(),
                 "prompt_attention_mask": mask.detach().cpu(), "prompt": p}
            f = self.cache_file("text", c)
            torch.save(r, f + ".tmp")
            os.replace(f + ".tmp", f)
        del pipe
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.ensure_loaded()
        st = self.st
        st["call"] = 0
        st["stop"] = steps if (steps is not None and steps < self.steps) else None
        te = self.text(clip)
        try:
            with self.hooked(hook), torch.autocast("cuda", dtype=torch.bfloat16):
                lat = self.pipe(prompt_embeds=te["prompt_embeds"], pooled_prompt_embeds=te["pooled_prompt_embeds"],
                                prompt_attention_mask=te["prompt_attention_mask"], height=self.h, width=self.w,
                                num_frames=self.frames, num_inference_steps=self.steps, guidance_scale=self.guidance,
                                generator=torch.Generator("cpu").manual_seed(seed), output_type="latent").frames
        except Stop:
            return None
        finally:
            st["stop"] = None
        return lat

    @torch.no_grad()
    def decode(self, latent):
        self.ensure_loaded()
        vae = self.pipe.vae
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = latent.to(vae.dtype) / vae.config.scaling_factor
            video = vae.decode(z, return_dict=False)[0]
            video = self.pipe.video_processor.postprocess_video(video, output_type="np")
        return HJ.to_u8(video[0])

    def out_proj(self, layer, step=0):
        if layer < self.n_double:
            m = self.tr.transformer_blocks[layer].attn.to_out[0]
            return getattr(m, "base_layer", m).weight
        m = self.tr.single_transformer_blocks[layer - self.n_double].proj_out
        return getattr(m, "base_layer", m).weight[:, :self.n_heads * self.head_dim]
