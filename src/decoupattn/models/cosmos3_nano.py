from __future__ import annotations

import json
import os

import numpy as np
import torch

from ._wanapi import SEED, Stop, VideoAdapter, dense_attn, to_uint8

CFG = 2


def merge_lse(o1, l1, o2, l2):
    H = l1.shape[0]
    m = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - m), torch.exp(l2 - m)
    z = w1 + w2
    a1 = (w1 / z).transpose(0, 1)[None, :, :, None]
    a2 = (w2 / z).transpose(0, 1)[None, :, :, None]
    o = o1.float().unflatten(2, (H, -1)) * a1 + o2.float().unflatten(2, (H, -1)) * a2
    return o.flatten(2, 3)


def _rot(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class C3Attn:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, adapter, layer):
        self.a, self.layer = adapter, layer

    def __call__(self, attn, und_seq, gen_seq, rotary_emb):
        a = self.a
        Hq, Hk, D = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
        q_und = attn.norm_q(attn.to_q(und_seq).view(-1, Hq, D))
        k_und = attn.norm_k(attn.to_k(und_seq).view(-1, Hk, D))
        v_und = attn.to_v(und_seq).view(-1, Hk, D)
        q_gen = attn.norm_added_q(attn.add_q_proj(gen_seq).view(-1, Hq, D))
        k_gen = attn.norm_added_k(attn.add_k_proj(gen_seq).view(-1, Hk, D))
        v_gen = attn.add_v_proj(gen_seq).view(-1, Hk, D)
        k_ug = attn.k_norm_und_for_gen(k_und) if attn.k_norm_und_for_gen is not None else k_und
        cu, su, cg, sg = rotary_emb
        cu, su, cg, sg = cu.unsqueeze(1), su.unsqueeze(1), cg.unsqueeze(1), sg.unsqueeze(1)
        q_und = q_und * cu + _rot(q_und) * su
        k_und = k_und * cu + _rot(k_und) * su
        k_ug = k_ug * cu + _rot(k_ug) * su
        q_gen = q_gen * cg + _rot(q_gen) * sg
        k_gen = k_gen * cg + _rot(k_gen) * sg
        causal_out = dense_attn(q_und[None], k_und[None], v_und[None], a.backend, causal=True)[0]
        a.ctx.layer = self.layer
        r = None
        if a.hook is not None:
            rep = Hq // Hk
            a.ctx.text = None
            if getattr(a.hook, "wants_text", False):
                a.ctx.text = (k_ug.repeat_interleave(rep, 1)[None], v_und.repeat_interleave(rep, 1)[None])
            r = a.hook(q_gen[None], k_gen.repeat_interleave(rep, 1)[None], v_gen.repeat_interleave(rep, 1)[None], a.ctx)
        if r is None:
            full = dense_attn(q_gen[None], torch.cat([k_ug, k_gen])[None], torch.cat([v_und, v_gen])[None], a.backend)
        else:
            if not isinstance(r, tuple):
                raise TypeError("a hook on Cosmos3-Nano must return (o, lse) for the text-key merge")
            o_s, l_s = r
            if l_s is None:
                full = o_s
            else:
                o_t, l_t = a.dense_lse(q_gen[None], k_ug[None], v_und[None])
                full = merge_lse(o_s, l_s, o_t, l_t)
        full = full[0].to(gen_seq.dtype)
        return attn.to_out(causal_out.to(und_seq.dtype)), attn.to_add_out(full)


class Cosmos3Nano(VideoAdapter):
    name = "nano"
    steps = 35
    layout = "rows"
    rows = (1, 2, 4, 7, 11, 15, 20, 26, 32)
    branches = ("cond", "uncond")
    n_layers = 36
    n_heads = 32
    n_kv_heads = 8
    head_dim = 128
    latent_grid = (48, 23, 40)
    block_shape = None
    needs_lse = True

    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["sampler"]
        self.h, self.w, self.frames, self.fps = s["height"], s["width"], s["frames"], s["fps"]
        self.guidance, self.flow_shift = s["guidance"], s["flow_shift"]
        self.pipe = self.tr = self.neg = None
        self.run = {"calls": 0, "max_steps": None}

    def load(self, device: str = "cuda") -> None:
        from diffusers import AutoencoderKLWan, Cosmos3OmniPipeline, Cosmos3OmniTransformer
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import AutoTokenizer
        root = self.path("model")
        tr = Cosmos3OmniTransformer.from_pretrained(root, subfolder="transformer", torch_dtype=torch.bfloat16)
        tr = tr.to(device).eval().requires_grad_(False)
        vae = AutoencoderKLWan.from_pretrained(root, subfolder="vae", torch_dtype=torch.bfloat16).to(device).eval()
        tok = AutoTokenizer.from_pretrained(root, subfolder="text_tokenizer")
        sch = UniPCMultistepScheduler.from_pretrained(root, subfolder="scheduler")
        sch = UniPCMultistepScheduler.from_config(sch.config, flow_shift=self.flow_shift)
        pipe = Cosmos3OmniPipeline(transformer=tr, text_tokenizer=tok, vae=vae, scheduler=sch, sound_tokenizer=None,
                                   enable_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        for i, layer in enumerate(tr.layers):
            layer.self_attn.set_processor(C3Attn(self, i))
        with open(os.path.join(root, "assets", "negative_prompt.json")) as f:
            self.neg = json.dumps(json.load(f))
        self.pipe, self.tr = pipe, tr
        self._wrap_forward()
        self.loaded = True

    def _wrap_forward(self):
        run, orig = self.run, self.tr.forward

        def fwd(*args, **kw):
            c = run["calls"]
            i, br = c // CFG, ("cond" if c % CFG == 0 else "uncond")
            if run["max_steps"] is not None and i >= run["max_steps"]:
                raise Stop()
            self.ctx.at(step=i, branch=br)
            self.ctx.sigma = float(self.pipe.scheduler.sigmas[i])
            run["calls"] = c + 1
            with self.timed(f"s{i}_{br}"):
                return orig(*args, **kw)
        self.tr.forward = fwd

    def cond_path(self, clip: str) -> str:
        return self.cache_file("cond", clip, ".png")

    def condition_image(self, clip: str):
        from PIL import Image
        p = self.cond_path(clip)
        if not os.path.exists(p):
            self.make_condition_images([clip])
        return Image.open(p).convert("RGB")

    def make_condition_images(self, clips: list[str]) -> None:
        from PIL import Image
        s = self.cfg["sampler"]
        for c in clips:
            p = self.cond_path(c)
            if os.path.exists(p):
                continue
            lat = self._sample(None, self.prompt(c), SEED, s["cond_height"], s["cond_width"], None)
            Image.fromarray(self.decode(lat)[0]).save(p + ".tmp.png")
            os.replace(p + ".tmp.png", p)

    def conditions(self, clip):
        return {"image": np.asarray(self.condition_image(clip))[None]}

    @torch.no_grad()
    def _sample(self, image, prompt, seed, h, w, steps):
        self.ensure_loaded()
        self.run.update(calls=0, max_steps=steps if steps and steps < self.steps else None)
        g = torch.Generator(device="cuda").manual_seed(seed)
        kw = {} if image is None else {"image": image}
        try:
            lat = self.pipe(prompt=prompt, negative_prompt=self.neg, num_frames=self.frames, height=h, width=w,
                            fps=self.fps, num_inference_steps=self.steps, guidance_scale=self.guidance, generator=g,
                            output_type="latent", enable_safety_check=False, **kw).video
        except Stop:
            return None
        finally:
            self.run["max_steps"] = None
        return lat.float()[None] if lat.dim() == 4 else lat.float()

    def generate(self, clip, hook=None, seed=SEED, steps=None):
        image = self.condition_image(clip)
        with self.hooked(hook):
            return self._sample(image, self.prompt(clip), seed, self.h, self.w, steps)

    @torch.no_grad()
    def decode(self, latent):
        self.ensure_loaded()
        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype, device="cuda").view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=vae.dtype)).to("cuda").view(1, -1, 1, 1, 1)
        z = latent.to("cuda", vae.dtype) / inv_std + mean
        return to_uint8(vae.decode(z).sample[0])

    def out_proj(self, layer, step=0):
        m = self.tr.layers[layer].self_attn.to_add_out
        return m.weight if hasattr(m, "weight") else m[0].weight
