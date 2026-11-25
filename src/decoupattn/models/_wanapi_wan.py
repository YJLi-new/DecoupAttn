from __future__ import annotations

import os

import numpy as np
import torch

from ._wanapi import SEED, VideoAdapter, to_uint8

STEP_LIST = (1000, 750, 500, 250)
SHIFT = 5.0


def lx4_schedule(step_list=STEP_LIST, shift=SHIFT):
    s = torch.linspace(1.0, 0.0, 1001)[:-1].double()
    s = shift * s / (1 + (shift - 1) * s)
    sig = s[[1000 - x for x in step_list]]
    return [float(x) for x in sig], [float(x) * 1000 for x in sig]


def init_noise(seed, h, w, f, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((1, 16, (f - 1) // 4 + 1, h // 8, w // 8), generator=g, device=device, dtype=torch.float32)


def rope_torch(x, freqs):
    xr = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))
    return torch.view_as_real(xr * freqs.transpose(1, 2)).flatten(3, 4).type_as(x)


def rope_qk(q, k, freqs):
    if q.is_cuda and q.dtype == torch.bfloat16 and q.shape[0] == 1 and freqs.dtype == torch.complex128:
        from ._wanapi_rope import rope
        return rope(q, freqs), rope(k, freqs)
    return rope_torch(q, freqs), rope_torch(k, freqs)


class WanSelfAttn:
    def __init__(self, adapter, layer):
        self.a, self.layer = adapter, layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        q = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        k = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if isinstance(rotary_emb, tuple):
            raise NotImplementedError("the Wan adapters follow diffusers 0.34 (complex RoPE); install diffusers==0.34.0")
        q, k = rope_qk(q, k, rotary_emb)
        self.a.ctx.layer = self.layer
        o = self.a.attend(q, k, v).type_as(q)
        return attn.to_out[1](attn.to_out[0](o))


def install_self_attn(adapter, transformer):
    for i, blk in enumerate(transformer.blocks):
        blk.attn1.set_processor(WanSelfAttn(adapter, i))


@torch.no_grad()
def encode_umt5(diffusers_dir: str, texts: list[str], device="cuda") -> list[torch.Tensor]:
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(diffusers_dir, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(diffusers_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    te = te.to(device).eval()
    out = []
    for t in texts:
        ti = tok([t], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                 return_attention_mask=True, return_tensors="pt")
        ids, m = ti.input_ids.to(device), ti.attention_mask.to(device)
        n = int(m.sum())
        e = te(ids, m).last_hidden_state[0, :n]
        out.append(torch.cat([e, e.new_zeros(512 - n, e.shape[1])]).to(torch.bfloat16).cpu())
    del te
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def wan_decode(vae, lat) -> np.ndarray:
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat)
    vid = vae.decode(lat / std + mean, return_dict=False)[0]
    return to_uint8(vid[0])


class WanFamily(VideoAdapter):
    steps = 4
    layout = "steps"
    rows = (1, 2, 3)
    branches = ("cond",)
    n_layers = 40
    n_heads = 40
    head_dim = 128

    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["sampler"]
        self.h, self.w, self.frames = s["height"], s["width"], s["frames"]
        self.tr = None
        self.vae = None
        self.device = "cuda"

    def load_transformer(self, device):
        raise NotImplementedError

    def load(self, device: str = "cuda") -> None:
        self.device = device
        self.tr = self.load_transformer(device)
        self.loaded = True

    def text(self, clip: str) -> torch.Tensor:
        p = self.cache_file("text", clip)
        if not os.path.exists(p):
            self.encode_texts([clip])
        return torch.load(p)["emb"]

    def encode_texts(self, clips: list[str]) -> None:
        todo = [c for c in clips if not os.path.exists(self.cache_file("text", c))]
        if not todo:
            return
        embs = encode_umt5(self.path("diffusers"), [self.prompt(c) for c in todo], self.device)
        for c, e in zip(todo, embs):
            p = self.cache_file("text", c)
            torch.save({"emb": e, "text": self.prompt(c)}, p + ".tmp")
            os.replace(p + ".tmp", p)

    def before_call(self, step: int) -> None:
        pass

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.ensure_loaded()
        e = self.text(clip)[None].to(self.device)
        lat = init_noise(seed, self.h, self.w, self.frames, self.device)
        sig, ts = lx4_schedule()
        n = len(sig) if steps is None else steps
        with self.hooked(hook):
            for i in range(n):
                self.ctx.at(step=i, branch="cond")
                self.ctx.sigma = sig[i]
                self.before_call(i)
                t = torch.tensor([ts[i]], device=self.device, dtype=torch.float32)
                with self.timed(f"s{i}_cond"):
                    vel = self.tr(lat.to(torch.bfloat16), t, e, return_dict=False)[0].float()
                x0 = lat - sig[i] * vel
                lat = x0 + sig[i + 1] * vel if i < n - 1 else x0
        return lat

    def load_vae(self):
        from diffusers import AutoencoderKLWan
        return AutoencoderKLWan.from_pretrained(self.path("diffusers"), subfolder="vae",
                                                torch_dtype=torch.float32).to(self.device).eval()

    def decode(self, latent):
        if self.vae is None:
            self.vae = self.load_vae()
        return wan_decode(self.vae, latent.to(self.device))
