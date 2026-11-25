from __future__ import annotations

import os
import sys

import numpy as np
import torch

from ..constants import SEED
from ._newrenderers import RenderAdapter, Stop, caption, nearest_frames

NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，"
            "倒着走")
GBUFFERS = ("basecolor", "depth", "metallic", "normal", "roughness")
TILES = {"tile_size": (30, 52), "tile_stride": (15, 26)}


class AlayaRenderer(RenderAdapter):
    name = "alaya"
    family = "renderer"
    steps = 50
    layout = "rows"
    rows = (1, 2, 4, 7, 11, 16, 22, 29, 37, 45)
    branches = ("cond", "uncond")
    n_layers = 30
    n_heads = 12
    head_dim = 128
    latent_grid = (21, 30, 52)
    block_shape = (4, 4, 8)
    frames, height, width = 81, 480, 832
    cfg_scale, shift = 5.0, 5.0

    def load(self, device: str = "cuda") -> None:
        repo = self.path("model", "repo")
        for p in (os.path.join(repo, "examples", "wanvideo", "model_inference"), repo):
            if p not in sys.path:
                sys.path.insert(0, p)
        from diffsynth.models import wan_video_dit as WD
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
        from gbuffer_utils import expand_patch_embedding, inject_gbuffer_unit
        ck = self.cfg["checkpoints"]
        wan = ck["wan"]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cuda",
            model_configs=[ModelConfig(path=os.path.join(ck["alaya"], "model.safetensors")),
                           ModelConfig(path=os.path.join(wan, "models_t5_umt5-xxl-enc-bf16.pth")),
                           ModelConfig(path=os.path.join(wan, "Wan2.1_VAE.pth"))],
            tokenizer_config=ModelConfig(path=os.path.join(wan, "google", "umt5-xxl")))
        expand_patch_embedding(self.pipe, num_gbuffers=len(GBUFFERS))
        inject_gbuffer_unit(self.pipe)
        blocks = self.pipe.dit.blocks
        sa = blocks[0].self_attn
        assert (len(blocks), sa.num_heads, sa.head_dim) == (self.n_layers, self.n_heads, self.head_dim)
        self._wo = [b.self_attn.o.weight for b in blocks]
        layer_of = {id(b.self_attn): i for i, b in enumerate(blocks)}
        nh, ad = self.n_heads, self

        def sa_forward(mod, x, freqs):
            q = mod.norm_q(mod.q(x))
            k = mod.norm_k(mod.k(x))
            v = mod.v(x)
            q = WD.rope_apply(q, freqs, nh)
            k = WD.rope_apply(k, freqs, nh)
            ad.ctx.layer = layer_of[id(mod)]
            B, S, _ = q.shape
            o = ad.route(q.view(B, S, nh, -1), k.view(B, S, nh, -1), v.view(B, S, nh, -1))
            return mod.o(o)

        WD.SelfAttention.forward = sa_forward
        self.loaded = True

    def gbuffers(self, clip: str, passes=GBUFFERS) -> list:
        from PIL import Image
        d = os.path.join(self.clip_dir(clip), "gbuffer")
        n = len([f for f in os.listdir(d) if f.endswith(".basecolor.png")])
        out = []
        for mod in passes:
            files = sorted(f for f in os.listdir(d) if f.endswith(f".{mod}.png"))
            cache, vid = {}, []
            for i in nearest_frames(n, self.frames):
                if i not in cache:
                    im = Image.open(os.path.join(d, files[i])).convert("RGB")
                    w, h = im.size
                    cw = round(h * self.width / self.height)
                    x0 = (w - cw) // 2
                    cache[i] = im.crop((x0, 0, x0 + cw, h)).resize((self.width, self.height), Image.LANCZOS)
                vid.append(cache[i])
            out.append(vid)
        return out

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.begin(hook, steps)
        pipe = self.pipe
        posi, nega = {"prompt": caption(self.clip_dir(clip))}, {"negative_prompt": NEGATIVE}
        shared = {"input_image": None, "end_image": None, "input_video": None, "denoising_strength": 1.0,
                  "control_video": None, "reference_image": None, "vace_video": None, "vace_video_mask": None,
                  "vace_reference_image": None, "vace_scale": 1.0, "seed": seed, "rand_device": "cpu",
                  "height": self.height, "width": self.width, "num_frames": self.frames, "cfg_scale": self.cfg_scale,
                  "cfg_merge": False, "sigma_shift": self.shift, "motion_bucket_id": None, "longcat_video": None,
                  "tiled": True, **TILES, "sliding_window_size": None, "sliding_window_stride": None,
                  "input_audio": None, "audio_sample_rate": 16000, "s2v_pose_video": None, "audio_embeds": None,
                  "s2v_pose_latents": None, "motion_video": None, "animate_pose_video": None,
                  "animate_face_video": None, "animate_inpaint_video": None, "animate_mask_video": None,
                  "vap_video": None, "gbuffer_videos": self.gbuffers(clip)}
        pipe.scheduler.set_timesteps(self.steps, denoising_strength=1.0, shift=self.shift)
        for unit in pipe.units:
            shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        try:
            for i, ts in enumerate(pipe.scheduler.timesteps):
                ts = ts.unsqueeze(0).to(dtype=pipe.torch_dtype, device="cuda")
                self.ctx.step, self.ctx.branch = i, "cond"
                with self.dit_call(f"s{i}_cond"):
                    p = pipe.model_fn(**models, **shared, **posi, timestep=ts)
                self.ctx.branch = "uncond"
                with self.dit_call(f"s{i}_uncond"):
                    ng = pipe.model_fn(**models, **shared, **nega, timestep=ts)
                shared["latents"] = pipe.scheduler.step(ng + self.cfg_scale * (p - ng), pipe.scheduler.timesteps[i],
                                                        shared["latents"])
        except Stop:
            return None
        finally:
            self.hook = None
        return shared["latents"]

    @torch.no_grad()
    def decode(self, latent):
        pipe = self.pipe
        pipe.load_models_to_device(["vae"])
        frames = pipe.vae_output_to_video(pipe.vae.decode(latent, device="cuda", tiled=True, **TILES))
        return np.stack([np.asarray(f.convert("RGB")) for f in frames]).astype(np.uint8)

    def conditions(self, clip):
        base = self.gbuffers(clip, ("basecolor",))[0]
        return {"basecolor": np.stack([np.asarray(f) for f in base]).astype(np.uint8)}
