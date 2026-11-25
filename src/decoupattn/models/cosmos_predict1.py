from __future__ import annotations

import os
import sys

import torch

from ..constants import SEED
from ._newrenderers import RenderAdapter, Stop, caption, link, read_pngs


class CosmosPredict1(RenderAdapter):
    name = "cp1"
    family = "video"
    steps = 35
    layout = "rows"
    rows = (1, 2, 4, 7, 11, 15, 20, 26, 32)
    branches = ("cond", "uncond")
    n_layers = 28
    n_heads = 32
    head_dim = 128
    latent_grid = (16, 44, 80)
    block_shape = None
    frames, height, width, fps = 121, 704, 1280, 24
    guidance = 7.0
    checkpoint = "Cosmos-Predict1-7B-Video2World"

    def load(self, device: str = "cuda") -> None:
        repo = self.path("model", "repo")
        ck = self.cfg["checkpoints"]
        wd = self.work_dir()
        link(os.path.join(wd, "cosmos_predict1"), os.path.join(repo, "cosmos_predict1"))
        link(os.path.join(wd, "checkpoints", self.checkpoint), ck["predict1"])
        link(os.path.join(wd, "checkpoints", "Cosmos-Tokenize1-CV8x8x8-720p"), ck["tokenizer"])
        os.chdir(wd)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import argparse

        from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
        from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline
        from cosmos_predict1.diffusion.module import attention as Am
        p = argparse.ArgumentParser()
        add_common_arguments(p)
        self.negative = p.parse_args([]).negative_prompt
        pipe = self.pipe = DiffusionVideo2WorldGenerationPipeline(
            inference_type="video2world", checkpoint_dir="checkpoints", checkpoint_name=self.checkpoint,
            enable_prompt_upsampler=False, offload_text_encoder_model=True, disable_guardrail=True,
            guidance=self.guidance, num_steps=self.steps, height=self.height, width=self.width, fps=self.fps,
            num_video_frames=self.frames, seed=SEED, num_input_frames=1)
        emb_file = ck.get("t5_embeddings")
        if emb_file:
            emb = torch.load(emb_file, weights_only=False)

            def text_emb(prompts, **kw):
                missing = [x for x in prompts if x not in emb]
                assert not missing, f"prompt(s) without a precomputed T5 embedding: {missing[:1]}"
                return [emb[x].cuda() for x in prompts], [None] * len(prompts)
            pipe._run_text_embedding_on_prompt_with_offload = text_emb
        model = pipe.model
        attns = [m for m in model.net.modules() if isinstance(m, Am.Attention) and m.is_selfattn]
        assert (len(attns), attns[0].heads, attns[0].dim_head) == (self.n_layers, self.n_heads, self.head_dim)
        self._wo = [m.to_out[0].weight for m in attns]
        layer_of = {id(m): i for i, m in enumerate(attns)}
        orig = Am.Attention.cal_attn
        ad = self

        def cal_attn(mod, q, k, v, mask=None):
            if not mod.is_selfattn:
                return orig(mod, q, k, v, mask)
            S, B, H, D = q.shape
            assert B == 1
            ad.ctx.layer = layer_of[id(mod)]
            ad.native = lambda q_, k_, v_: mod.attn_op(q_.view(S, 1, H, D), k_.view(S, 1, H, D), v_.view(S, 1, H, D),
                                                       core_attention_bias_type="no_bias",
                                                       core_attention_bias=None).view(1, S, H * D)
            o = ad.route(*(t.contiguous().view(1, S, H, D) for t in (q, k, v)))
            return mod.to_out(o.view(S, 1, H * D))
        Am.Attention.cal_attn = cal_attn
        self.st = st = {"call": 0, "latent": None}
        fwd = model.net.forward

        def forward(*a, **kw):
            i = st["call"]
            ad.ctx.step, ad.ctx.branch = i // 2, ("cond", "uncond")[i % 2]
            st["call"] += 1
            with ad.dit_call(f"s{i // 2}_{ad.ctx.branch}"):
                return fwd(*a, **kw)
        model.net.forward = forward

        def stop_at_decode(sample):
            st["latent"] = sample
            raise Stop()
        self._decode = pipe._run_tokenizer_decoding
        pipe._run_tokenizer_decoding = stop_at_decode
        self.loaded = True

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.begin(hook, steps)
        from cosmos_predict1.utils import misc
        st = self.st
        st["call"], st["latent"] = 0, None
        misc.set_random_seed(seed)
        self.pipe.seed = seed
        try:
            self.pipe.generate(prompt=caption(self.clip_dir(clip)), negative_prompt=self.negative,
                               image_or_video_path=os.path.join(self.clip_dir(clip), "gt", "rgb", "0000.png"))
        except Stop:
            pass
        finally:
            self.hook = None
        if self.stop_at is not None:
            return None
        assert st["call"] == 2 * self.steps and st["latent"] is not None, st["call"]
        return st["latent"]

    @torch.no_grad()
    def decode(self, latent):
        video = self._decode(latent)
        assert video.shape == (self.frames, self.height, self.width, 3), video.shape
        return video

    def conditions(self, clip):
        return {"image": read_pngs([os.path.join(self.clip_dir(clip), "gt", "rgb", "0000.png")])}
