from __future__ import annotations

import math
import os
import sys
import types

import torch

from ._wanapi import SEED, VideoAdapter, to_uint8

TRIG_T = [math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)]


def _hook_compute_attention(adapter, layer):
    def compute_attention(self, q, k, v, video_size=None, kv_cache_cfg=None):
        adapter.ctx.layer = layer
        o = adapter.attend(q, k, v)
        return self.output_dropout(self.output_proj(o.reshape(q.shape[0], q.shape[1], -1).type_as(q)))
    return compute_attention


class CosmosPredict25(VideoAdapter):
    name = "cp25"
    steps = 4
    layout = "steps"
    rows = (1, 2, 3)
    branches = ("cond",)
    n_layers = 28
    n_heads = 16
    head_dim = 128
    latent_grid = (24, 44, 80)
    block_shape = None

    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["sampler"]
        self.h, self.w, self.frames, self.fps = s["height"], s["width"], s["frames"], s["fps"]
        self.inf = self.model = self.blocks = None
        self._te_op = {}

    def _inference(self):
        repo = self.cfg["model"]["repo"]
        local = self.cfg["checkpoints"]
        tokenizer = self.path("tokenizer")
        cwd = os.getcwd()
        if repo not in sys.path:
            sys.path.insert(0, repo)
        os.chdir(repo)
        try:
            from cosmos_predict2._src.imaginaire.utils import checkpoint_db as CDB

            def local_hf(cmd_args):
                root = local[cmd_args[0]]
                i = cmd_args.index("--revision") + 2 if "--revision" in cmd_args else len(cmd_args)
                p = os.path.join(root, cmd_args[i]) if i < len(cmd_args) and not cmd_args[i].startswith("--") else root
                if p.endswith("tokenizer.pth") and not os.path.exists(p):
                    p = tokenizer
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                return p
            CDB._hf_download = local_hf
            from cosmos_oss.init import init_environment
            init_environment()
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
            from cosmos_predict2.config import MODEL_CHECKPOINTS, MODEL_KEYS
            ck = MODEL_CHECKPOINTS[MODEL_KEYS["2B/distilled"]]
            return Video2WorldInference(
                experiment_name=ck.experiment, ckpt_path=ck.s3.uri, s3_credential_path="", context_parallel_size=1,
                config_file="cosmos_predict2/_src/predict2/distill/configs/registry_predict2p5.py",
                experiment_opts=["model.config.init_student_with_teacher=False"])
        finally:
            os.chdir(cwd)

    def load(self, device: str = "cuda") -> None:
        self.inf = self._inference()
        self.model = self.inf.model
        self.blocks = self.model.net.blocks
        for i, blk in enumerate(self.blocks):
            self._te_op[i] = blk.self_attn.attn_op
            blk.self_attn.compute_attention = types.MethodType(_hook_compute_attention(self, i), blk.self_attn)
        te = getattr(self.model, "text_encoder", None)
        if te is not None and getattr(te, "model", None) is not None:
            te.model.to("cpu")
            torch.cuda.empty_cache()
        self.loaded = True

    def dense(self, q, k, v):
        if self.backend == "native":
            return self._te_op[self.ctx.layer](q, k, v)
        return super().dense(q, k, v)

    def text(self, clip: str) -> torch.Tensor:
        p = self.cache_file("text", clip)
        if not os.path.exists(p):
            self.encode_texts([clip])
        return torch.load(p)["emb"]

    @torch.no_grad()
    def encode_texts(self, clips: list[str]) -> None:
        todo = [c for c in clips if not os.path.exists(self.cache_file("text", c))]
        if not todo:
            return
        self.ensure_loaded()
        te = self.model.text_encoder
        te.model.to("cuda")
        for c in todo:
            e = te.compute_text_embeddings_online(data_batch={"ai_caption": [self.prompt(c)], "images": None},
                                                  input_caption_key="ai_caption")
            p = self.cache_file("text", c)
            torch.save({"emb": e[0].to(torch.bfloat16).cpu(), "text": self.prompt(c)}, p + ".tmp")
            os.replace(p + ".tmp", p)
        te.model.to("cpu")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.ensure_loaded()
        m = self.model
        n = self.steps if steps is None else steps
        emb = self.text(clip)
        vid = torch.zeros(1, 3, self.frames, self.h, self.w, dtype=torch.uint8, device="cuda")
        batch = {"dataset_name": "video_data", "video": vid,
                 "fps": torch.full((1,), float(self.fps), device="cuda").bfloat16(),
                 "padding_mask": torch.zeros(1, 1, self.h, self.w, device="cuda").bfloat16(),
                 "num_conditional_frames": 0, "t5_text_embeddings": emb[None].to("cuda", torch.bfloat16)}
        st = {"i": 0}
        orig = m.denoise

        def denoise(*a, **kw):
            i = st["i"]
            self.ctx.at(step=i, branch="cond")
            self.ctx.sigma = TRIG_T[i] if i < len(TRIG_T) else 0.0
            with self.timed(f"s{i}_cond"):
                out = orig(*a, **kw)
            st["i"] += 1
            return out

        m.denoise = denoise
        try:
            with self.hooked(hook):
                lat = m.generate_samples_from_batch(batch, seed=seed, n_sample=1, num_steps=n)
        finally:
            m.denoise = orig
        return lat.float()

    @torch.no_grad()
    def decode(self, latent):
        self.ensure_loaded()
        return to_uint8(self.model.decode(latent.to("cuda"))[0])

    def out_proj(self, layer, step=0):
        return self.blocks[layer].self_attn.output_proj.weight
