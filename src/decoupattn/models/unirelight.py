from __future__ import annotations

import os
import sys

import torch

from ..constants import SEED
from ._newrenderers import RenderAdapter, Stop, link, read_pngs, to_uint8

RESOLUTION = [480, 848]


class UniRelight(RenderAdapter):
    name = "unirelight"
    family = "renderer"
    steps = 35
    layout = "rows"
    rows = (1, 2, 4, 7, 11, 15, 20, 26, 32)
    branches = ("cond", "uncond")
    n_layers = 28
    n_heads = 32
    head_dim = 128
    latent_grid = (24, 30, 53)
    block_shape = None
    step_clamp = 34
    frames = 57

    def load(self, device: str = "cuda") -> None:
        repo = self.path("model", "repo")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        wd = self.work_dir()
        link(os.path.join(wd, "cosmos_predict1"), os.path.join(repo, "cosmos_predict1"))
        link(os.path.join(wd, "checkpoints", "Cosmos-Tokenize1-CV8x8x8-720p"), self.cfg["checkpoints"]["tokenizer"])
        os.chdir(wd)
        from cosmos_predict1.diffusion.inference.inference_utils import load_model_by_config, load_network_model
        from cosmos_predict1.diffusion.module import attention as Am
        from cosmos_predict1.diffusion.training.models.model_relight import MultiviewExtendDiffusionModelRelight
        model, _ = load_model_by_config(config_job_name="unirelight_cosmos_f57_480p",
                                        config_file="cosmos_predict1/diffusion/training/config/config_relight.py",
                                        model_class=MultiviewExtendDiffusionModelRelight)
        load_network_model(model=model, ckpt_path=os.path.join(self.cfg["checkpoints"]["unirelight"], "model.pt"))
        self.model = model = model.to(torch.bfloat16)
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
        self.st = st = {"call": 0, "den": 0}
        orig_x0, orig_den = model.get_x0_fn_from_batch_with_condition_latent, model.denoise

        def get_x0(*a, **kw):
            f = orig_x0(*a, **kw)

            def g(x, s):
                ad.ctx.step = st["call"]
                st["call"] += 1
                st["den"] = 0
                return f(x, s)
            return g

        def denoise(*a, **kw):
            ad.ctx.branch = "cond" if st["den"] == 0 else "uncond"
            st["den"] += 1
            with ad.dit_call(f"s{ad.ctx.step}_{ad.ctx.branch}"):
                return orig_den(*a, **kw)

        model.get_x0_fn_from_batch_with_condition_latent = get_x0
        model.denoise = denoise
        self.loaded = True

    def target_light(self, clip: str) -> str:
        import json
        sp = self.split_lists()
        for k in ("test", "calib", "val"):
            lst = sp[k]
            if clip in lst:
                nxt = lst[(lst.index(clip) + 1) % len(lst)]
                with open(os.path.join(self.clip_dir(nxt), "render.json")) as f:
                    p = json.load(f)["spec"]["hdri"]["path"]
                return os.path.join(self.path("data", "hdri"), os.path.basename(p))
        raise KeyError(clip)

    def input_dir(self, clip: str) -> str:
        d = os.path.join(self.work_dir(), "input", clip, clip)
        os.makedirs(d, exist_ok=True)
        g = os.path.join(self.clip_dir(clip), "gt", "rgb")
        for i, f in enumerate(sorted(os.listdir(g))[:self.frames]):
            link(os.path.join(d, f"{i:05d}.rgb.png"), os.path.join(g, f))
        return os.path.dirname(d)

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.begin(hook, steps)
        from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader import dataloader_utils as DU
        from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader import rendering_utils as RU
        from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader import utils_env_proj as EP
        from cosmos_predict1.diffusion.training.datasets.diffusion_renderer_dataloader.dataset_inference import (
            VideoGBufferDataset,
        )
        from cosmos_predict1.utils import misc
        model, st = self.model, self.st
        st["call"] = 0
        misc.set_random_seed(seed=seed, by_rank=True)
        ds = VideoGBufferDataset(root_dir=self.input_dir(clip), sample_n_frames=self.frames, image_extensions=None,
                                 resolution=RESOLUTION, resize_resolution=[486, 864], gbuf_labels=["rgb"],
                                 group_mode="folder", bg_color=(1., 1., 1.))
        batch = misc.to(DU.dict_collation_fn([ds[0]]), device="cuda", dtype=torch.bfloat16)
        batch["rgb_ref"] = batch["rgb"]
        dev = torch.device("cuda")
        ed = EP.process_environment_map(self.target_light(clip), resolution=RESOLUTION, num_frames=self.frames,
                                        fixed_pose=True, rotate_envlight=False, env_format=["proj"], device=dev)
        env_ldr = ed["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        env_log = ed["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        env_nrm = RU.envmap_vec(RESOLUTION, device=dev).unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3)
        batch.update(env_ldr=env_ldr, env_log=env_log, env_nrm=env_nrm.expand_as(env_ldr), video=batch["rgb_ref"])
        C = model.state_shape[0]
        F_ = (batch["rgb_ref"].shape[2] - 1) // 8 + 1
        Hh, Ww = batch["rgb_ref"].shape[3] // 8, batch["rgb_ref"].shape[4] // 8
        try:
            return model.generate_samples_from_batch(batch, guidance=0.0, seed=seed, state_shape=(C, F_ * 3, Hh, Ww),
                                                     is_negative_prompt=False, use_gt_albedo=False, num_steps=self.steps)
        except Stop:
            return None
        finally:
            self.hook = None

    @torch.no_grad()
    def decode(self, latent):
        video = self.model.decode(latent)
        return to_uint8(torch.chunk(video, 3, dim=2)[1][0])

    def conditions(self, clip):
        g = os.path.join(self.clip_dir(clip), "gt", "rgb")
        return {"rgb": read_pngs([os.path.join(g, f) for f in sorted(os.listdir(g))[:self.frames]])}
