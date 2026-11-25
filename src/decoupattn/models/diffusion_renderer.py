from __future__ import annotations

import json
import os
import sys

import torch

from ..constants import SEED
from ._newrenderers import RenderAdapter, Stop, read_pngs

PASSES = ("basecolor", "normal", "depth", "roughness", "metallic")
FPS = 24


def env_params(clip_dir: str, hdri_dir: str) -> dict:
    with open(os.path.join(clip_dir, "render.json")) as f:
        h = json.load(f)["spec"]["hdri"]
    return {"hdri": os.path.join(hdri_dir, os.path.basename(h["path"])), "env_flip": False, "env_rot": 90.0 + h["rot_deg"],
            "env_strength": h["strength"], "pose_file": os.path.join(clip_dir, "transforms.json")}


class DiffusionRenderer(RenderAdapter):
    name = "dr"
    family = "renderer"
    steps = 4
    layout = "steps"
    rows = (1, 2, 3)
    branches = ("cond",)
    n_layers = 28
    n_heads = 32
    head_dim = 128
    latent_grid = (8, 44, 80)
    block_shape = (2, 4, 16)
    frames, height, width = 57, 704, 1280
    checkpoint = "Diffusion_Renderer_Forward_Cosmos_7B"

    def load(self, device: str = "cuda") -> None:
        repo = self.path("model", "repo")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        os.chdir(repo)
        from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
        from cosmos_predict1.diffusion.module import attention as Am
        from cosmos_predict1.utils import misc
        misc.set_random_seed(SEED)
        self.pipe = DiffusionRendererPipeline(
            checkpoint_dir=self.cfg["checkpoints"]["dir"], checkpoint_name=self.checkpoint, guidance=0.0,
            num_steps=self.steps, height=self.height, width=self.width, fps=FPS,
            num_video_frames=self.frames, seed=SEED)
        model = self.model = self.pipe.model
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
                                                       core_attention_bias=None).reshape(1, S, H * D)
            o = ad.route(*(t.contiguous().view(1, S, H, D) for t in (q, k, v)))
            return mod.to_out(o.view(S, 1, H * D))
        Am.Attention.cal_attn = cal_attn
        self.st = st = {"call": 0, "cond": None}
        if self.cfg["model"].get("forward", "fast") == "fast":
            from ._dr_fast import FastNet, pin_autotuning
            pin_autotuning(self.cfg["model"].get("inductor_pins") or None)
            fwd = FastNet(model.net)
        else:
            fwd = model.net.forward

        def forward(*a, **kw):
            ad.ctx.step, ad.ctx.branch = st["call"], "cond"
            st["call"] += 1
            if st["cond"] is not None:
                lc = kw["latent_condition"]
                assert lc.shape == st["cond"].shape and lc.dtype == st["cond"].dtype, (lc.shape, st["cond"].shape)
                kw["latent_condition"] = st["cond"]
            with ad.dit_call(f"s{ad.ctx.step}_cond"):
                return fwd(*a, **kw)
        model.net.forward = forward
        self.loaded = True

    def batch(self, clip: str) -> dict:
        from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn
        from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import VideoGBufferDataset
        from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import envmap_vec
        from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import process_environment_map
        c = self.clip_dir(clip)
        ep = env_params(c, self.path("data", "hdri"))
        dev = torch.device("cuda")
        envd = process_environment_map(ep["hdri"], resolution=(self.height, self.width), num_frames=self.frames,
                                       fixed_pose=False, pose_file=ep["pose_file"], env_flip=ep["env_flip"], env_rot=ep["env_rot"], env_strength=ep["env_strength"],
                                       env_format=["proj"], device=dev)
        ds = VideoGBufferDataset(root_dir=os.path.join(c, "gbuffer"), sample_n_frames=self.frames,
                                 resolution=(self.height, self.width))
        assert len(ds) == 1, len(ds)
        batch = dict_collation_fn([ds[0]])
        assert all(batch[k].shape[2] == self.frames for k in PASSES), "missing G-buffer frames"
        batch["env_ldr"] = envd["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        batch["env_log"] = envd["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        nrm = envmap_vec([self.height, self.width], device=dev).unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3)
        batch["env_nrm"] = nrm.expand_as(batch["env_ldr"])
        return batch

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.begin(hook, steps)
        from cosmos_predict1.utils import misc
        m, st = self.model, self.st
        cpt = os.path.join(self.clip_dir(clip), "teacher", "cond.pt")
        st["call"], st["cond"] = 0, (torch.load(cpt).to("cuda") if os.path.exists(cpt) else None)
        db = misc.to(self.batch(clip), device="cuda", dtype=torch.bfloat16)
        if "video" not in db:
            for a in ("rgb",) + PASSES:
                if a in db:
                    db["video"] = db[a]
                    break
        Fq = (db["video"].shape[2] - 1) // 8 + 1
        Hq = db["video"].shape[3] // m.tokenizer.spatial_compression_factor
        Wq = db["video"].shape[4] // m.tokenizer.spatial_compression_factor
        misc.set_random_seed(seed)
        cond, _ = m._get_conditions(db, False)
        xt = torch.randn(size=(1, m.tokenizer.channel, Fq, Hq, Wq)) * m.scheduler.init_noise_sigma
        sch = m.scheduler
        sch.set_timesteps(self.steps)
        try:
            for t in sch.timesteps:
                xt = xt.to(**m.tensor_kwargs)
                xs = sch.scale_model_input(xt, timestep=t)
                t = t.to(**m.tensor_kwargs)
                out = m.net(x=xs, timesteps=t, **cond.to_dict())
                xt = sch.step(out, t, xt).prev_sample
        except Stop:
            return None
        finally:
            self.hook = None
            st["cond"] = None
        assert st["call"] == self.steps, st["call"]
        return xt

    def dense_heads(self, q, k, v, idx):
        if self.backend != "native":
            return super().dense_heads(q, k, v, idx)
        _, S, H, D = q.shape
        o = self.dense(q, k, v).view(S, H, D)
        return o[:, torch.as_tensor(idx, device=o.device)].unsqueeze(0).reshape(1, S, -1)

    @torch.no_grad()
    def decode(self, latent):
        torch.cuda.empty_cache()
        video = (1.0 + self.model.decode(latent)).clamp(0, 2) / 2
        return (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

    def conditions(self, clip):
        d = os.path.join(self.clip_dir(clip), "gbuffer")
        out = {}
        for p in ("basecolor", "normal"):
            files = sorted(f for f in os.listdir(d) if f.endswith(f".{p}.png"))[:self.frames]
            out[p] = read_pngs([os.path.join(d, f) for f in files])
        return out
