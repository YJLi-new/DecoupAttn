from __future__ import annotations

import os
import sys

import numpy as np
import torch

from ..constants import SEED
from ._newrenderers import RenderAdapter, Stop, caption, nearest_frames, read_pngs, to_uint8


class CosmosTransfer25(RenderAdapter):
    name = "t25"
    family = "controlled"
    steps = 4
    layout = "steps"
    rows = (1, 2, 3)
    branches = ("cond",)
    n_layers = 32
    n_heads = 16
    head_dim = 128
    latent_grid = (24, 44, 80)
    block_shape = None
    frames = 93
    fps = 16

    def load(self, device: str = "cuda") -> None:
        os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"
        repo = self.path("model", "repo")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        os.chdir(repo)
        ck = self.cfg["checkpoints"]
        local = {"nvidia/Cosmos-Transfer2.5-2B": ck["transfer"], "nvidia/Cosmos-Predict2.5-2B": ck["predict"],
                 "nvidia/Cosmos-Reason1-7B": ck["reason"]}
        from cosmos_transfer2._src.imaginaire.utils import checkpoint_db as CDB

        def local_hf(cmd_args):
            root = local.get(cmd_args[0], os.path.join(ck["transfer"], "_absent", cmd_args[0]))
            i = cmd_args.index("--revision") + 2
            p = os.path.join(root, cmd_args[i]) if i < len(cmd_args) and not cmd_args[i].startswith("--") else root
            if not os.path.exists(p):
                print(f"[t25] checkpoint not local (only an error if it gets loaded): {p}", flush=True)
            return p
        CDB._hf_download = local_hf
        CDB.CheckpointFileHf._download = lambda self_: local_hf([self_.repository, "--repo-type", "model", "--revision",
                                                                  self_.revision, self_.filename])
        from cosmos_oss.init import init_environment
        init_environment()
        from cosmos_transfer2._src.transfer2.networks import siglip2_image_context as SGI

        def siglip_from_config(name):
            from transformers import AutoConfig, Siglip2VisionModel
            c = AutoConfig.from_pretrained(name)
            c.vision_config.vision_use_head = False
            return Siglip2VisionModel(c.vision_config).eval().to("cuda", dtype=torch.bfloat16), None
        sg_fwd = SGI.SigLip2EmbImgContext.forward

        def sg_forward(self_, input_tensor=None, image_context=None):
            assert image_context is None
            return sg_fwd(self_, input_tensor, image_context)
        SGI.get_siglip2_model_processor = siglip_from_config
        SGI.SigLip2EmbImgContext.forward = sg_forward
        from pathlib import Path

        from cosmos_transfer2._src.predict2.networks import minimal_v4_dit as MD
        from cosmos_transfer2.config import SetupArguments
        from cosmos_transfer2.inference import Control2WorldInference
        setup = SetupArguments(output_dir=Path(os.path.join(self.work_dir(), "run_cfg")), model="edge/distilled",
                               disable_guardrails=True)
        self.inf = Control2WorldInference(setup, batch_hint_keys=["edge"])
        assert self.inf.is_distilled
        self.pipe = self.inf.inference_pipeline
        model = self.model = self.pipe.model
        attns = [m for m in model.net.modules() if isinstance(m, MD.Attention) and m.is_selfattn]
        assert (len(attns), attns[0].n_heads, attns[0].head_dim) == (self.n_layers, self.n_heads, self.head_dim)
        self._wo = [m.output_proj.weight for m in attns]
        layer_of = {id(m): i for i, m in enumerate(attns)}
        orig = MD.Attention.compute_attention
        ad = self

        def compute_attention(mod, q, k, v, *a, **kw):
            if id(mod) not in layer_of:
                return orig(mod, q, k, v, *a, **kw)
            B, S, H, D = q.shape
            assert B == 1
            ad.ctx.layer = layer_of[id(mod)]
            ad.native = lambda q_, k_, v_: mod.attn_op(q_, k_, v_).reshape(1, S, H * D)
            o = ad.route(q.contiguous(), k.contiguous(), v.contiguous())
            return mod.output_dropout(mod.output_proj(o))
        MD.Attention.compute_attention = compute_attention
        self.st = st = {"call": 0}
        den, dec = model.denoise, model.decode

        def denoise(*a, **kw):
            ad.ctx.step, ad.ctx.branch = st["call"], "cond"
            st["call"] += 1
            with ad.dit_call(f"s{ad.ctx.step}_cond"):
                return den(*a, **kw)

        def decode(latent):
            st["latent"] = latent
            raise Stop()
        model.denoise, model.decode = denoise, decode
        self._decode = dec
        self.loaded = True

    def inputs(self, clip: str) -> str:
        d = os.path.join(self.work_dir(), "inputs", clip)
        if not os.path.exists(os.path.join(d, "rgb.mp4")):
            import imageio
            os.makedirs(d, exist_ok=True)
            g = os.path.join(self.clip_dir(clip), "gt", "rgb")
            files = sorted(os.listdir(g))
            cache = {}
            vid = []
            for i in nearest_frames(len(files), self.frames):
                if i not in cache:
                    cache[i] = read_pngs([os.path.join(g, files[i])])[0]
                vid.append(cache[i])
            tmp = os.path.join(d, "rgb.tmp.mp4")
            imageio.mimwrite(tmp, vid, fps=self.fps, quality=9, macro_block_size=1, ffmpeg_params=["-threads", "4"])
            os.replace(tmp, os.path.join(d, "rgb.mp4"))
        if not os.path.exists(os.path.join(d, "prompt.txt")):
            with open(os.path.join(d, "prompt.txt"), "w") as f:
                f.write(caption(self.clip_dir(clip)) + "\n")
        return d

    @torch.no_grad()
    def generate(self, clip, hook=None, seed=SEED, steps=None):
        self.begin(hook, steps)
        from cosmos_transfer2.config import EdgeConfig, InferenceArguments
        st = self.st
        st["call"], st["latent"] = 0, None
        d = self.inputs(clip)
        with open(os.path.join(d, "prompt.txt")) as f:
            prompt = f.read().strip()
        smp = InferenceArguments(name=clip, video_path=os.path.join(d, "rgb.mp4"), prompt=prompt, seed=seed,
                                 num_steps=self.steps, edge=EdgeConfig())
        try:
            self.pipe.generate_img2world(
                video_path=str(smp.video_path), prompt=prompt, negative_prompt=None, image_context_path=None,
                context_frame_idx=smp.context_frame_index, max_frames=smp.max_frames, guidance=None, seed=smp.seed,
                resolution=smp.resolution, control_weight=smp.control_weight_dict["edge"],
                sigma_max=None if smp.sigma_max is None else float(smp.sigma_max), hint_key=smp.hint_keys,
                input_control_video_paths=smp.control_modalities, show_control_condition=smp.show_control_condition,
                seg_control_prompt=smp.seg_control_prompt, show_input=smp.show_input,
                keep_input_resolution=not smp.not_keep_input_resolution, preset_blur_strength=smp.preset_blur_strength,
                preset_edge_threshold=smp.preset_edge_threshold, num_conditional_frames=smp.num_conditional_frames,
                num_video_frames_per_chunk=smp.num_video_frames_per_chunk, num_steps=smp.num_steps,
                guided_generation_mask=None, guided_generation_step_threshold=smp.guided_generation_step_threshold,
                guided_generation_foreground_labels=None)
        except Stop:
            pass
        finally:
            self.hook = None
        if self.stop_at is not None:
            return None
        assert st["call"] == self.steps and st["latent"] is not None, st["call"]
        return st["latent"]

    @torch.no_grad()
    def decode(self, latent):
        return to_uint8(self._decode(latent)[0])

    def conditions(self, clip):
        import cv2
        g = os.path.join(self.clip_dir(clip), "gt", "rgb")
        files = sorted(os.listdir(g))
        rgb = read_pngs([os.path.join(g, files[i]) for i in nearest_frames(len(files), self.frames)])
        edge = np.stack([cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 100, 200) for f in rgb])
        return {"rgb": rgb, "edge": np.repeat(edge[..., None], 3, -1)}
