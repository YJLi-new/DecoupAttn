from __future__ import annotations

import os
import re

import torch

from ._wanapi_wan import WanFamily, install_self_attn

BOUNDARY = 2


def expert_index(step: int) -> int:
    return 0 if step < BOUNDARY else 1


class ExpertPair(torch.nn.Module):
    def __init__(self, adapter, high, low, offload=False):
        super().__init__()
        self.experts = torch.nn.ModuleList([high, low])
        self.a, self.offload = adapter, offload

    @property
    def cur(self):
        return self.experts[expert_index(self.a.ctx.step)]

    def ensure(self, step):
        if not self.offload:
            return
        i = expert_index(step)
        if next(self.experts[i].parameters()).device.type != "cuda":
            self.experts[1 - i].to("cpu")
            torch.cuda.empty_cache()
            self.experts[i].to("cuda")
            torch.cuda.synchronize()

    def forward(self, *args, **kw):
        self.ensure(self.a.ctx.step)
        return self.cur(*args, **kw)


class Wan22(WanFamily):
    name = "wan22"
    latent_grid = (21, 30, 52)
    block_shape = None
    latency = "calls"

    def load_transformer(self, device):
        from diffusers import WanTransformer3DModel
        merged = self.path("merged")
        if not os.path.isdir(os.path.join(merged, "transformer_2")):
            merge_experts(self.cfg)
        offload = bool(self.cfg["sampler"].get("offload", False))
        ex = []
        for sub in ("transformer", "transformer_2"):
            t = WanTransformer3DModel.from_pretrained(merged, subfolder=sub, torch_dtype=torch.bfloat16)
            ex.append(t.to("cpu" if (offload and sub == "transformer_2") else device).eval().requires_grad_(False))
            install_self_attn(self, ex[-1])
        return ExpertPair(self, *ex, offload=offload)

    def before_call(self, step):
        self.tr.ensure(step)

    def expert(self, step):
        return expert_index(step)

    def out_proj(self, layer, step=0):
        return self.tr.experts[expert_index(step)].blocks[layer].attn1.to_out[0].weight


_KMAP = [("self_attn.q", "attn1.to_q"), ("self_attn.k", "attn1.to_k"), ("self_attn.v", "attn1.to_v"),
         ("self_attn.o", "attn1.to_out.0"), ("self_attn.norm_q", "attn1.norm_q"), ("self_attn.norm_k", "attn1.norm_k"),
         ("cross_attn.q", "attn2.to_q"), ("cross_attn.k", "attn2.to_k"), ("cross_attn.v", "attn2.to_v"),
         ("cross_attn.o", "attn2.to_out.0"), ("cross_attn.norm_q", "attn2.norm_q"), ("cross_attn.norm_k", "attn2.norm_k"),
         ("ffn.0", "ffn.net.0.proj"), ("ffn.2", "ffn.net.2"), ("norm3", "norm2"), ("modulation", "scale_shift_table")]
_TOP = {"text_embedding.0": "condition_embedder.text_embedder.linear_1",
        "text_embedding.2": "condition_embedder.text_embedder.linear_2",
        "time_embedding.0": "condition_embedder.time_embedder.linear_1",
        "time_embedding.2": "condition_embedder.time_embedder.linear_2",
        "time_projection.1": "condition_embedder.time_proj", "head.head": "proj_out", "head.modulation": "scale_shift_table",
        "patch_embedding": "patch_embedding"}


def module_name(orig: str) -> str:
    m = re.match(r"blocks\.(\d+)\.(.+)$", orig)
    if m:
        for a, b in _KMAP:
            if m.group(2) == a:
                return f"blocks.{m.group(1)}.{b}"
        raise KeyError(orig)
    if orig in _TOP:
        return _TOP[orig]
    raise KeyError(orig)


@torch.no_grad()
def merge_lightx2v_lora(tr, path: str, strength: float = 1.0) -> dict:
    from safetensors.torch import load_file
    sd = load_file(path)
    pre = "diffusion_model."
    sd = {k.removeprefix(pre): v for k, v in sd.items()}
    params = dict(tr.named_parameters())
    n = {"lora": 0, "diff": 0, "diff_b": 0}
    for k in sorted(sd):
        if k.endswith((".lora_down.weight", ".lora_A.weight")):
            base = k.rsplit(".", 2)[0]
            up = sd.get(base + ".lora_up.weight", sd.get(base + ".lora_B.weight"))
            dn = sd[k].float()
            alpha = sd.get(base + ".alpha")
            scale = (float(alpha) / dn.shape[0]) if alpha is not None else 1.0
            w = params[module_name(base) + ".weight"]
            w.add_((up.float().to(w.device) @ dn.to(w.device)) * (scale * strength))
            n["lora"] += 1
        elif k.endswith(".diff"):
            mod = module_name(k[:-5])
            w = params[mod if mod.endswith("scale_shift_table") else mod + ".weight"]
            w.add_(sd[k].float().to(w.device).view_as(w) * strength)
            n["diff"] += 1
        elif k.endswith(".diff_b"):
            w = params[module_name(k[:-7]) + ".bias"]
            w.add_(sd[k].float().to(w.device) * strength)
            n["diff_b"] += 1
        elif not k.endswith((".lora_up.weight", ".lora_B.weight", ".alpha")):
            raise KeyError(f"unhandled LoRA key {k}")
    return n


def merge_experts(cfg: dict, device: str = "cpu") -> None:
    from diffusers import WanTransformer3DModel
    p = cfg["checkpoints"]
    for sub, noise in (("transformer", "high"), ("transformer_2", "low")):
        out = os.path.join(p["merged"], sub)
        if os.path.isdir(out):
            continue
        tr = WanTransformer3DModel.from_pretrained(p["wan22"], subfolder=sub, torch_dtype=torch.float32).to(device)
        n = merge_lightx2v_lora(tr, p["lora"].format(noise=noise))
        print(f"wan22 {sub}: merged {n}", flush=True)
        tr.to(torch.bfloat16).save_pretrained(out + ".tmp", max_shard_size="10GB")
        os.replace(out + ".tmp", out)
        del tr
