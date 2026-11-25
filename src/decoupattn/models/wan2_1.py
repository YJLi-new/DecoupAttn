from __future__ import annotations

import torch

from ._wanapi_wan import WanFamily, install_self_attn


class Wan21(WanFamily):
    name = "wan21"
    latent_grid = (21, 30, 52)
    block_shape = (4, 4, 8)

    def load_transformer(self, device):
        from diffusers import WanTransformer3DModel
        tr = WanTransformer3DModel.from_single_file(self.path("lightx2v"), config=self.path("diffusers"),
                                                    subfolder="transformer", torch_dtype=torch.bfloat16)
        tr = tr.to(device).eval().requires_grad_(False)
        install_self_attn(self, tr)
        return tr

    def out_proj(self, layer, step=0):
        return self.tr.blocks[layer].attn1.to_out[0].weight
