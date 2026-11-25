from __future__ import annotations

import numpy as np
import torch

from .constants import KERNEL_BLOCK

QUERY_TILING = "frame"


class Tiling:
    def __init__(self, F, H, Wd, tile=(1, 8, 16), device="cpu"):
        tf, th, tw = tile
        self.F, self.H, self.W, self.tile = F, H, Wd, tuple(tile)
        self.V = tf * th * tw
        assert self.V == 128, "query tiles are 128 rows (one FA3 query block)"
        Fp, Hp, Wp = -(-F // tf) * tf, -(-H // th) * th, -(-Wd // tw) * tw
        self.nf, self.nh, self.nw = Fp // tf, Hp // th, Wp // tw
        self.nt = self.nf * self.nh * self.nw
        self.S = F * H * Wd
        self.S_pad = self.nt * self.V
        f, h, w = np.meshgrid(np.arange(Fp), np.arange(Hp), np.arange(Wp), indexing="ij")
        f, h, w = f.ravel(), h.ravel(), w.ravel()
        tid = (f // tf) * (self.nh * self.nw) + (h // th) * self.nw + (w // tw)
        real = (f < F) & (h < H) & (w < Wd)
        src = np.where(real, f * H * Wd + np.minimum(h, H - 1) * Wd + np.minimum(w, Wd - 1), -1)
        order = np.argsort(tid, kind="stable")
        self.slot_src = torch.from_numpy(src[order]).to(device)
        self.slot_tile = torch.from_numpy(tid[order]).to(device)
        self.real = self.slot_src >= 0
        tok_tile = np.empty(self.S, np.int64)
        tok_tile[src[real]] = tid[real]
        self.tok_tile = torch.from_numpy(tok_tile).to(device)
        self.n_real = torch.bincount(self.tok_tile, minlength=self.nt)
        pad_tiles = torch.repeat_interleave(torch.arange(self.nt, device=device), self.V - self.n_real)
        self.cq = torch.cat([self.tok_tile, pad_tiles])
        self.ck = torch.cat([self.tok_tile, torch.full((self.S_pad - self.S,), self.nt, device=device)])

    @property
    def device(self):
        return self.tok_tile.device

    def to(self, device) -> Tiling:
        return Tiling(self.F, self.H, self.W, self.tile, device)

    def pad(self, x):
        return torch.cat([x, x.new_zeros((self.S_pad - self.S,) + tuple(x.shape[1:]))])


def tiling_for_plan(plan, grid, device="cpu") -> Tiling:
    assert plan.query_tiling == QUERY_TILING, plan.query_tiling
    return Tiling(*grid, plan.block_shape, device)


def own_mask(til) -> torch.Tensor:
    return torch.eye(til.nt, dtype=torch.bool, device=til.n_real.device)


def pair_weights(til) -> torch.Tensor:
    nr = til.n_real.double()
    return nr[:, None] * nr[None, :]


def exec_density(mask_hnn, til, cta_q=KERNEL_BLOCK[0], cta_kv=KERNEL_BLOCK[1]) -> float:
    nr = til.n_real.to(mask_hnn.device).double()
    row = (mask_hnn.double() * nr).sum(-1)
    ex = (torch.ceil(row / cta_kv) * cta_kv * cta_q).sum()
    return float(ex / (mask_hnn.shape[0] * til.S ** 2))


def exec_density_heads(mask, nr_k, S, cta_q=KERNEL_BLOCK[0], cta_kv=KERNEL_BLOCK[1]) -> torch.Tensor:
    row = (mask.double() * nr_k.double()[None, None, :]).sum(-1)
    return (torch.ceil(row / cta_kv) * cta_kv * cta_q).sum(-1) / float(S) ** 2


def real_density_heads(mask, w, S) -> torch.Tensor:
    return (mask.double() * w.double()[None]).sum((1, 2)) / float(S) ** 2


def real_density(mask_hnn, til) -> float:
    dev = mask_hnn.device
    w = pair_weights(til).to(dev)
    return float((mask_hnn.double() * w).sum() / (mask_hnn.shape[0] * til.S ** 2))


def pseudo_density(mask_hnn, til) -> float:
    dev = mask_hnn.device
    nr = til.n_real.double().to(dev)
    unsel = (~mask_hnn).double() * (nr > 0).double()[None, None, :]
    return float((unsel.sum(-1) * nr[None]).sum() / (mask_hnn.shape[0] * til.S ** 2))
