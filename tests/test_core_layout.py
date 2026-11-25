import math

import torch

from decoupattn.constants import BLOCK_SHAPES, KERNEL_BLOCK
from decoupattn.layout import (
    Tiling,
    exec_density,
    exec_density_heads,
    own_mask,
    pair_weights,
    pseudo_density,
    real_density,
    real_density_heads,
)

GRID = (3, 10, 20)


def _random_mask(til, H, p=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(H, til.nt, til.nt, generator=g) < p
    return m | own_mask(til)[None]


def test_tiling_covers_every_token_once():
    for shape in BLOCK_SHAPES:
        til = Tiling(*GRID, shape)
        assert til.S == math.prod(GRID)
        assert int(til.n_real.sum()) == til.S
        assert til.S_pad == til.nt * 128
        real = til.slot_src[til.real]
        assert torch.equal(torch.sort(real).values, torch.arange(til.S))
        assert torch.equal(til.tok_tile[real], til.slot_tile[til.real])


def test_block_is_a_box_of_the_grid():
    F, H, W = GRID
    til = Tiling(F, H, W, (1, 8, 16))
    s = torch.arange(til.S)
    f, h, w = s // (H * W), (s // W) % H, s % W
    expect = f * (til.nh * til.nw) + (h // 8) * til.nw + (w // 16)
    assert torch.equal(til.tok_tile, expect)


def test_real_density_equals_token_pair_count():
    til = Tiling(*GRID, (2, 4, 16))
    m = _random_mask(til, 3)
    t = til.tok_tile
    pairs = sum(int(m[h][t][:, t].sum()) for h in range(3))
    assert abs(real_density(m, til) - pairs / (3 * til.S ** 2)) < 1e-12
    rd = real_density_heads(m, pair_weights(til), til.S)
    assert abs(float(rd.mean()) - real_density(m, til)) < 1e-12


def test_exec_density_pads_every_block_to_the_kernel_block():
    til = Tiling(*GRID, (1, 16, 8))
    m = _random_mask(til, 2, seed=1)
    cq, ck = KERNEL_BLOCK
    executed = 0
    for h in range(2):
        for g in range(til.nt):
            keys = int(til.n_real[m[h, g]].sum())
            executed += cq * ck * math.ceil(keys / ck)
    assert abs(exec_density(m, til) - executed / (2 * til.S ** 2)) < 1e-12
    xh = exec_density_heads(m, til.n_real, til.S)
    assert abs(float(xh.mean()) - exec_density(m, til)) < 1e-12


def test_all_blocks_exceed_one_only_by_padding():
    til = Tiling(*GRID, (1, 8, 16))
    m = torch.ones(1, til.nt, til.nt, dtype=torch.bool)
    assert abs(real_density(m, til) - 1.0) < 1e-12
    assert exec_density(m, til) >= 1.0
    assert pseudo_density(m, til) == 0.0


def test_spatial_plan_keeps_the_order_of_its_own_tiling(monkeypatch):
    from decoupattn.kernels import block_sparse as BS

    seen = []

    def plan(cq, ck, allowed, side=None, **kw):
        seen.append((cq, ck, side if side is not None else BS.tiling_side(cq, ck, allowed.shape[-1])))

    monkeypatch.setattr(BS, "StaticPlan", plan)
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda self: 0)
    for shape in ((1, 16, 8), (2, 8, 8), (2, 4, 16)):
        til = Tiling(3, 20, 40, shape)
        for H in (2, 3, 2):
            BS.SpatialPlan(own_mask(til)[None].expand(H, -1, -1), til)
            cq, ck, (q_idx, k_idx, k_off, cnt) = seen[-1]
            assert q_idx.shape == k_idx.shape == (H, til.S_pad) and cnt.shape == (H, til.nt + 1)
            assert torch.equal(cq[-1], til.cq) and torch.equal(ck[-1], til.ck)
            assert torch.equal(q_idx[-1].long(), torch.sort(til.cq.long(), stable=True).indices)
            assert torch.equal(k_idx[-1].long(), torch.sort(til.ck.long(), stable=True).indices)
            assert torch.equal(cnt[-1, :til.nt], til.n_real.long())
            assert torch.equal(k_off[-1, 1:].long(), torch.cumsum(cnt[-1], 0))
        assert seen[-1][2] is seen[-3][2]
        del til
