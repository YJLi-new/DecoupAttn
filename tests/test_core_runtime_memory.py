import numpy as np
import pytest
import torch

from decoupattn.clustering import tile_labels
from decoupattn.context import CallContext
from decoupattn.layout import Tiling, own_mask
from decoupattn.plan import KIND_COMPONLY, KIND_DENSE, KIND_STATIC, Plan
from decoupattn.runtime import DecoupRuntime

GRID, SHAPE, T, L, H, D, C = (2, 10, 20), (1, 8, 16), 3, 3, 6, 16, 16


def make_plan():
    til = Tiling(*GRID, SHAPE)
    g = torch.Generator().manual_seed(1)
    eye = own_mask(til)
    mask = (torch.rand(T, L, H, til.nt, til.nt, generator=g) < 0.4) | eye
    kind = np.full((T, L, H), KIND_STATIC, np.int8)
    kind[0] = KIND_DENSE
    kind[:, 0] = KIND_DENSE
    kind[1, 1] = [KIND_STATIC, KIND_DENSE, KIND_COMPONLY, KIND_DENSE, KIND_STATIC, KIND_DENSE]
    kind[2, 2, 1:] = KIND_DENSE
    mask[torch.from_numpy(kind == KIND_COMPONLY)] = eye
    mask[torch.from_numpy(kind == KIND_DENSE)] = eye
    mask[0] = True
    mask[:, 0] = True
    est = np.ones((T, L), np.int8)
    est[2, 1] = 0
    return Plan(mask=mask.numpy(), step_row=np.arange(T), block_shape=SHAPE, kind=kind,
                k_per_head=np.ones((T, L, H), np.int16), estimator=est, reassign="reassign")


def generate(rt, branches=("cond", "uncond")):
    outs = {}
    for s in range(T):
        for l in range(L):
            for b in branches:
                g = torch.Generator().manual_seed(1000 * s + 10 * l + len(b))
                q, k, v = (torch.randn(1, GRID[0] * GRID[1] * GRID[2], H, D, generator=g) for _ in range(3))
                outs[(s, l, b)] = rt(q, k, v, CallContext(s, l, b))
    return outs


MEMORY = {
    "head_cap": {"head_cap": 2},
    "head_cap_1": {"head_cap": 1},
    "chunk_gib": {"head_chunk_gib": 1e-9},
    "dense_head_cap": {"dense_head_cap": 1},
    "label_free": {"label_free": True, "lean": True},
    "free_plans": {"free_plans_at_end": True},
    "hy": {"head_cap": 2, "dense_head_cap": 2, "head_chunk_gib": 1e-9, "label_free": True, "lean": True,
           "free_plans_at_end": True, "keep_subsets": False},
}


@pytest.mark.parametrize("lse", [True, False])
@pytest.mark.parametrize("name", list(MEMORY))
def test_memory_options_keep_outputs(name, lse):
    plan = make_plan()
    kw = {"backend": "reference", "device": "cpu", "clusters": C, "lse": lse}
    base = DecoupRuntime(plan, GRID, **kw)
    mem = DecoupRuntime(plan, GRID, **kw, **MEMORY[name])
    a, b = generate(base), generate(mem)
    for key in a:
        if a[key] is None:
            assert b[key] is None, key
            continue
        for x, y in zip(a[key] if lse else (a[key],), b[key] if lse else (b[key],)):
            assert torch.allclose(x, y, atol=1e-6, rtol=0), (name, key)
    assert base.stats["exec_density"] == mem.stats["exec_density"]
    if "head_cap" in MEMORY[name] or "head_chunk_gib" in MEMORY[name]:
        assert mem.stats.get("chunked_calls", 0) > 0


def test_chunk_count():
    plan = make_plan()
    til = Tiling(*GRID, SHAPE)
    m = torch.ones(5, til.nt, til.nt, dtype=torch.bool)
    assert DecoupRuntime(plan, GRID, backend="reference", device="cpu").n_chunks(m) == 1
    assert DecoupRuntime(plan, GRID, backend="reference", device="cpu", head_cap=2).n_chunks(m) == 3
    tot_gib = 4 * 5 * til.nt * til.S / 2 ** 30
    rt = DecoupRuntime(plan, GRID, backend="reference", device="cpu", head_chunk_gib=tot_gib / 1.5)
    assert rt.n_chunks(m) == 2


def test_tied_groups_merge_blocks_beyond_c():
    til = Tiling(*GRID, SHAPE)
    lab = tile_labels(til, 2, C=3)
    assert int(lab.max()) == 2 and torch.equal(lab[0], til.tok_tile * 3 // til.nt)
    assert torch.equal(tile_labels(til, 2, C=880)[1], til.tok_tile)


@pytest.mark.parametrize("model", ["hy", "nano", "cp1"])
def test_timing_builds_no_plan_inside_dit_calls(model):
    from decoupattn.config import load_config
    from decoupattn.eval.timing import generate_timed
    from decoupattn.models.base import ModelAdapter
    from decoupattn.runtime import runtime_for

    cfg = load_config(model, resolve=False)
    row_mode = cfg["runtime"].get("keep_plans") == "row"
    inside, built, rows = [False], [], []

    class Adapter:
        latent_grid, head_dim, branches, needs_lse, dense = GRID, D, tuple(cfg["sampler"]["branches"]), False, None
        dense_heads, dit_timer = ModelAdapter.dense_heads, ModelAdapter.dit_timer

        def __init__(self):
            self.cfg, self.timer, self.hook, self.ctx = cfg, None, None, CallContext()

        def generate(self, clip, hook, seed):
            self.hook = hook
            for s in range(T):
                for b in self.branches:
                    self.ctx = CallContext(s, 0, b)
                    with self.dit_timer(f"s{s}_{b}"):
                        inside[0] = True
                        for l in range(L):
                            g = torch.Generator().manual_seed(100 * s + l)
                            q, k, v = (torch.randn(1, GRID[0] * GRID[1] * GRID[2], H, D, generator=g) for _ in range(3))
                            hook(q, k, v, CallContext(s, l, b))
                            rows.append({k[0] for k in [*rt.plans, *rt.chunk_plans]})
                        inside[0] = False

    ad = Adapter()
    chunks = {"head_cap": 2} if cfg["runtime"].get("head_cap") else {}
    rt = runtime_for(ad, make_plan(), backend="reference", device="cpu", clusters=C, **chunks)
    make = rt._make_plan
    rt._make_plan = lambda m: built.append(inside[0]) or make(m)
    runs = [generate_timed(ad, rt, "c0") for _ in range(2)]
    assert built and not any(built)
    if chunks:
        assert rt.stats.get("chunked_calls", 0) > 0
    if row_mode:
        assert max(map(len, rows)) == 1 and len(set().union(*rows)) == T - 1
        assert all(r["excluded_ms"] > 0 for r in runs)
    else:
        assert all(r["excluded_ms"] == 0 for r in runs)
