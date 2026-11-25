import numpy as np
import pytest
import torch

from decoupattn.clustering import build_clusters, reassign_labels
from decoupattn.compensation import CENTROID, FULL
from decoupattn.context import CallContext
from decoupattn.kernels.reference import decoup_ref, dense_ref
from decoupattn.layout import Tiling, exec_density, own_mask
from decoupattn.plan import KIND_COMPONLY, KIND_DENSE, KIND_STATIC, Plan
from decoupattn.runtime import DecoupRuntime

GRID, SHAPE, T, L, H, D, C = (2, 10, 20), (1, 8, 16), 3, 3, 3, 16, 16
Dn, St, Co = KIND_DENSE, KIND_STATIC, KIND_COMPONLY


def make_plan(reassign="reassign"):
    til = Tiling(*GRID, SHAPE)
    g = torch.Generator().manual_seed(0)
    eye = own_mask(til)
    mask = (torch.rand(T, L, H, til.nt, til.nt, generator=g) < 0.4) | eye
    kind = np.full((T, L, H), Dn, np.int8)
    kind[1, 1] = [St, Dn, Co]
    kind[2, 1] = [St, St, St]
    kind[2, 2] = [Dn, Co, Dn]
    mask[torch.from_numpy(kind == Co)] = eye
    mask[torch.from_numpy(kind == Dn)] = eye
    mask[0] = True
    mask[:, 0] = True
    est = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int8)
    return Plan(mask=mask.numpy(), step_row=np.arange(T), block_shape=SHAPE, kind=kind,
                k_per_head=np.ones((T, L, H), np.int16), estimator=est, reassign=reassign), til


def qkv(seed):
    g = torch.Generator().manual_seed(seed)
    S = GRID[0] * GRID[1] * GRID[2]
    return [torch.randn(1, S, H, D, generator=g) for _ in range(3)]


def run(rt, branch="cond"):
    outs = {}
    for s in range(T):
        for l in range(L):
            q, k, v = qkv(100 * s + l)
            outs[(s, l)] = (rt(q, k, v, CallContext(s, l, branch)), (q, k, v))
    return outs


def expected(rt_plan, til, outs, schedule):
    exp, cen, lab = {}, {}, {}
    for s in range(T):
        for l in range(L):
            q, k, v = outs[(s, l)][1]
            any_sparse = (rt_plan.kind[1:, l] != Dn).any()
            if l >= 1 and any_sparse:
                if s == 1:
                    lab[l], cen[l] = build_clusters(k, til, C, 10)
                elif s > 1 and schedule == "reassign":
                    lab[l] = reassign_labels(k, cen[l], til, C)
            if s == 0 or l == 0 or (rt_plan.kind[s, l] == Dn).all():
                exp[(s, l)] = None
                continue
            sh = np.nonzero(rt_plan.kind[s, l] != Dn)[0]
            dh = np.nonzero(rt_plan.kind[s, l] == Dn)[0]
            est = FULL if rt_plan.estimator[s, l] == 1 else CENTROID
            m = torch.from_numpy(rt_plan.mask[s, l])
            o = torch.empty(til.S, H, D)
            o[:, sh] = decoup_ref(q, k, v, m, lab[l].lab, C, til, sh.tolist(), estimator=est)
            if len(dh):
                o[:, dh] = dense_ref(q, k, v, dh.tolist())
            exp[(s, l)] = o.reshape(1, til.S, H * D)
    return exp


@pytest.mark.parametrize("schedule", ["reassign", None])
def test_tiny_plan_reference_backend(schedule):
    plan, til = make_plan(schedule)
    rt = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C)
    outs = run(rt)
    exp = expected(plan, til, outs, "reassign" if schedule else "frozen")
    for key, (o, _) in outs.items():
        if exp[key] is None:
            assert o is None, key
        else:
            assert torch.allclose(o, exp[key], atol=1e-5), key
    sm = rt.summary()
    assert sm["sparse_calls"] == 3 and rt.stats["cluster_builds"] == (4 if schedule else 2)
    x = exec_density(torch.from_numpy(plan.mask[1, 1][[0, 2]]), til)
    assert abs(rt.stats["exec_density"][0] - (2 * x + 1) / 3) < 1e-12


def test_all_blocks_exact_is_dense():
    plan, til = make_plan()
    plan.mask[:] = True
    rt = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C)
    for o, (q, k, v) in run(rt).values():
        if o is not None:
            assert torch.allclose(o, dense_ref(q, k, v, range(H)).reshape(1, til.S, -1), atol=1e-5)


def test_lse_and_overwrite_modes_agree():
    plan, _ = make_plan()
    a = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C, lse=True)
    b = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C, lse=True, mixed="overwrite",
                      cluster_all_layers=True)
    oa, ob = run(a), run(b)
    for key in oa:
        ra, rb = oa[key][0], ob[key][0]
        if ra is None:
            assert rb is None
            continue
        assert torch.allclose(ra[0], rb[0], atol=1e-5) and torch.allclose(ra[1], rb[1], atol=1e-5)
        q, k, v = oa[key][1]
        dh = np.nonzero(plan.kind[key] == Dn)[0]
        if len(dh):
            _, lse = dense_ref(q, k, v, dh.tolist(), return_lse=True)
            assert torch.allclose(ra[1][dh], lse, atol=1e-5)


def test_ablation_switches():
    plan, til = make_plan()
    q, k, v = qkv(7)
    cen = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C, estimator=CENTROID)
    full = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C)
    for rt in (cen, full):
        rt(q, k, v, CallContext(1, 2))
    c1 = cen(q, k, v, CallContext(2, 2))
    lab = cen.cache.clusters(2).lab
    ref = decoup_ref(q, k, v, torch.from_numpy(plan.mask[2, 2]), lab, C, til, [1], estimator=CENTROID)
    assert torch.allclose(c1.view(til.S, H, D)[:, 1], ref[:, 0], atol=1e-5)
    tied = DecoupRuntime(plan, GRID, backend="reference", device="cpu", tied=True)
    tied(q, k, v, CallContext(1, 2))
    assert torch.equal(tied.cache.clusters(2).lab[0], til.tok_tile)
    frozen = DecoupRuntime(plan, GRID, backend="reference", device="cpu", clusters=C, schedule="frozen")
    frozen(q, k, v, CallContext(1, 2))
    first = frozen.cache.clusters(2)
    frozen(*qkv(8), CallContext(2, 2))
    assert frozen.cache.clusters(2) is first
