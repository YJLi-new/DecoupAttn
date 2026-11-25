import numpy as np
import pytest
import torch

from decoupattn.compensation import CENTROID, FULL
from decoupattn.context import CallContext
from decoupattn.kernels.reference import decoup_ref, dense_ref
from decoupattn.layout import Tiling, own_mask
from decoupattn.plan import KIND_COMPONLY, KIND_DENSE, KIND_STATIC, Plan
from decoupattn.runtime import DecoupRuntime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

GRID, SHAPE, T, L, H, D = (3, 20, 40), (1, 8, 16), 3, 3, 4, 128
Dn, St, Co = KIND_DENSE, KIND_STATIC, KIND_COMPONLY
TOL = 0.05
MODES = {"fast": {}, "exact": {"fast": False}, "lse": {"lse": True}}


def make_plan(reassign="reassign"):
    til = Tiling(*GRID, SHAPE)
    g = torch.Generator().manual_seed(0)
    eye = own_mask(til)
    mask = (torch.rand(T, L, H, til.nt, til.nt, generator=g) < 0.25) | eye
    kind = np.full((T, L, H), Dn, np.int8)
    kind[1, 1] = [St, Dn, Co, St]
    kind[2, 1] = [St, St, St, St]
    kind[2, 2] = [Dn, Co, Dn, Dn]
    mask[torch.from_numpy(kind == Co)] = eye
    mask[torch.from_numpy(kind == Dn)] = eye
    mask[0] = True
    mask[:, 0] = True
    est = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int8)
    return Plan(mask=mask.numpy(), step_row=np.arange(T), block_shape=SHAPE, kind=kind,
                k_per_head=np.ones((T, L, H), np.int16), estimator=est, reassign=reassign)


def qkv(seed):
    g = torch.Generator().manual_seed(seed)
    S = GRID[0] * GRID[1] * GRID[2]
    return [torch.randn(1, S, H, D, generator=g).to("cuda", torch.bfloat16) for _ in range(3)]


def rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def reference(rt, plan, q, k, v, s, l):
    til, S = rt.til, q.shape[1]
    sh, dh = np.nonzero(plan.kind[s, l] != Dn)[0], np.nonzero(plan.kind[s, l] == Dn)[0]
    o, ls = torch.empty(S, H, D, device="cuda"), torch.empty(H, S, device="cuda")
    est = FULL if plan.estimator[s, l] == 1 else CENTROID
    m = torch.from_numpy(plan.mask[s, l]).cuda()
    o[:, sh], ls[sh] = decoup_ref(q, k, v, m, rt.cache.clusters(l).lab, rt.cache.C, til, sh.tolist(), estimator=est,
                                  return_lse=True)
    if len(dh):
        o[:, dh], ls[dh] = dense_ref(q, k, v, dh.tolist(), return_lse=True)
    return o.reshape(1, S, H * D), ls


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("schedule", ["reassign", None])
def test_kernels_match_the_reference(mode, schedule):
    plan = make_plan(schedule)
    rt = DecoupRuntime(plan, GRID, backend="kernels", device="cuda", **MODES[mode])
    rt.build_all()
    sparse = 0
    for s in range(T):
        for l in range(L):
            q, k, v = qkv(100 * s + l)
            r = rt(q, k, v, CallContext(s, l, "cond"))
            if s == 0 or l == 0 or (plan.kind[s, l] == Dn).all():
                assert r is None, (s, l)
                continue
            o, ls = r if mode == "lse" else (r, None)
            exp_o, exp_ls = reference(rt, plan, q, k, v, s, l)
            err = rel(o, exp_o)
            print(f"{mode} schedule={schedule} call ({s}, {l}): relative error {err:.4f}")
            assert o.shape == exp_o.shape and torch.isfinite(o.float()).all() and err < TOL, (s, l, err)
            if ls is not None:
                assert float((ls - exp_ls).abs().max()) < TOL, (s, l)
            sparse += 1
    assert sparse == 3 and rt.summary()["sparse_calls"] == 3


@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_all_blocks_exact_is_dense(mode):
    plan = make_plan()
    plan.mask[:] = True
    rt = DecoupRuntime(plan, GRID, backend="kernels", device="cuda", **MODES[mode])
    for s in range(1, T):
        q, k, v = qkv(7 * s)
        o = rt(q, k, v, CallContext(s, 1, "cond"))
        dense = dense_ref(q, k, v, range(H)).reshape(1, q.shape[1], H * D)
        err = rel(o, dense)
        print(f"{mode} all blocks exact, step {s}: relative error to dense {err:.4f}")
        assert err < TOL, (s, err)


def test_tied_and_frozen_variants_run():
    plan = make_plan()
    for kw in ({"tied": True}, {"schedule": "frozen"}, {"estimator": CENTROID}):
        rt = DecoupRuntime(plan, GRID, backend="kernels", device="cuda", **kw)
        for s in range(1, T):
            q, k, v = qkv(11 * s)
            o = rt(q, k, v, CallContext(s, 1, "cond"))
            assert o.shape == (1, q.shape[1], H * D) and torch.isfinite(o.float()).all(), (kw, s)
