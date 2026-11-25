import math

import pytest
import torch

from decoupattn.compensation import CENTROID, FULL, decoup_estimate, head_stats
from decoupattn.kernels.reference import decoup_ref, dense_ref, spatial_ref
from decoupattn.layout import Tiling, own_mask

GRID, SHAPE, H, D, C = (2, 10, 20), (1, 8, 16), 3, 16, 12


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(0)
    til = Tiling(*GRID, SHAPE)
    S = til.S
    q, k, v = (torch.randn(1, S, H, D, generator=g) * s for s in (1.0, 1.0, 1.0))
    lab = torch.randint(0, C, (H, S), generator=g)
    m = (torch.rand(H, til.nt, til.nt, generator=g) < 0.4) | own_mask(til)[None]
    return til, q, k, v, lab, m


@pytest.mark.parametrize("est", [FULL, CENTROID])
def test_all_blocks_exact_is_dense(data, est):
    til, q, k, v, lab, _ = data
    m = torch.ones(H, til.nt, til.nt, dtype=torch.bool)
    ref = dense_ref(q, k, v, range(H))
    out, lse = decoup_ref(q, k, v, m, lab, C, til, range(H), estimator=est, return_lse=True)
    assert torch.allclose(out, ref, atol=1e-5)
    _, lse_d = dense_ref(q, k, v, range(H), return_lse=True)
    assert torch.allclose(lse, lse_d, atol=1e-5)
    assert torch.allclose(spatial_ref(q, k, v, m, til, range(H)), ref, atol=1e-5)


def _residual_form(q, k, v, m, lab, til, full):
    S = til.S
    t = til.tok_tile
    ys, yd, err = [], [], []
    for h in range(H):
        qh, kh, vh = (x[0, :, h].double() for x in (q, k, v))
        lb = lab[h]
        cnt, km, vm, var, hm, _ = (x.double() for x in head_stats(kh.float(), vh.float(), lb, C))
        phi = torch.exp(qh @ km.T / math.sqrt(D))
        omega = phi * (1 + (qh * qh) @ var.T / (2 * D)) if full else phi
        a = omega[..., None] * vm[None] + (phi[..., None] * (qh @ hm.T / math.sqrt(D))[:, None] if full else 0.0)
        e = torch.exp(qh @ kh.T / math.sqrt(D))
        nbc = torch.zeros(til.nt, C, dtype=torch.float64).index_put_((t, lb), torch.ones(S, dtype=torch.float64),
                                                                    accumulate=True)
        Nb = torch.zeros(S, til.nt, D, dtype=torch.float64).index_add_(1, t, e[..., None] * vh[None])
        Zb = torch.zeros(S, til.nt, dtype=torch.float64).index_add_(1, t, e)
        dN = Nb - torch.einsum("bc,icd->ibd", nbc, a)
        dZ = Zb - omega @ nbc.T
        sel = m[h][t].double()
        N = (cnt[None, :, None] * a).sum(1) + (sel[..., None] * dN).sum(1)
        Z = (cnt[None] * omega).sum(1) + (sel * dZ).sum(1)
        y_hat = N / Z[:, None]
        y = Nb.sum(1) / Zb.sum(1)[:, None]
        r = dN - y[:, None] * dZ[..., None]
        ys.append(y_hat)
        yd.append(y)
        err.append(-((1 - sel)[..., None] * r).sum(1) / Z[:, None])
    return torch.stack(ys, 1), torch.stack(yd, 1), torch.stack(err, 1)


@pytest.mark.parametrize("est", [FULL, CENTROID])
def test_residual_identity_equals_decoupled_estimator(data, est):
    til, q, k, v, lab, m = data
    y_hat, y, err = _residual_form(q, k, v, m, lab, til, est == FULL)
    out = decoup_ref(q, k, v, m, lab, C, til, range(H), estimator=est)
    assert torch.allclose(out.double(), y_hat, atol=2e-5)
    assert torch.allclose(y_hat - y, err, atol=1e-9)


def test_centroid_equals_full_without_spread():
    g = torch.Generator().manual_seed(1)
    til = Tiling(*GRID, SHAPE)
    S = til.S
    lab = torch.randint(0, C, (H, S), generator=g)
    cen = torch.randn(H, C, D, generator=g)
    k = torch.stack([cen[h][lab[h]] for h in range(H)], 1)[None]
    q, v = torch.randn(1, S, H, D, generator=g), torch.randn(1, S, H, D, generator=g)
    m = (torch.rand(H, til.nt, til.nt, generator=g) < 0.3) | own_mask(til)[None]
    full = decoup_ref(q, k, v, m, lab, C, til, range(H), estimator=FULL)
    cent = decoup_ref(q, k, v, m, lab, C, til, range(H), estimator=CENTROID)
    assert torch.allclose(full, cent, atol=1e-5)
    *_, var, hm, _ = head_stats(k[0, :, 0], v[0, :, 0], lab[0], C)
    assert float(var.abs().max()) < 1e-10 and float(hm.abs().max()) < 1e-5


def test_zero_corrections_are_the_centroid_estimator():
    g = torch.Generator().manual_seed(2)
    qh, kh, vh = (torch.randn(40, D, generator=g) for _ in range(3))
    km, vm = torch.randn(C, D, generator=g), torch.randn(C, D, generator=g)
    exact = torch.rand(40, 40, generator=g) < 0.5
    n = torch.randint(0, 4, (40, C), generator=g).float()
    a = decoup_estimate(qh, kh, vh, exact, n, km, vm, torch.zeros(C, D), torch.zeros(D, D))
    b = decoup_estimate(qh, kh, vh, exact, n, km, vm)
    assert torch.allclose(a, b, atol=1e-6)
