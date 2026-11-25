import itertools

import numpy as np
import pytest

from decoupattn.constants import ESTIMATORS
from decoupattn.planner.allocate import allocate, density, lagrangian_plan, price

FULL, CENT = ESTIMATORS


def instance(seed, n_calls=2, heads=2, n_opt=3, weighted=True):
    rng = np.random.default_rng(seed)
    n = n_calls * heads
    g = np.repeat(np.arange(n_calls), heads)
    X = np.concatenate([np.ones((n, 1)), rng.uniform(0.02, 0.6, (n, n_opt - 1))], 1)
    E = {c: np.concatenate([np.zeros((n, 1)), rng.uniform(0.0, 1.0, (n, n_opt - 1))], 1) for c in ESTIMATORS}
    w = rng.integers(1, 4, n).astype(float) if weighted else np.ones(n)
    J = rng.uniform(0.2, 2.0, n)
    return price(E, J, w), X, g, n_calls, w


def priced(Ew, ch, est, g):
    ar = np.arange(len(ch))
    return float(np.where(est[g] == 1, Ew[FULL][ar, ch], Ew[CENT][ar, ch]).sum())


def all_plans(n, n_opt, G):
    for ch in itertools.product(range(n_opt), repeat=n):
        for est in itertools.product((0, 1), repeat=G):
            yield np.array(ch), np.array(est)


@pytest.mark.parametrize("seed", range(6))
def test_lagrangian_minimiser_is_best_at_its_density(seed):
    Ew, X, g, G, w = instance(seed)
    plans = list(all_plans(X.shape[0], X.shape[1], G))
    for mu in (0.0, 0.05, 0.3, 1.0, 5.0):
        ch, est = lagrangian_plan(Ew, X, g, G, w, mu)
        d0, p0 = density(X, w, ch), priced(Ew, ch, est, g)
        for c2, e2 in plans:
            if density(X, w, c2) <= d0 + 1e-12:
                assert priced(Ew, c2, e2, g) >= p0 - 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_lagrangian_minimises_L_mu(seed):
    Ew, X, g, G, w = instance(seed)
    for mu in (0.0, 0.2, 2.0):
        ch, est = lagrangian_plan(Ew, X, g, G, w, mu)
        best = min(priced(Ew, c, e, g) + mu * (w * X[np.arange(len(c)), c]).sum()
                   for c, e in all_plans(X.shape[0], X.shape[1], G))
        got = priced(Ew, ch, est, g) + mu * (w * X[np.arange(len(ch)), ch]).sum()
        assert got == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_density_monotone_in_mu(seed):
    Ew, X, g, G, w = instance(seed, n_calls=4, heads=3, n_opt=5)
    ds = [density(X, w, lagrangian_plan(Ew, X, g, G, w, mu)[0]) for mu in np.linspace(0, 3, 40)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(ds))


def test_per_call_estimator_choice():
    Ew, X, g, G, w = instance(3, n_calls=5, heads=4, n_opt=6)
    mu = 0.4
    ch, est = lagrangian_plan(Ew, X, g, G, w, mu)
    for c in range(G):
        m = g == c
        tot = {e: (Ew[e][m] + mu * w[m, None] * X[m]).min(1).sum() for e in ESTIMATORS}
        want = 1 if tot[FULL] <= tot[CENT] else 0
        assert est[c] == want
        f = Ew[FULL if want else CENT][m] + mu * w[m, None] * X[m]
        assert np.array_equal(ch[m], f.argmin(1))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("target", (0.2, 0.35))
def test_allocate_fits_target_and_fill_helps(seed, target):
    Ew, X, g, G, w = instance(seed, n_calls=6, heads=4, n_opt=8)
    ch, est, info = allocate(Ew, X, g, G, w, target=target)
    assert density(X, w, ch) <= target + 1e-12
    ch0, est0 = lagrangian_plan(Ew, X, g, G, w, info["mu"])
    assert np.array_equal(est, est0)
    assert density(X, w, ch0) <= target + 1e-12
    assert priced(Ew, ch, est, g) <= priced(Ew, ch0, est0, g) + 1e-12
    if info["fill_moves"]:
        assert density(X, w, ch) >= density(X, w, ch0)


def test_greedy_fill_leaves_no_affordable_improvement():
    Ew, X, g, G, w = instance(11, n_calls=6, heads=4, n_opt=8)
    target = 0.25
    ch, est, _ = allocate(Ew, X, g, G, w, target=target)
    ar = np.arange(len(ch))
    Eu = np.where(est[g][:, None] == 1, Ew[FULL], Ew[CENT])
    slack = target * w.sum() - (w * X[ar, ch]).sum()
    dE = Eu - Eu[ar, ch][:, None]
    dX = w[:, None] * (X - X[ar, ch][:, None])
    assert not ((dE < 0) & (dX > 0) & (dX <= slack)).any()
