from __future__ import annotations

import numpy as np

from ..constants import ESTIMATORS, TARGET_DENSITY

FULL, CENTROID = ESTIMATORS


def price(E: dict, J: np.ndarray, w: np.ndarray) -> dict:
    return {c: E[c] * (J * w)[:, None] for c in ESTIMATORS}


def lagrangian_plan(Ew: dict, X: np.ndarray, g: np.ndarray, G: int, w: np.ndarray, mu: float):
    ar = np.arange(X.shape[0])
    best = {}
    tot = []
    for c in ESTIMATORS:
        f = Ew[c] + mu * w[:, None] * X
        j = f.argmin(1)
        best[c] = j
        tot.append(np.bincount(g, f[ar, j], minlength=G))
    tot = np.array(tot)
    est = (tot[0] <= tot[1]).astype(np.int64)
    choice = np.where(est[g] == 1, best[FULL], best[CENTROID])
    return choice, est


def density(X: np.ndarray, w: np.ndarray, choice: np.ndarray) -> float:
    return float((w * X[np.arange(len(choice)), choice]).sum() / w.sum())


def allocate(Ew: dict, X: np.ndarray, g: np.ndarray, G: int, w: np.ndarray, target: float = TARGET_DENSITY,
             iters: int = 200):
    ar = np.arange(X.shape[0])
    W = w.sum()

    def dens(ch):
        return float((w * X[ar, ch]).sum() / W)

    lo, hi = 0.0, 1.0
    while dens(lagrangian_plan(Ew, X, g, G, w, hi)[0]) > target:
        hi *= 4
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if dens(lagrangian_plan(Ew, X, g, G, w, mid)[0]) > target:
            lo = mid
        else:
            hi = mid
    ch, est = lagrangian_plan(Ew, X, g, G, w, hi)
    Eu = np.where(est[g][:, None] == 1, Ew[FULL], Ew[CENTROID])
    Xw = w[:, None] * X
    slack = target * W - Xw[ar, ch].sum()
    moves = 0
    while True:
        dE = Eu - Eu[ar, ch][:, None]
        dX = Xw - Xw[ar, ch][:, None]
        ok = (dE < 0) & (dX > 0) & (dX <= slack)
        if not ok.any():
            break
        sc = np.where(ok, -dE / np.where(dX > 0, dX, 1.0), -np.inf)
        u, o = np.unravel_index(np.argmax(sc), sc.shape)
        slack -= dX[u, o]
        ch[u] = o
        moves += 1
    return ch, est, {"mu": hi, "fill_moves": moves}
