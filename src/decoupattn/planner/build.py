from __future__ import annotations

import numpy as np
import torch

from ..constants import ESTIMATORS, I_COMPONLY, I_DENSE, LEVELS, OPTIONS, TARGET_DENSITY
from ..plan import KIND_COMPONLY, KIND_DENSE, KIND_STATIC, Plan
from .allocate import allocate, density, price
from .dtable import rank_of
from .units import Units, k_blocks


def choose(units: Units, E: dict, X: np.ndarray, J: np.ndarray, target: float = TARGET_DENSITY,
           estimator: str | None = None):
    w = units.weights
    Es = {c: E[estimator] for c in ESTIMATORS} if estimator else E
    ch, est, info = allocate(price(Es, J, w), X, units.calls, units.n_calls, w, target=target)
    if estimator:
        est = np.full(units.n_calls, int(estimator == ESTIMATORS[0]), np.int64)
    return ch, est, info


def build_plan(units: Units, D: np.ndarray, block_shape, choice, est_call, reassign=None, extra: dict | None = None) -> Plan:
    L, H, R = units.n_layers, units.n_heads, units.n_rows
    NT = D.shape[-1]
    own = torch.eye(NT, dtype=torch.bool)
    steps = units.layout == "steps"
    P = 4 if steps else R
    off = 1 if steps else 0
    kind_r = np.full((P, L, H), KIND_DENSE, np.int8)
    K_r = np.ones((P, L, H), np.int64)
    for (r, l, h), j in zip(units.units, choice):
        if j == I_DENSE:
            kind_r[r + off, l, h] = KIND_DENSE
        elif j == I_COMPONLY:
            kind_r[r + off, l, h] = KIND_COMPONLY
        else:
            kind_r[r + off, l, h] = KIND_STATIC
            K_r[r + off, l, h] = k_blocks(LEVELS[j - 1], NT)
    mask = np.ones((P, L, H, NT, NT), bool)
    for r in range(R):
        for l in range(1, L):
            rk = rank_of(torch.from_numpy(np.asarray(D[r, l])), own)
            mask[r + off, l] = (rk < torch.from_numpy(K_r[r + off, l])[:, None, None]).numpy()
    est_r = np.ones((R, L), np.int8)
    est_r[:, 1:] = np.asarray(est_call, np.int8).reshape(R, L - 1)
    if steps:
        si = np.arange(4)
        kind, K = kind_r, K_r
        est = np.ones((4, L), np.int8)
        est[1:] = est_r
        rows = row_weight = None
    else:
        si = units.step_row
        kind, K = kind_r[si].copy(), K_r[si].copy()
        kind[0], K[0] = KIND_DENSE, 1
        est = est_r[si].copy()
        est[0] = 1
        rows, row_weight = np.array(units.rows), units.row_weight
    return Plan(mask=mask, step_row=si, block_shape=tuple(int(x) for x in block_shape), kind=kind,
                k_per_head=K.astype(np.int16), estimator=est, query_tiling="frame", choice=np.asarray(choice, np.int16),
                rows=rows, row_weight=row_weight, reassign=("reassign" if reassign is True else reassign) or None,
                dynamic_blocks=np.zeros(kind.shape, np.int16), extra=dict(extra or {}))


def summary(units: Units, E: dict, X: np.ndarray, RD: np.ndarray, J: np.ndarray, choice, est_call) -> dict:
    w, g = units.weights, units.calls
    ar = np.arange(len(choice))
    isd = choice == I_DENSE
    cen = (~isd) & (est_call[g] == 0)
    Eu = np.where(est_call[g][:, None] == 1, E[ESTIMATORS[0]], E[ESTIMATORS[1]])

    def wmean(v, m):
        return float((w[m] * v[m]).sum() / w[m].sum())

    allm = np.ones(len(choice), bool)
    per = {}
    for r, s in enumerate(units.rows):
        m = units.units[:, 0] == r
        per[str(s)] = {"weight": float(w[m][0]), "density": float(X[ar[m], choice[m]].mean()),
                       "pair_density": float(RD[ar[m], choice[m]].mean()), "dense": float(isd[m].mean()),
                       "comp_only": float((choice[m] == I_COMPONLY).mean()), "centroid_units": float(cen[m].mean())}
    return {"density": density(X, w, choice), "pair_density": wmean(RD[ar, choice], allm),
            "share_dense": wmean(isd.astype(float), allm), "share_comp_only": wmean((choice == I_COMPONLY).astype(float), allm),
            "share_centroid_units": wmean(cen.astype(float), allm),
            "centroid_calls": int(((est_call == 0) & (np.bincount(g, ~isd, minlength=units.n_calls) > 0)).sum()),
            "calls": units.n_calls, "per_row": per,
            "options": {OPTIONS[j]: int((choice == j).sum()) for j in range(len(OPTIONS)) if (choice == j).any()},
            "priced_distortion": float((w * J * Eu[ar, choice]).sum())}
