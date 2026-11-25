from __future__ import annotations

import copy

import numpy as np

from ..constants import ESTIMATORS
from ..plan import Plan
from . import sensitivity
from .build import build_plan, choose
from .units import Units

RESOLVED = ("U", "D", "Centroid", "Tied", "Frozen", "Rebuild")
COPIED = ("Tied_dagger",)
SCHEDULE = {"Frozen": False, "Rebuild": "rebuild"}


def est_of_plan(units: Units, plan: Plan) -> np.ndarray:
    est = np.asarray(plan.estimator).astype(np.int64)
    L = units.n_layers
    if units.layout == "steps":
        return est[1:4, 1:].reshape(-1)
    si = units.step_row
    first = [int(np.nonzero(si[1:] == r)[0][0]) + 1 for r in range(units.n_rows)]
    return est[first][:, 1:L].reshape(-1)


def copy_plan(plan: Plan, group: str | None = None) -> Plan:
    p = copy.copy(plan)
    p.extra = dict(plan.extra)
    if group:
        p.extra["abl_group"] = np.array(group)
    return p


def solve_variant(name: str, units: Units, E: dict, X: np.ndarray, sens: dict, D: np.ndarray, block_shape,
                  reassign, main_plan: Plan | None = None, extra: dict | None = None, E_main: dict | None = None):
    extra = dict(extra or {})
    J = None if name == "U" else sensitivity.load(units, sens, (E_main or E)[ESTIMATORS[0]])
    if name == "U":
        ch, est, info = choose(units, E, X, np.ones(len(units)))
    elif name == "D":
        assert all(np.array_equal(E[c], E[ESTIMATORS[0]]) for c in ESTIMATORS), "variant D needs one error column"
        ch, est, info = choose(units, E, X, J)
        est = est_of_plan(units, main_plan)
    elif name == "Centroid":
        ch, est, info = choose(units, E, X, J, estimator=ESTIMATORS[1])
    elif name == "Tied":
        ch, est, info = choose(units, E, X, J)
        extra["abl_group"] = np.array("tile")
    elif name in SCHEDULE:
        ch, est, info = choose(units, E, X, J)
        reassign = SCHEDULE[name]
    else:
        raise ValueError(f"{name}: not a re-solved variant {RESOLVED}")
    return build_plan(units, D, block_shape, ch, est, reassign, extra), ch, est, info


def derived_variant(name: str, main_plan: Plan) -> Plan:
    if name == "Tied_dagger":
        return copy_plan(main_plan, group="tile")
    raise ValueError(f"{name}: not a derived variant {COPIED}")
