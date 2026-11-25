import numpy as np
import pytest
import torch

from decoupattn.constants import I_COMPONLY, I_DENSE, LEVELS, OPTIONS
from decoupattn.plan import KIND_COMPONLY, KIND_DENSE, KIND_STATIC, load_plan
from decoupattn.planner import variants
from decoupattn.planner.build import build_plan, choose
from decoupattn.planner.dtable import rank_of
from decoupattn.planner.units import Units, k_blocks, row_weights, step_rows

ALAYA_ROWS = (1, 2, 4, 7, 11, 16, 22, 29, 37, 45)


def test_step_rows_nearest_ties_to_earlier():
    si = step_rows(50, ALAYA_ROWS)
    rows = np.array(ALAYA_ROWS)
    for s in range(50):
        d = np.abs(rows - s)
        assert si[s] == np.nonzero(d == d.min())[0][0]
    assert si[3] == 1 and si[0] == 0


def test_row_weights_count_sparse_steps():
    w = row_weights(50, ALAYA_ROWS)
    assert w.sum() == 49 and (w >= 1).all()
    u = Units("rows", 50, ALAYA_ROWS, 5, 3, ("cond", "uncond"))
    assert np.array_equal(u.weights, w[u.units[:, 0]])
    assert u.measured_branches == ("cond", "uncond")


def test_steps_layout_units():
    u = Units("steps", 4, (1, 2, 3), 4, 2, ("cond",))
    assert len(u) == 3 * 3 * 2 and (u.weights == 1).all()
    assert np.array_equal(u.step_row, np.arange(4))
    assert u.table_row(2, "cond") == 1 and u.table_row(0, "cond") is None and u.table_row(2, "uncond") is None
    assert np.array_equal(u.unit_steps, u.units[:, 0] + 1)
    assert np.array_equal(np.unique(u.calls), np.arange(u.n_calls))


def random_D(R, L, H, NT, seed=0):
    return np.random.default_rng(seed).random((R, L, H, NT, NT)).astype(np.float32)


def test_rank_diagonal_first_then_descending():
    D = torch.from_numpy(random_D(1, 1, 2, 6)[0, 0])
    own = torch.eye(6, dtype=torch.bool)
    rk = rank_of(D, own)
    for h in range(2):
        for g in range(6):
            assert rk[h, g, g] == 0
            others = [b for b in range(6) if b != g]
            order = sorted(others, key=lambda b: -float(D[h, g, b]))
            assert [int(rk[h, g, b]) for b in order] == list(range(1, 6))


@pytest.mark.parametrize("layout,steps,rows", [("steps", 4, (1, 2, 3)), ("rows", 12, (1, 2, 4, 7))])
def test_build_plan_masks_kinds_estimators(layout, steps, rows):
    L, H, NT = 4, 3, 20
    u = Units(layout, steps, rows, L, H)
    rng = np.random.default_rng(1)
    choice = rng.integers(0, len(OPTIONS), len(u))
    est = rng.integers(0, 2, u.n_calls)
    D = random_D(len(rows), L, H, NT)
    p = build_plan(u, D, (2, 4, 16), choice, est, reassign=True)
    off = 1 if layout == "steps" else 0
    for (r, l, h), j in zip(u.units, choice):
        m = p.mask[r + off, l, h]
        assert m[np.arange(NT), np.arange(NT)].all()
        K = k_blocks(LEVELS[j - 1], NT) if I_DENSE < j < I_COMPONLY else 1
        assert (m.sum(-1) == K).all()
        rk = rank_of(torch.from_numpy(D[r, l]), torch.eye(NT, dtype=torch.bool))[h].numpy()
        assert np.array_equal(m, rk < K)
        steps_of_row = np.nonzero(p.step_row == r + off)[0]
        steps_of_row = steps_of_row[steps_of_row > 0]
        want = KIND_DENSE if j == I_DENSE else KIND_COMPONLY if j == I_COMPONLY else KIND_STATIC
        assert (p.kind[steps_of_row, l, h] == want).all()
        assert (p.estimator[steps_of_row, l] == est[r * (L - 1) + l - 1]).all()
    assert (p.kind[0] == KIND_DENSE).all() and (p.kind[:, 0] == KIND_DENSE).all()
    assert (p.estimator[0] == 1).all() and (p.estimator[:, 0] == 1).all()
    assert p.mask[:, 0].all()
    assert p.reassign == "reassign"
    assert np.array_equal(variants.est_of_plan(u, p), est)


def test_plan_round_trip(tmp_path):
    u = Units("rows", 12, (1, 2, 4, 7), 4, 3)
    rng = np.random.default_rng(2)
    p = build_plan(u, random_D(4, 4, 3, 10), (1, 8, 16), rng.integers(0, 8, len(u)), rng.integers(0, 2, u.n_calls),
                   reassign=False, extra={"v3": np.int8(1), "tiling": np.array("LQ")})
    p.save(str(tmp_path / "plan.npz"))
    q = load_plan(str(tmp_path / "plan.npz"))
    for a in ("mask", "step_row", "kind", "k_per_head", "estimator", "choice", "rows", "row_weight", "dynamic_blocks"):
        assert np.array_equal(getattr(p, a), getattr(q, a)), a
    assert q.block_shape == (1, 8, 16) and q.reassign is None and q.query_tiling == "frame"
    assert int(q.extra["v3"]) == 1 and str(q.extra["tiling"]) == "LQ"


def test_choose_forced_estimator_and_variants():
    u = Units("steps", 4, (1, 2, 3), 3, 2)
    rng = np.random.default_rng(3)
    n = len(u)
    X = np.concatenate([np.ones((n, 1)), rng.uniform(0.03, 0.6, (n, 7))], 1)
    E = {"Full": np.c_[np.zeros(n), rng.random((n, 7))], "Centroid": np.c_[np.zeros(n), rng.random((n, 7)) + 0.5]}
    ch, est, _ = choose(u, E, X, np.ones(n), estimator="Centroid")
    assert (est == 0).all()
    ch2, _, _ = choose(u, {"Full": E["Centroid"], "Centroid": E["Centroid"]}, X, np.ones(n))
    assert np.array_equal(ch, ch2)
    main = build_plan(u, random_D(3, 3, 2, 8), (2, 4, 16), ch, est, reassign=True)
    assert variants.RESOLVED == ("U", "D", "Centroid", "Tied", "Frozen", "Rebuild") and variants.COPIED == ("Tied_dagger",)
    for name, sched in (("Frozen", None), ("Rebuild", "rebuild")):
        pv, chv, _, _ = variants.solve_variant(name, u, E, X, {"kind": "uniform"}, random_D(3, 3, 2, 8), (2, 4, 16), True)
        assert pv.reassign == sched and np.array_equal(chv, choose(u, E, X, np.ones(n))[0])
        with pytest.raises(ValueError):
            variants.derived_variant(name, main)
    td = variants.derived_variant("Tied_dagger", main)
    assert str(td.extra["abl_group"]) == "tile" and "abl_group" not in main.extra
    _, chu, _, _ = variants.solve_variant("U", u, E, X, {"kind": "uniform"}, random_D(3, 3, 2, 8), (2, 4, 16), True)
    assert np.array_equal(chu, choose(u, E, X, np.ones(n))[0])
    one = {"Full": E["Full"], "Centroid": E["Full"]}
    _, _, estd, _ = variants.solve_variant("D", u, one, X, {"kind": "uniform"}, random_D(3, 3, 2, 8), (2, 4, 16), True,
                                           main_plan=main)
    assert np.array_equal(estd, est)


def test_variants_keep_the_sensitivity_of_the_full_method(monkeypatch):
    u = Units("steps", 4, (1, 2, 3), 3, 2)
    rng = np.random.default_rng(4)
    n = len(u)
    X = np.concatenate([np.ones((n, 1)), rng.uniform(0.03, 0.6, (n, 7))], 1)
    E_main = {"Full": np.c_[np.zeros(n), rng.random((n, 7))], "Centroid": np.c_[np.zeros(n), rng.random((n, 7)) + 0.5]}
    raw = {"Full": E_main["Full"] * 3.0, "Centroid": E_main["Full"] * 3.0}
    tied = {"Full": E_main["Full"] * 2.0, "Centroid": E_main["Centroid"]}
    seen = []
    monkeypatch.setattr(variants.sensitivity, "load", lambda units, sens, E_full: seen.append(E_full) or np.ones(n))
    ch, est, _ = choose(u, E_main, X, np.ones(n))
    main = build_plan(u, random_D(3, 3, 2, 8), (2, 4, 16), ch, est, reassign=True)
    for name, E in (("D", raw), ("Tied", tied), ("Frozen", tied), ("Centroid", E_main)):
        variants.solve_variant(name, u, E, X, {"kind": "injection"}, random_D(3, 3, 2, 8), (2, 4, 16), True,
                               main_plan=main, E_main=E_main)
    assert len(seen) == 4 and all(x is E_main["Full"] for x in seen)


@pytest.mark.parametrize("layout,steps,rows,group", [("steps", 4, (1, 2, 3), ("g1_t0", 2, 2, 1, 2)),
                                                      ("rows", 12, (1, 2, 4, 7), ("g0_t2", 1, 5, 4, 5))])
def test_injection_group_plan(layout, steps, rows, group):
    from decoupattn.planner.sensitivity import I_REF, group_plan

    u = Units(layout, steps, rows, 6, 2)
    p = group_plan(u, random_D(len(rows), 6, 2, 20), (2, 4, 16), group, reassign=True)
    _, s0, s1, l0, l1 = group
    for t in range(p.n_steps):
        for l in range(6):
            want = KIND_STATIC if (s0 <= t <= s1 and l0 <= l <= l1) else KIND_DENSE
            assert (p.kind[t, l] == want).all(), (t, l)
    assert (p.estimator == 1).all() and p.reassign == "reassign"
    K = k_blocks(LEVELS[I_REF - 1], 20)
    assert (p.mask[p.step_row[s0], l0].sum(-1) == K).all()


def test_default_block_shape_least_padding():
    from decoupattn.planner.block_shape import best_tile

    assert best_tile(21, 45, 80) == (1, 8, 16) and best_tile(21, 30, 52) == (1, 16, 8)
    assert best_tile(24, 44, 80) == best_tile(16, 44, 80) == (2, 4, 16)
    assert best_tile(48, 23, 40) == (2, 8, 8) and best_tile(24, 30, 53) == (1, 16, 8)


def test_block_shape_clips_fit_the_splits():
    import json

    from decoupattn.config import load_config, repo_path
    from decoupattn.constants import BLOCK_SHAPES
    from decoupattn.models import MODELS
    from decoupattn.planner import block_shape as bs

    assert set(bs.DEFAULT_CANDIDATES) == set(BLOCK_SHAPES) and (1, 4, 32) not in bs.DEFAULT_CANDIDATES
    for key in MODELS:
        cfg = load_config(key, resolve=False)
        if "block_shape" in cfg["plan"]:
            continue
        with open(repo_path(cfg["data"]["splits"])) as f:
            sp = json.load(f)
        nc = cfg["planner"].get("block_shape_clips", bs.BLOCK_SHAPE_CLIPS)
        assert nc in (4, 8) and nc <= len(sp["calib"]) and len(sp["val"]) >= bs.BLOCK_SHAPE_VAL_CLIPS, key

    class Short:
        def clips(self, split):
            return ["a", "b", "c", "d"]

    with pytest.raises(ValueError):
        bs.probe(Short(), 8)
