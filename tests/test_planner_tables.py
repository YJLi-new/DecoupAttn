import json
import math

import numpy as np
import pytest
import torch

from decoupattn.compensation import head_stats
from decoupattn.constants import OPTIONS, SEED
from decoupattn.layout import Tiling
from decoupattn.planner import block_shape as bs
from decoupattn.planner import rules, sensitivity
from decoupattn.planner.distortion import tables_from
from decoupattn.planner.dtable import residual_map
from decoupattn.planner.units import Units


def reference_D(q, k, v, til, lab, C):
    H, D = q.shape[2], q.shape[3]
    out = np.zeros((H, til.nt, til.nt))
    tok = til.tok_tile.numpy()
    for h in range(H):
        kh, vh, qh = k[0, :, h].float(), v[0, :, h].float(), q[0, :, h].double()
        lb = lab[h].long()
        _, km, vm, var, hm, _ = head_stats(kh, vh, lb, C)
        kb, vb, kv = (x.bfloat16().double() for x in (km, vm, var))
        hm = hm.double()
        kh, vh = kh.double(), vh.double()
        s = qh @ kh.T / math.sqrt(D)
        lse = torch.logsumexp(s, 1)
        P = (s - lse[:, None]).exp()
        y = P @ vh
        phi = ((qh @ kb.T) / math.sqrt(D) - lse[:, None]).exp()
        om = phi * (1 + (qh * qh) @ kv.T / (2 * D))
        hq = qh @ hm.T / math.sqrt(D)
        for i in range(qh.shape[0]):
            g = tok[i]
            for b in range(til.nt):
                j = np.nonzero(tok == b)[0]
                c = lb[j]
                Z = P[i, j].sum()
                N = (P[i, j, None] * vh[j]).sum(0)
                Zp = om[i, c].sum()
                Np = (om[i, c, None] * vb[c]).sum(0) + phi[i, c].sum() * hq[i]
                r = (N - Np) - y[i] * (Z - Zp)
                out[h, g, b] += float(r.pow(2).sum() / max(float(y[i].pow(2).sum()), 1e-30))
        out[h] /= np.bincount(tok, minlength=til.nt)[:, None]
        np.fill_diagonal(out[h], 0.0)
    return out


def test_residual_map_matches_definition():
    torch.manual_seed(0)
    til = Tiling(1, 12, 16, (1, 8, 16))
    S, H, D, C = til.S, 2, 16, 7
    q, k, v = (torch.randn(1, S, H, D) * 0.7 for _ in range(3))
    lab = torch.randint(0, C, (H, S))
    got = residual_map(q, k, v, til, lab, C).double().numpy()
    want = reference_D(q, k, v, til, lab, C)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=1e-7)
    assert (np.diagonal(got, axis1=1, axis2=2) == 0).all()


def fake_tables(R, L, H, seed=0, ncomp=2):
    rng = np.random.default_rng(seed)
    z = {"pd2": rng.random((R, L, ncomp, 7, H)), "xd": rng.random((R, L, 7, H)), "rdn": rng.random((R, L, 7, H)),
         "comps": np.array(["Full", "Centroid"][:ncomp])}
    z["pd2"][:, 0] = np.nan
    return z


def test_tables_from_maps_units():
    u = Units("rows", 12, (1, 2, 4), 3, 2)
    Z = [fake_tables(3, 3, 2, s) for s in range(2)]
    E, X, _ = tables_from(u, Z)
    for i, (r, l, h) in enumerate(u.units):
        assert E["Full"][i, 0] == 0 and X[i, 0] == 1
        for j in range(1, 8):
            assert E["Centroid"][i, j] == np.mean([z["pd2"][r, l, 1, j - 1, h] for z in Z])
            assert X[i, j] == np.mean([z["xd"][r, l, j - 1, h] for z in Z])
    E1, _, _ = tables_from(u, [fake_tables(3, 3, 2, 0, ncomp=1)])
    assert np.array_equal(E1["Full"], E1["Centroid"])


def where(u, g):
    return dict(zip(("steps", "layers"), sensitivity._expected(u, sensitivity.default_step_groups(u))[g]))


def test_injection_J(tmp_path):
    u = Units("steps", 4, (1, 2, 3), 7, 2)
    E = np.random.default_rng(0).random((len(u), len(OPTIONS))) + 0.1
    lat = {"space": sensitivity.SPACE}
    recs = [{"group": "dense", "clip": c, "mse": 0.5, **lat, **where(u, "dense")} for c in ("a", "b")]
    mse = {}
    for gi in range(3):
        for ti in range(3):
            mse[(gi, ti)] = 2.0 + gi + 3 * ti
            recs += [{"group": f"g{gi}_t{ti}", "clip": c, "mse": mse[(gi, ti)] + d, **lat, **where(u, f"g{gi}_t{ti}")}
                     for c, d in (("a", 0.1), ("b", -0.1))]
    p = tmp_path / "J.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    J, _ = sensitivity.from_injection(u, str(p), E)
    st, ly = u.unit_steps, u.units[:, 1]
    raw = np.zeros(len(u))
    for gi in range(3):
        for ti, (l0, l1) in enumerate([(1, 2), (3, 4), (5, 6)]):
            m = (st == gi + 1) & (ly >= l0) & (ly <= l1)
            raw[m] = (mse[(gi, ti)] - 0.5) / E[m, sensitivity.I_REF].sum()
    np.testing.assert_allclose(J, raw / raw.mean())
    assert J.mean() == pytest.approx(1.0)
    other = [dict(r, clip="z", mse=50.0 * r["mse"]) for r in recs]
    p.write_text("\n".join(json.dumps(r) for r in recs + other))
    np.testing.assert_allclose(sensitivity.from_injection(u, str(p), E, clips=["a", "b"])[0], J)
    stale = [dict(r, steps=[1, 2]) if r["group"].startswith("g0_") else r for r in recs]
    p.write_text("\n".join(json.dumps(r) for r in stale))
    with pytest.raises(ValueError):
        sensitivity.from_injection(u, str(p), E)
    recs[2]["mse"] = recs[3]["mse"] = 0.1
    p.write_text("\n".join(json.dumps(r) for r in recs))
    with pytest.raises(AssertionError):
        sensitivity.from_injection(u, str(p), E)
    p.write_text("\n".join(json.dumps(r) for r in recs[2:]))
    with pytest.raises(ValueError):
        sensitivity.from_injection(u, str(p), E)
    recs[0].pop("space")
    p.write_text("\n".join(json.dumps(r) for r in recs))
    with pytest.raises(ValueError):
        sensitivity.from_injection(u, str(p), E)


def test_step_groups_match_the_denominator():
    u = Units("rows", 35, (1, 2, 4, 7, 11, 15, 20, 26, 32), 7, 2)
    with pytest.raises(ValueError):
        sensitivity.check_step_groups(u, [(1, 8), (9, 17), (18, 26), (27, 34)])
    aligned = sensitivity.aligned_step_groups(u, [(1, 8), (9, 17), (18, 26), (27, 34)])
    assert aligned == [(1, 9), (10, 17), (18, 29), (30, 34)]
    for sg in (aligned, sensitivity.default_step_groups(u)):
        sensitivity.check_step_groups(u, sg)
        for s0, s1 in sg:
            rows = sensitivity.group_rows(u, s0, s1)
            w = sum(u.row_weight[u.rows.index(r)] for r in rows)
            assert sensitivity.steps_of_rows(u, rows) == list(range(s0, s1 + 1)) and w == s1 - s0 + 1
    assert sum(b - a + 1 for a, b in sensitivity.default_step_groups(u)) == u.steps - 1
    s = Units("steps", 4, (1, 2, 3), 7, 2)
    assert sensitivity.default_step_groups(s) == [(1, 1), (2, 2), (3, 3)]
    sensitivity.check_step_groups(s, [(1, 2), (3, 3)])
    with pytest.raises(ValueError):
        sensitivity.check_step_groups(s, [(4, 5)])


def test_measure_resumes_clip_by_clip(tmp_path, monkeypatch):
    import decoupattn.runtime as rt

    u = Units("steps", 4, (1, 2, 3), 7, 2)
    monkeypatch.setattr(Units, "of", classmethod(lambda cls, a: u))
    monkeypatch.setattr(sensitivity, "group_plan", lambda *a, **k: None)
    monkeypatch.setattr(rt, "runtime_for", lambda adapter, plan, **k: "sparse")

    class Fake:
        def __init__(self):
            self.gen = []

        def clips(self, split):
            return ["a", "b"]

        def generate(self, clip, hook, seed):
            self.gen.append(clip)
            return torch.full((2,), float(len(self.gen)))

    names = ["dense"] + [g[0] for g in sensitivity.groups(u, sensitivity.default_step_groups(u))]
    p = tmp_path / f"J_s{SEED}.jsonl"
    old = [{"group": g, "clip": "a", "space": sensitivity.SPACE, "mse": 1.0, **where(u, g)} for g in names]
    old += [{"group": g, "clip": "b", "space": sensitivity.SPACE, "mse": -1.0, **where(u, g)} for g in names[:2]]
    p.write_text("".join(json.dumps(r) + "\n" for r in old))
    f = Fake()
    sensitivity.measure(f, None, None, None, str(tmp_path), n_clips=2)
    assert f.gen == ["b"] * (len(names) + 1)
    recs = [json.loads(x) for x in p.read_text().splitlines()]
    assert sorted((r["clip"], r["group"]) for r in recs) == sorted((c, g) for c in "ab" for g in names)
    assert all(r["mse"] == 1.0 for r in recs if r["clip"] == "a")
    assert all(r["mse"] >= 0.0 for r in recs if r["clip"] == "b")


def test_injection_file_of_another_measurement_is_measured_again(tmp_path, monkeypatch):
    import decoupattn.runtime as rt

    u = Units("steps", 4, (1, 2, 3), 7, 2)
    monkeypatch.setattr(Units, "of", classmethod(lambda cls, a: u))
    monkeypatch.setattr(sensitivity, "group_plan", lambda *a, **k: None)
    monkeypatch.setattr(rt, "runtime_for", lambda adapter, plan, **k: "sparse")

    class Fake:
        gen = 0

        def clips(self, split):
            return ["a", "b"]

        def generate(self, clip, hook, seed):
            self.gen += 1
            return torch.full((2,), float(self.gen))

    names = ["dense"] + [g[0] for g in sensitivity.groups(u, sensitivity.default_step_groups(u))]
    p = tmp_path / f"J_s{SEED}.jsonl"
    p.write_text("".join(json.dumps({"group": g, "clip": c, "mse": -1.0}) + "\n" for c in "ab" for g in names))
    assert sensitivity.pending(Fake(), str(p), n_clips=2) == ["a", "b"]
    f = Fake()
    sensitivity.measure(f, None, None, None, str(tmp_path), n_clips=2)
    assert f.gen == 2 * (len(names) + 1)
    recs = [json.loads(x) for x in p.read_text().splitlines()]
    assert sorted((r["clip"], r["group"]) for r in recs) == sorted((c, g) for c in "ab" for g in names)
    assert all(r["space"] == sensitivity.SPACE and r["mse"] >= 0.0 for r in recs)
    assert sensitivity.pending(f, str(p), n_clips=2) == []


def test_block_shape_decision():
    cmp = bs.paired([30.0, 31.0, 29.0], [30.2, 31.1, 29.2])
    out, tie = bs.decide("1x8x16", "2x4x16", cmp)
    assert not tie and out["shape"] == "2x4x16"
    out, tie = bs.decide("1x8x16", "2x4x16", bs.paired([30.0, 31.0], [30.05, 30.98]))
    assert tie
    out, _ = bs.decide("1x8x16", "2x4x16", bs.paired([30.0, 31.0], [30.05, 30.98]), {"1x8x16": 10.0, "2x4x16": 9.0})
    assert out["shape"] == "2x4x16"
    out, _ = bs.decide("1x8x16", "2x4x16", bs.paired([30.0], [29.8]))
    assert out["shape"] == "1x8x16"


def test_stale_rule_and_speed_check():
    recs = [{"frozen": [0.04, 0.04], "reassign": [0.01, 0.01]}]
    assert rules.stale_db(recs) == pytest.approx(20 * math.log10(2))
    assert rules.reassign_rule(rules.stale_db(recs)) and not rules.reassign_rule(0.46)
    assert rules.speed_check(1.13) == "sparse" and rules.speed_check(1.02) == "dense" and rules.speed_check(None) == "pending"


def test_tables_complete_needs_final_latent_J(tmp_path, monkeypatch):
    from decoupattn.planner import pipeline

    u = Units("steps", 4, (1, 2, 3), 7, 2)
    monkeypatch.setattr(Units, "of", classmethod(lambda cls, a: u))

    class Fake:
        def __init__(self):
            self.cfg = {"planner": {"distortion_clips": 2, "sens_clips": 2}}

        def clips(self, split):
            return ["a", "b"]

    for d in ("tables", "distortion", "sensitivity"):
        (tmp_path / d).mkdir()
    (tmp_path / "tables" / "D.npz").write_bytes(b"")
    for c in "ab":
        (tmp_path / "distortion" / f"{c}_s{SEED}.npz").write_bytes(b"")
    names = ["dense"] + [g[0] for g in sensitivity.groups(u, sensitivity.default_step_groups(u))]
    p = tmp_path / "sensitivity" / f"J_s{SEED}.jsonl"
    p.write_text("".join(json.dumps({"group": g, "clip": c, "mse": 1.0}) + "\n" for c in "ab" for g in names))
    assert not pipeline.tables_complete(Fake(), str(tmp_path))
    lat = {"space": sensitivity.SPACE}
    p.write_text("".join(json.dumps({"group": g, "clip": c, "mse": 1.0, **lat, **where(u, g)}) + "\n"
                         for c in "ab" for g in names))
    assert pipeline.tables_complete(Fake(), str(tmp_path))
    f = Fake()
    f.cfg["planner"]["sensitivity"] = {"kind": "injection", "path": "x.jsonl"}
    with pytest.raises(ValueError):
        pipeline.sens_source(f, str(tmp_path))
    f.cfg["planner"]["sensitivity"] = {"kind": "map", "path": "x.json"}
    with pytest.raises(ValueError):
        pipeline.sens_source(f, str(tmp_path))
