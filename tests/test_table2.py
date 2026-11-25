from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from decoupattn import table2 as P
from decoupattn.cli import parser
from decoupattn.config import load_config
from decoupattn.eval import table2 as T
from decoupattn.eval.table1 import MODEL_KEYS
from decoupattn.plan import Plan, load_plan
from decoupattn.planner import pipeline as C


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def record(psnr: dict[str, list[float]], ms: dict[str, float] | None = None, model: str = "cp25") -> dict:
    variants = {v: {"clips": {f"c{i}": {"psnr": p} for i, p in enumerate(x)}} for v, x in psnr.items()}
    return {"model": model, "variants": variants, "timing": {v: {"method_ms": x} for v, x in (ms or {}).items()}}


FULL = [20.0, 22.0, 27.0, 30.0, 25.0, 24.0, 21.0, 29.0]


def losing(by: float, better_on: int = 0) -> list[float]:
    return [p + 0.1 if i < better_on else p - by for i, p in enumerate(FULL)]


def full_record(model: str = "cp25", frozen: bool = True) -> dict:
    psnr = {"Full": FULL, **{v: losing(0.5 + 0.25 * i) for i, v in enumerate(T.VARIANTS) if frozen or v != "Frozen"}}
    ms = {v: 1000.0 - 10.0 * i for i, v in enumerate(psnr)}
    return record(psnr, ms, model)


def test_row_is_the_paired_difference_to_the_full_method():
    var = [19.0, 22.5, 25.0, 30.0, 24.0, 23.0, 20.5, 26.0]
    row = T.model_row(record({"Full": FULL, "Tied": var}, {"Full": 1000.4, "Tied": 900.0}))
    diff = np.array(var) - np.array(FULL)
    assert row["Full"]["psnr"] == pytest.approx(np.mean(FULL), abs=0.005)
    assert row["Full"]["psnr_sd"] == pytest.approx(np.std(FULL, ddof=1), abs=0.005) and row["Full"]["ms"] == 1000
    cell = row["Tied"]
    assert cell["dpsnr"] == pytest.approx(diff.mean(), abs=0.005)
    assert cell["dpsnr_sd"] == pytest.approx(diff.std(ddof=1), abs=0.005)
    assert cell["wins"] == int((diff < 0).sum()) == 6 and cell["n"] == 8
    assert cell["dms"] == -100
    assert set(row) == {"Full", "Tied"}


def test_row_needs_the_full_method_and_the_same_clips():
    assert T.model_row(record({"Tied": FULL})) == {}
    with pytest.raises(ValueError):
        T.model_row(record({"Full": FULL, "Tied": FULL[:-1]}))
    with pytest.raises(ValueError):
        T.model_row(record({"Full": FULL}), expected=[f"c{i}" for i in range(9)])


def test_latency_differences():
    rec = record({"Full": FULL, "Tied": FULL, "U": FULL, "D": FULL}, {"Full": 1000.0, "Tied": 900.0, "U": 1100.0})
    assert T.delta_ms(rec, "Tied") == -100.0 and T.delta_ms(rec, "D") is None
    rec["timing"]["U"]["full_ms"] = 1050.0
    assert T.delta_ms(rec, "U") == 50.0
    rec["timing"]["U"]["delta_ms"] = 42.0
    assert T.delta_ms(rec, "U") == 42.0
    row = T.model_row(rec)
    assert row["U"]["dms"] == 42 and row["D"]["dms"] is None


def test_latencies_round_half_away_from_zero():
    assert T.printed(2.5, "ms") == 3 and T.printed(-2.5, "dms") == -3 and T.printed(-2.4, "dms") == -2
    assert T.printed(None, "dms") is None and T.printed(1.005, "psnr") == pytest.approx(1.0, abs=0.011)


def test_support_rule():
    assert T.supports({"dpsnr": -0.30, "wins": 6}) and T.supports({"dpsnr": -2.0, "wins": 8})
    assert not T.supports({"dpsnr": -0.29, "wins": 8}) and not T.supports({"dpsnr": -2.0, "wins": 5})


def test_average_and_support_counts():
    records = {m: full_record(m, frozen=m != "dr") for m in MODEL_KEYS}
    records["hy"]["variants"]["Tied"] = {"clips": {f"c{i}": {"psnr": p} for i, p in enumerate(losing(3.0, better_on=3))}}
    table = T.assemble(records)
    avg = table["average"]
    assert set(table) == {*MODEL_KEYS, "average"} and "Frozen" not in table["dr"]
    assert avg["Full"]["psnr"] == pytest.approx(np.mean(FULL), abs=0.005) and avg["Full"]["ms"] == 1000
    assert (avg["Tied"]["supp"], avg["Tied"]["of"]) == (9, 10)
    assert (avg["Frozen"]["supp"], avg["Frozen"]["of"]) == (9, 9)
    assert avg["U"]["dpsnr"] == pytest.approx(table["cp25"]["U"]["dpsnr"]) and avg["U"]["dms"] == table["cp25"]["U"]["dms"]
    assert "average" not in T.assemble({m: records[m] for m in MODEL_KEYS[:3]})


def test_compare_reports_each_cell():
    table = T.assemble({"cp25": full_record()})
    paper = {"cp25": {"Full": {"psnr": table["cp25"]["Full"]["psnr"], "ms": 999},
                      "Tied": {"dpsnr": table["cp25"]["Tied"]["dpsnr"], "wins": 8}},
             "hy": {"Full": {"psnr": 1.0}}}
    status = {(c["model"], c["column"], c["field"]): c["status"] for c in T.compare(table, paper)}
    assert status[("cp25", "Full", "psnr")] == "matched" and status[("cp25", "Full", "ms")] == "mismatch"
    assert status[("cp25", "Tied", "wins")] == "matched" and status[("hy", "Full", "psnr")] == "missing"


def test_outputs(tmp_path):
    table = T.assemble({m: full_record(m, frozen=m != "dr") for m in MODEL_KEYS})
    paths = T.write_outputs(table, str(tmp_path))
    assert json.loads(read(paths["json"]))["cp25"]["Tied"]["dpsnr"] == table["cp25"]["Tied"]["dpsnr"]
    assert len(read(paths["csv"]).strip().split("\n")) == 1 + 11 * 7 - 1
    md = read(paths["md"])
    assert md.count("\n") == 2 + 2 * 7 and "Supp." in md and "9/9" in md
    tex = read(paths["tex"])
    assert tex.count("\\\\") == 2 * 7 and "\\tms{" in tex
    assert "Supp." not in T.to_markdown(T.assemble({"cp25": full_record()}))


def test_reference_records_cover_the_table():
    paper = T.load_paper()
    assert set(paper) == {*MODEL_KEYS, "average"}
    for m in MODEL_KEYS:
        rec = T.load_reference(m)
        split = json.loads(read(os.path.join(os.path.dirname(T.REFERENCE_DIR), "splits", f"{m}.json")))
        assert set(rec["variants"]) == set(paper[m]) == set(rec["timing"]), m
        for v in rec["variants"]:
            assert list(rec["variants"][v]["clips"]) == split["val"], (m, v)
    assert "Frozen" not in paper["dr"] and all("Frozen" in paper[m] for m in MODEL_KEYS if m != "dr")


def test_reference_records_reproduce_the_printed_table():
    table = T.assemble({m: T.load_reference(m) for m in MODEL_KEYS})
    cmp = T.compare(table, T.load_paper())
    assert len(cmp) > 250 and not [c for c in cmp if c["status"] == "missing"]
    off = [c for c in cmp if c["status"] == "mismatch"]
    assert len(off) <= 3 and all(c["field"] == "dpsnr" and abs(c["got"] - c["paper"]) < 0.015 for c in off)


def test_parser():
    a = parser().parse_args(["table2", "--model", "cp25", "--run-dir", "R", "--variants", "U,Frozen", "--stages", "assemble"])
    assert (a.model, a.run_dir, a.seed, a.check, a.from_reference, a.plan) == ("cp25", "R", 1000, False, False, None)
    assert P.parse_list(a.variants, T.VARIANTS, "variants") == ("U", "Frozen")
    with pytest.raises(SystemExit):
        P.parse_list("U,Rebuild", T.VARIANTS, "variants")
    with pytest.raises(SystemExit):
        parser().parse_args(["table2", "--model", "sd3"])


def test_dense_reference_tag():
    assert P.dense_tag(load_config("cp25", resolve=False)) == "dense"
    assert P.dense_tag(load_config("dr", resolve=False)) == "dense_fa3"


def write_run(run: str, rows) -> None:
    for i, v in enumerate(rows):
        clips = [{"clip": f"val_{j:02d}", "psnr": 30.0 - i + 0.1 * j, "ssim": 0.9} for j in range(8)]
        dump({"tag": T.tag(v), "seed": 1000, "clips": clips}, os.path.join(run, "metrics", f"{T.tag(v)}.json"))
        dump({"method_ms": 500.0 - 20 * i, "dense_ms": 900.0}, os.path.join(run, "timing", f"{T.tag(v)}.json"))


def test_collect_and_assemble_from_run_directories(tmp_path, capsys):
    run = str(tmp_path / "cp25")
    write_run(run, T.ROWS)
    rec = T.collect(run, "cp25")
    assert set(rec["variants"]) == set(T.ROWS) and rec["seed"] == 1000
    table = P.run_model(parser().parse_args(["table2", "--model", "cp25", "--run-dir", run, "--stages", "assemble"]))
    assert table["cp25"]["Centroid"]["dpsnr"] == pytest.approx(-3.0) and table["cp25"]["Centroid"]["dms"] == -60
    assert table["cp25"]["Centroid"]["wins"] == 8
    assert os.path.exists(os.path.join(run, "table2", "table2.csv")) and "Centroid" in capsys.readouterr().out
    write_run(str(tmp_path / "dr"), T.ROWS[1:])
    table = P.assemble_all(parser().parse_args(["table2", "--model", "all", "--run-dir", str(tmp_path), "--check"]))
    assert set(table) == {"cp25"} and os.path.exists(os.path.join(str(tmp_path), "table2", "table2.md"))
    assert "against the printed table" in capsys.readouterr().out


def test_from_reference_command(capsys, tmp_path):
    P.main(parser().parse_args(["table2", "--model", "all", "--from-reference", "--check", "--out", str(tmp_path)]))
    out = capsys.readouterr().out
    assert "Avg." in out and "cells matched" in out and os.path.exists(os.path.join(str(tmp_path), "table2.tex"))
    P.main(parser().parse_args(["table2", "--model", "dr", "--from-reference"]))
    out = capsys.readouterr().out
    assert "DR" in out and "Frozen" not in out


def tiny_plan(reassign: str | None = "reassign") -> Plan:
    return Plan(mask=np.ones((1, 1, 1, 1, 1), bool), step_row=np.zeros(1, np.int64), block_shape=(1, 2, 4),
                kind=np.zeros((1, 1, 1), np.int8), k_per_head=np.ones((1, 1, 1), np.int16),
                estimator=np.ones((1, 1), np.int8), reassign=reassign)


class FakeAdapter:
    name, block_shape = "cp25", None

    def __init__(self):
        self.cfg, self.timer, self.generated = {}, None, []

    def load(self, device="cuda"):
        pass

    def clips(self, split):
        return [f"{split}_{i:02d}" for i in range(3)]

    def generate(self, clip, hook=None, seed=1000, steps=None):
        x = np.random.default_rng(seed + int(clip[-2:])).integers(40, 200, size=(2, 16, 16, 3)).astype(np.float64)
        self.generated.append((clip, hook.shift if hook is not None else None))
        return x if hook is None else x + hook.shift

    def decode(self, latent):
        return np.clip(latent, 0, 255).astype(np.uint8)


class FakeHook:
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, q, k, v, ctx):
        return None


SHIFT = {"plan": 2.0, "Tied": 9.0, "Tied_dagger": 12.0, "Centroid": 6.0, "U": 7.0, "D": 5.0, "Frozen": 4.0}


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    from decoupattn import cli
    from decoupattn.eval import frames
    calls = {"adapters": [], "calibrate": [], "variants": []}

    def adapter_for(model, dense_backend=None):
        calls["adapters"].append(FakeAdapter())
        return calls["adapters"][-1]

    def calibrate(adapter, run_dir, plan, seed=1000, device="cuda"):
        calls["calibrate"].append(plan)
        dump({}, os.path.join(run_dir, "tables", "D.npz"))

    def calibrate_variants(adapter, run_dir, names, device="cuda", plan=None):
        calls["variants"].append(tuple(names))
        for n in names:
            os.makedirs(os.path.join(run_dir, "variants"), exist_ok=True)
            shutil.copy(plan, P.variant_plan(run_dir, n))

    def save_npy(path, video, fps=16):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path[:-4] + ".npy", video)

    monkeypatch.setattr(cli, "adapter_for", adapter_for)
    monkeypatch.setattr(cli, "make_runtime", lambda adapter, plan, **ov: FakeHook(SHIFT[os.path.basename(plan)[:-4]]))
    monkeypatch.setattr(cli, "log", lambda *a: None)
    monkeypatch.setattr(P, "calibrate", calibrate)
    monkeypatch.setattr(C, "tables_complete", lambda adapter, run_dir: os.path.exists(f"{run_dir}/tables/D.npz"))
    monkeypatch.setattr(C, "calibrate_variants", calibrate_variants)
    monkeypatch.setattr(frames, "save_video", save_npy)
    run = str(tmp_path / "cp25")
    return run, calls, ["table2", "--model", "cp25", "--run-dir", run, "--device", "cpu"]


def test_stages_on_a_stand_in_model(stand_in):
    run, calls, args = stand_in
    tiny_plan().save(os.path.join(run, "plan.npz"))
    table = P.run_model(parser().parse_args(args))
    assert calls["calibrate"] == [os.path.join(run, "plan.npz")] and calls["variants"] == [T.VARIANTS]
    ad = calls["adapters"][-1]
    assert ("val_00", None) in ad.generated and ("val_02", SHIFT["U"]) in ad.generated
    assert ("test_00", SHIFT["Frozen"]) in ad.generated
    assert sorted(os.listdir(os.path.join(run, "metrics"))) == sorted(f"{T.tag(v)}.json" for v in T.ROWS)
    assert sorted(os.listdir(os.path.join(run, "timing"))) == sorted(f"{T.tag(v)}.json" for v in T.ROWS)
    row = table["cp25"]
    assert set(row) == set(T.ROWS) and all(row[v]["n"] == 3 and row[v]["wins"] == 3 for v in T.VARIANTS)
    assert row["Tied_dagger"]["dpsnr"] < row["Tied"]["dpsnr"] < row["Frozen"]["dpsnr"] < 0
    assert all(row[v]["dms"] is not None for v in T.VARIANTS) and row["Full"]["ms"] is not None
    assert os.path.exists(os.path.join(run, "table2", "table2.tex"))
    P.run_model(parser().parse_args(args))
    assert len(calls["calibrate"]) == 1 and len(calls["variants"]) == 1 and calls["adapters"][-1].generated == []


def test_no_frozen_variant_when_the_plan_keeps_its_clusters(stand_in):
    run, calls, args = stand_in
    plan = os.path.join(os.path.dirname(run), "released", "plan.npz")
    tiny_plan(reassign=None).save(plan)
    row = P.run_model(parser().parse_args([*args, "--plan", plan]))["cp25"]
    assert set(row) == set(T.ROWS) - {"Frozen"} and calls["variants"] == [T.VARIANTS[:-1]]
    assert calls["calibrate"] == [plan] and not os.path.exists(P.variant_plan(run, "Frozen"))


def test_derived_variants_need_the_plan_only(stand_in, monkeypatch):
    run, _, args = stand_in
    monkeypatch.undo()
    monkeypatch.setattr("decoupattn.cli.adapter_for", lambda model, dense_backend=None: FakeAdapter())
    monkeypatch.setattr("decoupattn.cli.log", lambda *a: None)
    tiny_plan().save(os.path.join(run, "plan.npz"))
    P.run_model(parser().parse_args([*args, "--variants", "Tied_dagger", "--stages", "calibrate,variants"]))
    assert not os.path.exists(os.path.join(run, "tables"))
    assert str(load_plan(P.variant_plan(run, "Tied_dagger")).extra["abl_group"]) == "tile"
    assert not os.path.exists(P.variant_plan(run, "Frozen"))
    with pytest.raises(SystemExit):
        P.run_model(parser().parse_args(["table2", "--model", "cp25", "--run-dir", run + "_new", "--stages", "variants"]))


def test_calibration_keeps_an_existing_plan(tmp_path, monkeypatch):
    from decoupattn import cli
    seen = []
    monkeypatch.setattr(cli, "block_shape_callbacks", lambda adapter, run_dir, seed: (None, None))
    monkeypatch.setattr(C, "calibrate", lambda adapter, run_dir, stages, **kw: seen.append(tuple(stages)))
    run, ad = str(tmp_path / "run"), FakeAdapter()
    P.calibrate(ad, run, os.path.join(run, "plan.npz"), device="cpu")
    assert seen == [C.STAGES] and not os.path.exists(os.path.join(run, "block_shape"))
    tiny_plan().save(os.path.join(run, "plan.npz"))
    P.calibrate(ad, run, os.path.join(run, "plan.npz"), device="cpu")
    assert "allocate" not in seen[-1] and "block_shape" not in seen[-1] and "dtable" in seen[-1]
    assert C.chosen_shape(ad, run) == (1, 2, 4) and C.reassign_of(run) is True
    C.adopt_plan(run, tiny_plan(reassign=None))
    assert C.reassign_of(run) is True
