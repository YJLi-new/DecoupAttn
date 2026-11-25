from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from decoupattn import table1 as P
from decoupattn.cli import parser
from decoupattn.config import load_config
from decoupattn.eval import table1 as T
from decoupattn.models import MODELS


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def dump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def record(psnr: dict[str, list[float]], timing: dict | None = None, dense_ms: float | None = None) -> dict:
    methods = {c: {"clips": {f"c{i}": {"psnr": p, "ssim": p / 100, "lpips": 1 - p / 100} for i, p in enumerate(v)}}
               for c, v in psnr.items()}
    rec = {"model": "cp25", "methods": methods, "timing": timing or {}}
    if dense_ms:
        rec["dense_ms"] = dense_ms
    return rec


def full_record(shift: float = 0.0) -> dict:
    psnr = {c: [20.0 + i + shift, 22.0 + i + shift, 27.0 + i + shift] for i, c in enumerate(T.COLUMNS)}
    timing = {c: {"dense_ms": 1000.0 + 10 * i, "method_ms": 500.0 + 100 * i} for i, c in enumerate(T.COLUMNS[1:])}
    return record(psnr, timing, dense_ms=1200.0)


def test_clip_stats_are_mean_and_sample_sd():
    v = [20.0, 22.0, 27.0]
    s = T.clip_stats(record({"ours": v})["methods"]["ours"]["clips"])
    assert s["n"] == 3
    assert s["psnr"] == pytest.approx(np.mean(v)) and s["psnr_sd"] == pytest.approx(np.std(v, ddof=1))
    assert s["lpips"] == pytest.approx(1 - np.mean(v) / 100)
    one = T.clip_stats({"a": {"psnr": 30.0}})
    assert one["psnr"] == 30.0 and one["psnr_sd"] is None and "ssim" not in one


def test_clip_stats_rejects_a_wrong_clip_set():
    clips = record({"ours": [1.0, 2.0]})["methods"]["ours"]["clips"]
    T.clip_stats(clips, ["c0", "c1"])
    with pytest.raises(ValueError):
        T.clip_stats(clips, ["c0", "c1", "c2"])


def test_speedup_anchors():
    rec = record({"ours": [1.0, 2.0]}, {"ours": {"dense_ms": 1000.0, "method_ms": 800.0},
                                        "calibatt": {"dense_ms": 900.0, "method_ms": 450.0, "own_dense": True}}, 1200.0)
    assert T.speedup(rec, "floor") == 1.0
    assert T.speedup(rec, "ours", "process") == pytest.approx(1.25)
    assert T.speedup(rec, "ours", "common") == pytest.approx(1.5)
    assert T.speedup(rec, "calibatt", "common") == pytest.approx(2.0)
    assert T.speedup(rec, "svg2") is None
    del rec["dense_ms"]
    assert T.speedup(rec, "ours", "common") == pytest.approx(1.25)
    with pytest.raises(ValueError):
        T.speedup(rec, "ours", "fastest")


def test_row_delta_uses_printed_values():
    row = T.model_row(full_record())
    assert set(row) == {*T.COLUMNS, "delta"}
    base = [row[b]["psnr"] for b in T.BASELINES]
    assert row["delta"]["psnr"] == pytest.approx(round(row["ours"]["psnr"] - sum(base) / 7, 2))
    sp = [row[b]["speedup"] for b in T.BASELINES]
    assert row["delta"]["speedup"] == pytest.approx(float(f"{row['ours']['speedup'] - sum(sp) / 7:.2f}"))
    assert row["floor"]["speedup"] == 1.0 and row["ours"]["n"] == 3
    for cell in row.values():
        for f, v in cell.items():
            if f in T.DIGITS and v is not None:
                assert v == float(f"{v:.{T.DIGITS[f]}f}")


def test_partial_row_has_no_delta_and_no_average():
    rec = full_record()
    del rec["methods"]["svoo"]
    table = T.assemble({"cp25": rec})
    assert "svoo" not in table["cp25"] and "delta" not in table["cp25"] and "average" not in table


def test_average_row_and_best():
    records = {m: dict(full_record(shift=i), model=m) for i, m in enumerate(T.MODEL_KEYS)}
    table = T.assemble(records)
    avg = table["average"]
    assert avg["ours"]["psnr"] == pytest.approx(np.mean([table[m]["ours"]["psnr"] for m in T.MODEL_KEYS]), abs=0.005)
    assert "psnr_sd" not in avg["ours"] and "delta" in avg
    assert T.best(table["cp25"], "psnr") == "ours" and T.best(table["cp25"], "lpips") == "ours"
    table["cp25"]["svgear"]["psnr"] = 99.0
    assert T.best(table["cp25"], "psnr") == "svgear"


def test_compare_reports_each_cell():
    table = T.assemble({"cp25": full_record()})
    paper = json.loads(json.dumps({"cp25": {c: {k: v for k, v in cell.items() if k != "n"} for c, cell in
                                            table["cp25"].items()}}))
    assert all(c["status"] == "matched" for c in T.compare(table, paper))
    paper["cp25"]["ours"]["psnr"] += 0.01
    del table["cp25"]["sol"]
    cmp = T.compare(table, paper)
    bad = [c for c in cmp if c["status"] == "mismatch"]
    assert [(c["column"], c["field"]) for c in bad] == [("ours", "psnr")]
    assert {c["column"] for c in cmp if c["status"] == "missing"} == {"sol"}
    s = T.compare_summary(cmp)["cp25"]
    assert s["ours"]["mismatch"] and not s["floor"]["mismatch"] and s["sol"]["missing"]


def test_outputs(tmp_path):
    records = {m: dict(full_record(shift=i), model=m) for i, m in enumerate(T.MODEL_KEYS)}
    table = T.assemble(records)
    paths = T.write_outputs(table, str(tmp_path))
    assert json.loads(read(paths["json"]))["cp25"]["ours"]["psnr"] == table["cp25"]["ours"]["psnr"]
    rows = read(paths["csv"]).strip().split("\n")
    assert len(rows) == 1 + 11 * 10
    md = read(paths["md"])
    assert md.count("\n") == 2 + 2 * 11 and "**" in md
    tex = read(paths["tex"])
    assert tex.count("\\\\") == 2 * 11 and "\\vsd{" in tex and "\\spd{" in tex


def test_reference_records_cover_the_table():
    paper = T.load_paper()
    assert set(paper) == {*T.MODEL_KEYS, "average"} and set(T.MODEL_KEYS) == set(MODELS)
    for m in T.MODEL_KEYS:
        rec = T.load_reference(m)
        split = json.loads(read(os.path.join(os.path.dirname(T.REFERENCE_DIR), "splits", f"{m}.json")))
        test = [c for c in split["test"] if c not in split.get("dropped", [])]
        for col in T.COLUMNS:
            assert set(rec["methods"][col]["clips"]) == set(test), (m, col)
        assert set(rec["timing"]) == set(T.COLUMNS[1:])


@pytest.mark.parametrize("anchor", T.ANCHORS)
def test_reference_records_reproduce_the_printed_quality(anchor):
    table = T.assemble({m: T.load_reference(m) for m in T.MODEL_KEYS}, anchor)
    cmp = T.compare(table, T.load_paper())
    assert not [c for c in cmp if c["status"] == "missing"]
    psnr = [c for c in cmp if c["field"] in ("psnr", "psnr_sd")]
    assert psnr and all(c["status"] == "matched" for c in psnr)
    for c in cmp:
        if c["field"] in ("ssim", "ssim_sd", "lpips", "lpips_sd"):
            assert abs(c["got"] - c["paper"]) < 0.0015, c
    speed = [c for c in cmp if c["field"] == "speedup"]
    if anchor == "common":
        off = {(c["model"], c["column"]) for c in speed if c["status"] != "matched"}
        assert off <= {("t25", "calibatt"), ("dr", "calibatt"), ("alaya", "svg2")}, off
        assert all(abs(c["got"] - c["paper"]) < 0.0101 for c in speed)
        assert {(c["model"], c["column"]): c["got"] for c in speed if c["model"] in ("wan22", "dr")
                and c["column"] in ("sol", "pisa")} == {("wan22", "sol"): 1.31, ("wan22", "pisa"): 1.26,
                                                         ("dr", "sol"): 1.21, ("dr", "pisa"): 1.23}
    else:
        assert all(abs(c["got"] - c["paper"]) < 0.1 for c in speed)


def test_parse_list():
    assert P.parse_list(None, P.STAGES, "stages") == P.STAGES
    assert P.parse_list("none", T.BASELINES, "methods") == ()
    assert P.parse_list("timing, dense", P.STAGES, "stages") == ("dense", "timing")
    with pytest.raises(SystemExit):
        P.parse_list("dense,render", P.STAGES, "stages")


def test_parser():
    a = parser().parse_args(["table1", "--model", "cp25", "--run-dir", "R", "--methods", "svg2", "--stages", "assemble"])
    assert (a.model, a.run_dir, a.anchor, a.seed, a.check, a.from_reference) == ("cp25", "R", "common", 1000, False, False)
    a = parser().parse_args(["generate", "--model", "cp25", "--run-dir", "R", "--baseline", "svgear"])
    assert a.baseline == "svgear" and not a.dense and a.plan is None
    a = parser().parse_args(["generate", "--model", "cp25", "--run-dir", "R", "--dense", "--dense-backend", "fi"])
    assert a.dense and a.dense_backend == "fi"
    a = parser().parse_args(["time", "--model", "cp25", "--run-dir", "R", "--baseline", "pisa"])
    assert a.baseline == "pisa"
    with pytest.raises(SystemExit):
        parser().parse_args(["generate", "--model", "cp25", "--run-dir", "R", "--dense", "--baseline", "pisa"])
    with pytest.raises(SystemExit):
        parser().parse_args(["table1", "--model", "cp25", "--anchor", "fastest"])


def test_every_model_names_its_backends():
    for m in MODELS:
        cfg = load_config(m, resolve=False)
        ref, floor = T.reference_backend(cfg), cfg["model"]["floor_backend"]
        assert floor != ref and T.floor_tag(cfg) == f"dense_{floor}"


def test_set_dense_backend():
    class A:
        def __init__(self):
            self.cfg = {"model": {"dense_backend": "fa3"}}
            self.backend = "fa3"

    class B:
        def __init__(self):
            self.cfg = {"model": {"dense_backend": "fa3"}}

        @property
        def backend(self):
            return self.cfg["model"]["dense_backend"]

    for ad in (A(), B()):
        P.set_dense_backend(ad, "fi")
        assert ad.backend == "fi" and ad.cfg["model"]["dense_backend"] == "fi"


def write_run(run: str, cfg: dict, columns, own=()) -> None:
    tags = {"floor": T.floor_tag(cfg), "ours": T.OURS_TAG}
    for i, col in enumerate(columns):
        tag = tags.get(col, col)
        os.makedirs(os.path.join(run, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(run, "timing"), exist_ok=True)
        clips = [{"clip": f"test_{j:02d}", "psnr": 20.0 + i + j, "ssim": 0.5 + 0.01 * i, "lpips": 0.3 - 0.01 * i}
                 for j in range(4)]
        dump({"tag": tag, "clips": clips}, os.path.join(run, "metrics", f"{tag}.json"))
        if col != "floor":
            t = {"dense_ms": 1000.0, "method_ms": 400.0 + 100 * i, "speedup": 0.0}
            if col in own:
                t["own_pipeline"] = True
            dump(t, os.path.join(run, "timing", f"{tag}.json"))


def test_collect_and_assemble_from_run_directories(tmp_path, capsys):
    cfg = load_config("cp25", resolve=False)
    run = str(tmp_path / "cp25")
    write_run(run, cfg, T.COLUMNS, own=("calibatt",))
    rec = T.collect(run, "cp25", cfg)
    assert set(rec["methods"]) == set(T.COLUMNS) and rec["timing"]["calibatt"]["own_dense"]
    assert rec["methods"]["ours"]["clips"]["test_03"]["psnr"] == 20.0 + 8 + 3
    a = parser().parse_args(["table1", "--model", "cp25", "--run-dir", run, "--stages", "assemble"])
    table = P.run_model(a)
    assert table["cp25"]["ours"]["speedup"] == pytest.approx(1000.0 / (400.0 + 100 * 8), abs=0.005)
    assert os.path.exists(os.path.join(run, "table1", "table1.csv")) and "CalibAtt" in capsys.readouterr().out
    a = parser().parse_args(["table1", "--model", "all", "--run-dir", str(tmp_path), "--check"])
    table = P.assemble_all(a)
    assert set(table) == {"cp25"} and os.path.exists(os.path.join(str(tmp_path), "table1", "table1.md"))
    assert "against the printed table" in capsys.readouterr().out


def test_partial_run_directory(tmp_path):
    cfg = load_config("dr", resolve=False)
    run = str(tmp_path / "dr")
    write_run(run, cfg, ("floor", "svg2", "ours"))
    row = T.assemble({"dr": T.collect(run, "dr", cfg)})["dr"]
    assert set(row) == {"floor", "svg2", "ours"} and not math.isnan(row["ours"]["psnr"])


def test_from_reference_command(capsys, tmp_path):
    a = parser().parse_args(["table1", "--model", "all", "--from-reference", "--check", "--out", str(tmp_path)])
    P.main(a)
    out = capsys.readouterr().out
    assert "Avg." in out and "cells matched" in out and os.path.exists(os.path.join(str(tmp_path), "table1.tex"))
    a = parser().parse_args(["table1", "--model", "hy", "--from-reference"])
    P.main(a)
    assert "HY" in capsys.readouterr().out


class FakeAdapter:
    name = "cp25"

    def __init__(self, cfg):
        self.cfg, self.timer, self.loaded, self.generated = cfg, None, False, []
        self.backend = cfg["model"]["dense_backend"]

    def load(self, device="cuda"):
        self.loaded = True

    def clips(self, split):
        return ["test_00", "test_01", "test_02"]

    def generate(self, clip, hook=None, seed=1000, steps=None):
        rng = np.random.default_rng(seed + int(clip[-2:]))
        x = rng.integers(40, 200, size=(2, 16, 16, 3)).astype(np.float64)
        x += {"fa3": 0.0, "fi": 1.0}[self.backend]
        if hook is not None:
            x += hook(None, None, None, None)
        self.generated.append((clip, self.backend, hook is not None))
        return x

    def decode(self, latent):
        return np.clip(latent, 0, 255).astype(np.uint8)


def save_npy(path, video, fps=16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path[:-4] + ".npy", video)


class FakeHook:
    def __init__(self, shift):
        self.shift, self.resets = shift, 0

    def reset(self):
        self.resets += 1

    def __call__(self, q, k, v, ctx):
        return self.shift

    def summary(self):
        return {"density": 0.2}


class FakeBaseline:
    standalone, reference, timing_settings = False, "dense", None

    def __init__(self, name, shift):
        self.name, self.shift, self.settings, self.prepared = name, shift, {"knob": shift}, 0

    def prepare(self, adapter, run_dir, device="cuda", log=print):
        self.prepared += 1

    def hook(self, adapter, run_dir):
        return FakeHook(self.shift)

    def timing_hook(self, adapter, run_dir):
        return self.hook(adapter, run_dir)


def test_stages_on_a_stand_in_model(tmp_path, monkeypatch):
    import copy

    from decoupattn import baselines, cli
    from decoupattn.eval import frames
    cfg = load_config("cp25", resolve=False)
    adapters = []

    def adapter_for(model, dense_backend=None):
        c = copy.deepcopy(cfg)
        c["model"]["dense_backend"] = dense_backend or c["model"]["dense_backend"]
        adapters.append(FakeAdapter(c))
        return adapters[-1]

    made = {m: FakeBaseline(m, 3.0 * (i + 2)) for i, m in enumerate(T.BASELINES)}
    monkeypatch.setattr(cli, "adapter_for", adapter_for)
    monkeypatch.setattr(cli, "make_runtime", lambda adapter, plan, **ov: FakeHook(2.0))
    monkeypatch.setattr(cli, "log", lambda *a: None)
    monkeypatch.setattr(baselines, "get_baseline", lambda method, model, **kw: made[method])
    monkeypatch.setattr(frames, "save_video", save_npy)
    run = str(tmp_path / "cp25")
    plan = str(tmp_path / "plan.npz")
    open(plan, "w").close()
    args = ["table1", "--model", "cp25", "--run-dir", run, "--plan", plan, "--no-lpips", "--device", "cpu"]
    P.run_model(parser().parse_args([*args, "--methods", "none", "--stages", "dense,floor,ours,metrics,timing"]))
    ad = adapters[-1]
    assert ("test_00", "fa3", False) in ad.generated and ("test_00", "fi", False) in ad.generated
    assert ad.backend == "fa3"
    assert sorted(os.listdir(os.path.join(run, "metrics"))) == ["decoupattn.json", "dense_fi.json"]
    assert not any(b.prepared for b in made.values())
    P.run_model(parser().parse_args([*args, "--methods", "svgear", "--stages", "baselines,metrics,timing"]))
    assert made["svgear"].prepared and not made["pisa"].prepared
    others = ",".join(m for m in T.BASELINES if m != "svgear")
    P.run_model(parser().parse_args([*args, "--methods", others, "--stages", "baselines,metrics,timing"]))
    n = len(adapters[-1].generated)
    P.run_model(parser().parse_args([*args, "--methods", others, "--stages", "baselines,metrics"]))
    assert len(adapters[-1].generated) == 0 and n > 0
    table = P.run_model(parser().parse_args([*args, "--stages", "assemble"]))
    row = table["cp25"]
    assert set(row) == {*T.COLUMNS, "delta"} and all(row[c]["n"] == 3 for c in T.COLUMNS)
    assert row["floor"]["psnr"] > row["ours"]["psnr"] > row["calibatt"]["psnr"] > row["sparsepr"]["psnr"]
    assert all(row[c]["speedup"] is not None for c in T.COLUMNS)
    assert os.path.exists(os.path.join(run, "table1", "table1.tex"))


def test_ours_is_left_out_without_a_plan(tmp_path, monkeypatch):
    import copy

    from decoupattn import cli
    from decoupattn.eval import frames
    cfg = load_config("cp25", resolve=False)
    monkeypatch.setattr(cli, "adapter_for", lambda model, dense_backend=None: FakeAdapter(copy.deepcopy(cfg)))
    monkeypatch.setattr(cli, "log", lambda *a: None)
    monkeypatch.setattr(frames, "save_video", save_npy)
    run = str(tmp_path / "cp25")
    a = parser().parse_args(["table1", "--model", "cp25", "--run-dir", run, "--methods", "none", "--no-lpips",
                             "--device", "cpu"])
    row = P.run_model(a)["cp25"]
    assert set(row) == {"floor"}
