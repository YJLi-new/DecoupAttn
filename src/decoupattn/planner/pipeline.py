from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..constants import DISTORTION_CLIPS, DTABLE_CLIPS, ESTIMATORS, SEED, TARGET_DENSITY
from ..plan import load_plan
from . import block_shape as bs
from . import distortion, dtable, rules, sensitivity, variants
from .build import build_plan, choose, summary
from .units import Units

STAGES = ("block_shape", "reassign", "dtable", "distortion", "sensitivity", "allocate")
CLIP_DEFAULTS = {"dtable_clips": DTABLE_CLIPS, "distortion_clips": DISTORTION_CLIPS, "sens_clips": sensitivity.SENS_CLIPS,
                 "stale_clips": rules.STALE_CLIPS, "block_shape_clips": bs.BLOCK_SHAPE_CLIPS}


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _pcfg(adapter) -> dict:
    return dict(adapter.cfg.get("planner", {}))


def n_clips(adapter, key: str) -> int:
    return int(_pcfg(adapter).get(key, CLIP_DEFAULTS[key]))


def _mass_path(run_dir, shape):
    return f"{run_dir}/block_shape/mass_{bs.shape_name(shape)}.pt"


def chosen_shape(adapter, run_dir) -> tuple[int, int, int]:
    if adapter.block_shape:
        return tuple(adapter.block_shape)
    return bs.parse(_read_json(f"{run_dir}/block_shape/block_shape.json")["shape"])


def tiling(adapter, run_dir, device="cuda"):
    from ..layout import Tiling

    return Tiling(*adapter.latent_grid, chosen_shape(adapter, run_dir), device=device)


def reassign_of(run_dir) -> bool:
    return bool(_read_json(f"{run_dir}/reassign/reassign.json")["reassign"])


def _own_sens_path(run_dir) -> str:
    return f"{run_dir}/sensitivity/J_s{SEED}.jsonl"


def sens_source(adapter, run_dir) -> dict:
    src = dict(_pcfg(adapter).get("sensitivity", {}))
    src.setdefault("kind", "injection")
    if src["kind"] not in ("injection", "uniform"):
        raise ValueError(f"[planner.sensitivity]: kind {src['kind']!r} (injection | uniform)")
    if src["kind"] == "injection":
        if "path" in src:
            raise ValueError("[planner.sensitivity]: an injection J is measured into <run_dir>/sensitivity; "
                             "it takes no 'path'")
        src["path"] = _own_sens_path(run_dir)
        src["clips"] = list(adapter.clips("calib")[:n_clips(adapter, "sens_clips")])
    return src


def _step_groups(src):
    return [tuple(x) for x in src["step_groups"]] if src.get("step_groups") else None


def tables_complete(adapter, run_dir) -> bool:
    if not os.path.exists(f"{run_dir}/tables/D.npz") or not all(map(os.path.exists, distortion_files(adapter, run_dir))):
        return False
    src = sens_source(adapter, run_dir)
    return src["kind"] != "injection" or not sensitivity.pending(adapter, src["path"], _step_groups(src),
                                                                 n_clips(adapter, "sens_clips"))


def distortion_files(adapter, run_dir, sub="distortion"):
    return [f"{run_dir}/{sub}/{c}_s{SEED}.npz" for c in adapter.clips("calib")[:n_clips(adapter, "distortion_clips")]]


def plan_extra(adapter) -> dict:
    return {k: np.int8(v) if isinstance(v, (bool, int)) else np.array(v)
            for k, v in _pcfg(adapter).get("plan_extra", {}).items()}


def stage_block_shape(adapter, run_dir, val_psnr=None, time_ms=None, device="cuda"):
    out = f"{run_dir}/block_shape"
    os.makedirs(out, exist_ok=True)
    if adapter.block_shape:
        _write_json(f"{out}/block_shape.json", {"shape": bs.shape_name(adapter.block_shape), "reason": "fixed by the model"})
        return
    nc = n_clips(adapter, "block_shape_clips")
    if not os.path.exists(f"{out}/probe.json"):
        rec, means, _ = bs.probe(adapter, nc, device=device)
        for t, m in means.items():
            torch.save(m, _mass_path(run_dir, t))
        _write_json(f"{out}/probe.json", rec)
    rec = _read_json(f"{out}/probe.json")
    A, B = rec["default"], rec["challenger"]
    units = Units.of(adapter)
    plans = {}
    for name in (A, B):
        path = f"{out}/val_{name}.npz"
        if not os.path.exists(path):
            from ..layout import Tiling

            til = Tiling(*adapter.latent_grid, bs.parse(name), device=device)
            plan, _, _ = bs.val_plan(units, torch.load(_mass_path(run_dir, bs.parse(name))), til, device)
            plan.save(path)
        plans[name] = load_plan(path)
    vf, tf = f"{out}/val.json", f"{out}/timing.json"
    if not os.path.exists(vf):
        if val_psnr is None:
            raise RuntimeError(f"block shape: run the validation plans {out}/val_{A}.npz and val_{B}.npz on "
                               f"{bs.BLOCK_SHAPE_VAL_CLIPS} validation clips and write {vf}")
        clips = adapter.clips("val")[:bs.BLOCK_SHAPE_VAL_CLIPS]
        _write_json(vf, {"A": list(val_psnr(plans[A], clips)), "B": list(val_psnr(plans[B], clips)), "clips": clips})
    v = _read_json(vf)
    cmp = bs.paired(v["A"], v["B"])
    ms = _read_json(tf) if os.path.exists(tf) else None
    res, tie = bs.decide(A, B, cmp, ms)
    if tie:
        if time_ms is None:
            raise RuntimeError(f"block shape: tie within {bs.BLOCK_SHAPE_MARGIN_DB} dB; time both validation plans and "
                               f"write {tf}")
        ms = {A: float(time_ms(plans[A])), B: float(time_ms(plans[B]))}
        _write_json(tf, ms)
        res, _ = bs.decide(A, B, cmp, ms)
    _write_json(f"{out}/block_shape.json", res)


def _fixed_shape_mass(adapter, til, shape, path):
    units = Units.of(adapter)
    hook = bs.MassHook(units, {shape: til})
    clips = adapter.clips("calib")[:n_clips(adapter, "block_shape_clips")]
    for c in clips:
        hook.on = True
        adapter.generate(c, hook, seed=SEED, steps=units.observed_steps)
        hook.on = False
    P = 4 if units.layout == "steps" else units.n_rows
    mean = torch.zeros(P, units.n_layers, units.n_heads, til.nt, til.nt, dtype=torch.float16)
    for (s, l), sm in hook.acc[shape].items():
        mean[s if units.layout == "steps" else units.rows.index(s), l] = (sm / len(clips)).half()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(mean, path)


def stage_reassign(adapter, run_dir, device="cuda"):
    out = f"{run_dir}/reassign"
    cfg = _pcfg(adapter)
    if "stale_db" in cfg:
        d, src = float(cfg["stale_db"]), "model configuration"
    else:
        shape = chosen_shape(adapter, run_dir)
        til = tiling(adapter, run_dir, device)
        mp = _mass_path(run_dir, shape)
        if not os.path.exists(mp):
            _fixed_shape_mass(adapter, til, shape, mp)
        nc = n_clips(adapter, "stale_clips")
        d = rules.measure_stale(adapter, til, torch.load(mp), out, steps=cfg.get("stale_steps"), n_clips=nc)
        src = f"measured ({nc} calibration clips)"
    _write_json(f"{out}/reassign.json", {"delta_stale_db": d, "reassign": rules.reassign_rule(d), "source": src})


def stage_dtable(adapter, run_dir, device="cuda"):
    dtable.measure(adapter, tiling(adapter, run_dir, device), reassign_of(run_dir), f"{run_dir}/tables",
                   n_clips(adapter, "dtable_clips"))


def stage_distortion(adapter, run_dir, device="cuda"):
    D = dtable.load(f"{run_dir}/tables/D.npz")
    distortion.measure(adapter, tiling(adapter, run_dir, device), D, reassign_of(run_dir), f"{run_dir}/distortion",
                       n_clips(adapter, "distortion_clips"))


def stage_sensitivity(adapter, run_dir, device="cuda"):
    src = sens_source(adapter, run_dir)
    if src["kind"] != "injection":
        return
    D = dtable.load(f"{run_dir}/tables/D.npz")
    sensitivity.measure(adapter, D, chosen_shape(adapter, run_dir), reassign_of(run_dir), f"{run_dir}/sensitivity",
                        _step_groups(src), n_clips(adapter, "sens_clips"), extra=plan_extra(adapter))


def stage_allocate(adapter, run_dir, target=TARGET_DENSITY):
    units = Units.of(adapter)
    E, X, RD = distortion.load_tables(units, distortion_files(adapter, run_dir))
    src = sens_source(adapter, run_dir)
    J = sensitivity.load(units, src, E[ESTIMATORS[0]])
    ch, est, info = choose(units, E, X, J, target)
    D = dtable.load(f"{run_dir}/tables/D.npz")
    shape = chosen_shape(adapter, run_dir)
    reassign = reassign_of(run_dir)
    plan = build_plan(units, D, shape, ch, est, reassign, plan_extra(adapter))
    plan.save(f"{run_dir}/plan.npz")
    smax = _pcfg(adapter).get("smax")
    _write_json(f"{run_dir}/plan.json", {"model": adapter.name, "block_shape": bs.shape_name(shape),
                                         "reassign": "reassign" if reassign else "once", "target_density": target,
                                         "speed_check": rules.speed_check(smax), "S_max": smax, "sensitivity": src,
                                         "solver": info, **summary(units, E, X, RD, J, ch, est)})
    return plan


STAGE_FUNCS = {"block_shape": stage_block_shape, "reassign": stage_reassign, "dtable": stage_dtable,
               "distortion": stage_distortion, "sensitivity": stage_sensitivity}


def calibrate(adapter, run_dir: str, stages=STAGES, val_psnr=None, time_ms=None, device="cuda"):
    os.makedirs(run_dir, exist_ok=True)
    plan = None
    for st in STAGES:
        if st not in stages:
            continue
        if st == "block_shape":
            stage_block_shape(adapter, run_dir, val_psnr, time_ms, device)
        elif st == "allocate":
            plan = stage_allocate(adapter, run_dir)
        else:
            STAGE_FUNCS[st](adapter, run_dir, device)
    return plan


def adopt_plan(run_dir: str, plan) -> None:
    for sub, obj in (("block_shape", {"shape": bs.shape_name(plan.block_shape)}),
                     ("reassign", {"reassign": plan.reassign == "reassign"})):
        path = f"{run_dir}/{sub}/{sub}.json"
        if not os.path.exists(path):
            _write_json(path, {**obj, "source": "the plan of the full method"})


def calibrate_variants(adapter, run_dir: str, names=variants.RESOLVED + variants.COPIED, device="cuda",
                       plan: str | None = None) -> dict:
    main = load_plan(plan or f"{run_dir}/plan.npz")
    out = {}
    tables = None
    for name in names:
        if name in variants.COPIED:
            p = variants.derived_variant(name, main)
        else:
            if tables is None:
                units0 = Units.of(adapter)
                tables = (units0, dtable.load(f"{run_dir}/tables/D.npz"), chosen_shape(adapter, run_dir),
                          reassign_of(run_dir), sens_source(adapter, run_dir),
                          distortion.load_tables(units0, distortion_files(adapter, run_dir))[0])
            units, D, shape, reassign, src, E_main = tables
            sub = "distortion"
            schedule = variants.SCHEDULE.get(name, reassign)
            if name in ("D", "Tied") or (name in variants.SCHEDULE and schedule != reassign):
                sub = f"variants/{name}/distortion"
                kw = {"estimators": ("none",)} if name == "D" else {"tied": True} if name == "Tied" else {}
                distortion.measure(adapter, tiling(adapter, run_dir, device), D, schedule, f"{run_dir}/{sub}",
                                   n_clips(adapter, "distortion_clips"), **kw)
            E, X, _ = distortion.load_tables(units, distortion_files(adapter, run_dir, sub))
            p = variants.solve_variant(name, units, E, X, src, D, shape, reassign, main, plan_extra(adapter),
                                       E_main=E_main)[0]
        p.save(f"{run_dir}/variants/{name}.npz")
        out[name] = p
    return out
