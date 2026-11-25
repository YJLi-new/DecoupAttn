from __future__ import annotations

import argparse
import os

from .constants import SEED
from .eval import table1 as T1
from .eval import table2 as T
from .table1 import have_videos, parse_list

STAGES = ("calibrate", "variants", "dense", "generate", "metrics", "timing", "assemble")
GPU_STAGES = ("calibrate", "variants", "dense", "generate", "timing")


def add_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser("table2", help="reproduce Table 2: one model's column, or the whole table from its columns",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=[*T1.MODEL_KEYS, "all"],
                   help="a model (run its column) or 'all' (assemble the table from <run-dir>/<model>)")
    p.add_argument("--run-dir", help="run directory of the model; with --model all: the directory of the ten of them")
    p.add_argument("--plan", help="plan .npz of the full method (default <run-dir>/plan.npz)")
    p.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGES)} (default: all, in this order)")
    p.add_argument("--variants", help=f"comma-separated variants among {','.join(T.VARIANTS)} (default: all)")
    p.add_argument("--clip", help="clip of the timing protocol (default: the first test clip)")
    p.add_argument("--from-reference", action="store_true",
                   help="assemble from the stored records of the paper's runs (data/table2/reference) instead of runs")
    p.add_argument("--check", action="store_true", help="compare every assembled cell with the printed table")
    p.add_argument("--out", help="output directory of the assembled table (default <run-dir>/table2)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda")
    return p


def dense_tag(cfg: dict) -> str:
    backend = cfg["model"]["dense_backend"]
    return "dense" if T1.reference_backend(cfg) == backend else f"dense_{backend}"


def variant_plan(run_dir: str, name: str) -> str:
    return os.path.join(run_dir, "variants", f"{name}.npz")


def calibrate(adapter, run_dir: str, plan: str, seed: int = SEED, device: str = "cuda") -> None:
    from . import cli
    from .plan import load_plan
    from .planner import pipeline as C
    stages = C.STAGES
    if os.path.exists(plan):
        C.adopt_plan(run_dir, load_plan(plan))
        stages = ("dtable", "distortion", "sensitivity")
    val_psnr, time_ms = cli.block_shape_callbacks(adapter, run_dir, seed)
    C.calibrate(adapter, run_dir, stages=stages, val_psnr=val_psnr, time_ms=time_ms, device=device)


def run_model(a) -> dict:
    from . import cli
    from .config import load_config
    from .planner.pipeline import tables_complete
    from .planner.variants import RESOLVED
    stages = parse_list(a.stages, STAGES, "stages")
    names = parse_list(a.variants, T.VARIANTS, "variants")
    if not a.run_dir:
        raise SystemExit("--run-dir is required")
    R = a.run_dir
    plan = a.plan or os.path.join(R, "plan.npz")
    dense = dense_tag(load_config(a.model, resolve=False))
    adapter = clips = None
    if any(s in stages for s in (*GPU_STAGES, "metrics")):
        adapter = cli.adapter_for(a.model)
        clips = adapter.clips("val")

    if "calibrate" in stages and any(v in RESOLVED for v in names) \
            and not (os.path.exists(plan) and tables_complete(adapter, R)):
        cli.log(f"table2 {a.model}: calibration tables")
        calibrate(adapter, R, plan, a.seed, a.device)
    if any(s in stages for s in ("variants", "generate", "timing")) and not os.path.exists(plan):
        raise SystemExit(f"no plan at {plan}: run the calibrate stage, or pass --plan")
    if any(s in stages for s in ("variants", "generate", "metrics", "timing")):
        from .plan import load_plan
        if os.path.exists(plan) and load_plan(plan).reassign != "reassign" and "Frozen" in names:
            cli.log(f"table2 {a.model}: the plan keeps its clusters, so there is no Frozen variant")
            names = tuple(v for v in names if v != "Frozen")
    plans = {"Full": plan, **{v: variant_plan(R, v) for v in names}}
    if "variants" in stages:
        from .planner.pipeline import calibrate_variants
        todo = tuple(v for v in names if not os.path.exists(plans[v]))
        if todo:
            cli.log(f"table2 {a.model}: variant plans {', '.join(todo)}")
            calibrate_variants(adapter, R, names=todo, device=a.device, plan=plan)
    if "dense" in stages:
        cli.log(f"table2 {a.model}: dense reference, {len(clips)} validation clips")
        cli.generate_clips(adapter, clips, os.path.join(R, "videos", dense), None, a.seed,
                           {"model": a.model, "tag": dense})
    if "generate" in stages:
        adapter.load()
        for v, path in plans.items():
            if not have_videos(R, T.tag(v), clips):
                cli.log(f"table2 {a.model}: {v}")
                cli.generate_clips(adapter, clips, os.path.join(R, "videos", T.tag(v)), cli.make_runtime(adapter, path),
                                   a.seed, {"model": a.model, "tag": T.tag(v), "plan": path})
    if "metrics" in stages:
        for v in plans:
            if not os.path.exists(os.path.join(R, "metrics", f"{T.tag(v)}.json")) and have_videos(R, T.tag(v), clips) \
                    and have_videos(R, dense, clips):
                cli.evaluate_tag(adapter, R, T.tag(v), dense, clips, a.device)
    if "timing" in stages:
        from .eval.timing import time_model
        adapter.load()
        clip = a.clip or adapter.clips("test")[0]
        for v, path in plans.items():
            out = os.path.join(R, "timing", f"{T.tag(v)}.json")
            if not os.path.exists(out):
                res = time_model(adapter, lambda path=path: cli.make_runtime(adapter, path), clip, a.seed, log=cli.log)
                res.update(model=a.model, plan=path)
                cli.write_json(out, res)
    if "assemble" not in stages:
        return {}
    table = T.assemble({a.model: T.collect(R, a.model)})
    finish(a, table, a.out or os.path.join(R, "table2"))
    return table


def assemble_all(a) -> dict:
    if a.from_reference:
        records = {m: T.load_reference(m) for m in T1.MODEL_KEYS}
        out = a.out
    else:
        if not a.run_dir:
            raise SystemExit("--run-dir (the directory that holds <model>/ of every model) or --from-reference is required")
        records = {m: T.collect(os.path.join(a.run_dir, m), m) for m in T1.MODEL_KEYS}
        out = a.out or os.path.join(a.run_dir, "table2")
    table = T.assemble(records)
    finish(a, table, out)
    return table


def finish(a, table: dict, out: str | None) -> None:
    print(T.to_markdown(table))
    if out:
        paths = T.write_outputs(table, out)
        print("written:", ", ".join(paths.values()))
    if a.check:
        report(T.compare(table, {k: v for k, v in T.load_paper().items() if k in table}))


def report(cmp: list[dict]) -> None:
    n = {s: sum(c["status"] == s for c in cmp) for s in ("matched", "mismatch", "missing")}
    print(f"against the printed table: {n['matched']} cells matched, {n['mismatch']} differ, {n['missing']} not available")
    for c in cmp:
        if c["status"] == "mismatch":
            print(f"  {c['model']} {c['column']} {c['field']}: {T.fmt(c['got'], c['field'])} vs printed "
                  f"{T.fmt(c['paper'], c['field'])}")
    missing = sorted({(c["model"], c["column"]) for c in cmp if c["status"] == "missing"})
    if missing:
        print("  not available: " + ", ".join(f"{m} {v}" for m, v in missing))


def main(a) -> None:
    if a.from_reference and a.model != "all":
        finish(a, T.assemble({a.model: T.load_reference(a.model)}), a.out)
    elif a.model == "all":
        assemble_all(a)
    else:
        run_model(a)


__all__ = ["STAGES", "add_parser", "assemble_all", "calibrate", "dense_tag", "main", "run_model"]
