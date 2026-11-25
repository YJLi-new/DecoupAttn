from __future__ import annotations

import argparse
import os

from .constants import SEED
from .eval import table1 as T

STAGES = ("dense", "floor", "ours", "baselines", "metrics", "timing", "assemble")
GPU_STAGES = ("dense", "floor", "ours", "baselines", "timing")
DENSE_TAG = "dense"


def add_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser("table1", help="reproduce Table 1: one model's row, or the whole table from its rows",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=[*T.MODEL_KEYS, "all"],
                   help="a model (run its row) or 'all' (assemble the table from <run-dir>/<model>)")
    p.add_argument("--run-dir", help="run directory of the model; with --model all: the directory of the ten of them")
    p.add_argument("--plan", help="plan .npz of DecoupAttn (default <run-dir>/plan.npz)")
    p.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGES)} (default: all, in this order)")
    p.add_argument("--methods", help=f"comma-separated baselines among {','.join(T.BASELINES)}, or 'none' (default: all)")
    p.add_argument("--anchor", choices=T.ANCHORS, default="common",
                   help="dense latency of the speedups: one per model, shared by every method (default), or that of "
                        "each method's own timing process")
    p.add_argument("--clip", help="clip of the timing protocol (default: the first test clip)")
    p.add_argument("--from-reference", action="store_true",
                   help="assemble from the stored records of the paper's runs (data/table1/reference) instead of runs")
    p.add_argument("--check", action="store_true", help="compare every assembled cell with the printed table")
    p.add_argument("--out", help="output directory of the assembled table (default <run-dir>/table1)")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda")
    return p


def parse_list(text: str | None, allowed: tuple[str, ...], what: str) -> tuple[str, ...]:
    if text is None:
        return allowed
    if text.strip().lower() == "none":
        return ()
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in names if x not in allowed]
    if unknown:
        raise SystemExit(f"unknown {what}: {', '.join(unknown)} (choose among {', '.join(allowed)})")
    return tuple(x for x in allowed if x in names)


def set_dense_backend(adapter, backend: str) -> None:
    adapter.cfg["model"]["dense_backend"] = backend
    if "backend" in vars(adapter):
        adapter.backend = backend


def test_clips(adapter) -> list[str]:
    return adapter.clips("test")


def have_videos(run_dir: str, tag: str, clips: list[str]) -> bool:
    from .eval.frames import find_video
    return all(find_video(os.path.join(run_dir, "videos", tag, c)) is not None for c in clips)


def metric_jobs(run_dir: str, cfg: dict, methods: tuple[str, ...], references: dict[str, str],
                clips: list[str]) -> list[tuple[str, str]]:
    jobs = [(T.floor_tag(cfg), DENSE_TAG), (T.OURS_TAG, DENSE_TAG)] + [(m, references[m]) for m in methods]
    return [(tag, ref) for tag, ref in jobs if have_videos(run_dir, tag, clips) and have_videos(run_dir, ref, clips)]


def run_model(a) -> dict:
    from . import cli
    from .baselines import get_baseline
    from .config import load_config
    stages = parse_list(a.stages, STAGES, "stages")
    methods = parse_list(a.methods, T.BASELINES, "methods")
    if not a.run_dir:
        raise SystemExit("--run-dir is required")
    R = a.run_dir
    cfg = load_config(a.model, resolve=False)
    ref_backend, floor = T.reference_backend(cfg), cfg["model"]["floor_backend"]
    baselines = {m: get_baseline(m, a.model) for m in methods}
    adapter = None
    if any(s in stages for s in (*GPU_STAGES, "metrics")):
        adapter = cli.adapter_for(a.model, ref_backend)
    clips = test_clips(adapter) if adapter is not None else None

    if "dense" in stages:
        cli.log(f"table1 {a.model}: dense reference ({ref_backend}), {len(clips)} clips")
        cli.generate_clips(adapter, clips, os.path.join(R, "videos", DENSE_TAG), None, a.seed,
                           {"model": a.model, "tag": DENSE_TAG, "dense_backend": ref_backend})
    if "floor" in stages:
        cli.log(f"table1 {a.model}: floor ({floor})")
        set_dense_backend(adapter, floor)
        try:
            cli.generate_clips(adapter, clips, os.path.join(R, "videos", T.floor_tag(cfg)), None, a.seed,
                               {"model": a.model, "tag": T.floor_tag(cfg), "dense_backend": floor})
        finally:
            set_dense_backend(adapter, ref_backend)
    plan = a.plan or os.path.join(R, "plan.npz")
    if "ours" in stages and not os.path.exists(plan):
        cli.log(f"table1 {a.model}: no plan at {plan} (calibrate one, or pass --plan); the Ours column is left out")
    elif "ours" in stages:
        adapter.load()
        cli.generate_clips(adapter, clips, os.path.join(R, "videos", T.OURS_TAG), cli.make_runtime(adapter, plan), a.seed,
                           {"model": a.model, "tag": T.OURS_TAG, "plan": plan})
    if "baselines" in stages or ("timing" in stages and methods):
        adapter.load()
        for b in baselines.values():
            b.prepare(adapter, R, device=a.device, log=cli.log)
    if "baselines" in stages:
        for b in baselines.values():
            cli.generate_baseline(adapter, b, R, clips, a.seed)
    if "metrics" in stages:
        from .eval import metrics as M
        lp = None if a.no_lpips else M.lpips_model(a.device)
        for tag, ref in metric_jobs(R, cfg, methods, {m: b.reference for m, b in baselines.items()}, clips):
            if not os.path.exists(os.path.join(R, "metrics", f"{tag}.json")):
                cli.evaluate_tag(adapter, R, tag, ref, clips, a.device, lp)
    if "timing" in stages:
        from .eval.timing import time_model
        clip = a.clip or clips[0]
        out = os.path.join(R, "timing", f"{T.OURS_TAG}.json")
        if not os.path.exists(out) and os.path.exists(plan):
            res = time_model(adapter, lambda: cli.make_runtime(adapter, plan), clip, a.seed, log=cli.log)
            res.update(model=a.model, plan=plan, dense_backend=ref_backend)
            cli.write_json(out, res)
        for m, b in baselines.items():
            out = os.path.join(R, "timing", f"{m}.json")
            if not os.path.exists(out):
                res = cli.time_baseline(adapter, b, R, clip, a.seed)
                res["dense_backend"] = ref_backend
                cli.write_json(out, res)
    if "assemble" not in stages:
        return {}
    record = T.collect(R, a.model, cfg)
    table = T.assemble({a.model: record}, a.anchor)
    finish(a, table, a.out or os.path.join(R, "table1"))
    return table


def assemble_all(a) -> dict:
    from .config import load_config
    if a.from_reference:
        records = {m: T.load_reference(m) for m in T.MODEL_KEYS}
        out = a.out
    else:
        if not a.run_dir:
            raise SystemExit("--run-dir (the directory that holds <model>/ of every model) or --from-reference is required")
        records = {}
        for m in T.MODEL_KEYS:
            rec = T.collect(os.path.join(a.run_dir, m), m, load_config(m, resolve=False))
            if rec["methods"]:
                records[m] = rec
        out = a.out or os.path.join(a.run_dir, "table1")
    table = T.assemble(records, a.anchor)
    finish(a, table, out)
    return table


def finish(a, table: dict, out: str | None) -> None:
    print(T.to_markdown(table))
    if out:
        paths = T.write_outputs(table, out)
        print("written:", ", ".join(paths.values()))
    if a.check:
        report(T.compare({k: v for k, v in table.items()}, {k: v for k, v in T.load_paper().items() if k in table}))


def report(cmp: list[dict]) -> None:
    n = {s: sum(c["status"] == s for c in cmp) for s in ("matched", "mismatch", "missing")}
    print(f"against the printed table: {n['matched']} cells matched, {n['mismatch']} differ, {n['missing']} not available")
    summary = T.compare_summary(cmp)
    cols = (*T.COLUMNS, "delta")
    print("| model | " + " | ".join(T.LABELS.get(c, "Δ") for c in cols) + " |")
    print("| --- | " + " | ".join("---" for _ in cols) + " |")
    for model, row in summary.items():
        cells = []
        for c in cols:
            s = row.get(c)
            if s is None:
                cells.append("")
            elif s["mismatch"]:
                cells.append("; ".join(f"{f} {T.fmt(g, f)} vs {T.fmt(p, f)}" for f, g, p in s["mismatch"]))
            elif s["missing"]:
                cells.append("n/a" if not s["matched"] else f"ok, n/a: {','.join(s['missing'])}")
            else:
                cells.append("ok")
        print(f"| {model} | " + " | ".join(cells) + " |")


def main(a) -> None:
    if a.from_reference and a.model != "all":
        table = T.assemble({a.model: T.load_reference(a.model)}, a.anchor)
        finish(a, table, a.out)
    elif a.model == "all":
        assemble_all(a)
    else:
        run_model(a)


__all__ = ["STAGES", "add_parser", "assemble_all", "main", "parse_list", "run_model", "set_dense_backend"]
