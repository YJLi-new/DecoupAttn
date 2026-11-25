from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .constants import ESTIMATORS, SEED
from .models import MODELS

SCHEDULES = ("frozen", "reassign", "rebuild")
BASELINES = ("calibatt", "svoo", "svg2", "svgear", "sol", "pisa", "sparsepr")


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def adapter_for(model: str, dense_backend: str | None = None):
    from .config import load_config
    from .models import get_adapter
    cfg = load_config(model)
    if dense_backend:
        cfg["model"]["dense_backend"] = dense_backend
    return get_adapter(model, cfg)


def baseline_for(a, adapter):
    from .baselines import get_baseline
    b = get_baseline(a.baseline, a.model)
    adapter.load()
    b.prepare(adapter, a.run_dir, log=log)
    return b


def overrides(a) -> dict:
    ov = {}
    if getattr(a, "schedule", None):
        ov["schedule"] = a.schedule
    if getattr(a, "tied", None) is not None:
        ov["tied"] = a.tied
    if getattr(a, "estimator", None):
        ov["estimator"] = a.estimator
    return ov


def make_runtime(adapter, plan, **ov):
    from .plan import load_plan
    from .runtime import runtime_for
    rt = runtime_for(adapter, load_plan(plan) if isinstance(plan, str) else plan, **ov)
    rt.build_all()
    return rt


def _positive(s: str) -> int:
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {s}")
    return n


def _stats(hook):
    if hook is None:
        return None
    s = getattr(hook, "summary", None)
    return s() if callable(s) else getattr(hook, "stats", None)


def generate_clips(adapter, clips, out, hook=None, seed=SEED, meta=None) -> list[str]:
    from .eval.frames import find_video, save_video
    todo = [c for c in clips if find_video(os.path.join(out, c)) is None]
    t0 = time.time()
    for i, c in enumerate(todo):
        if hasattr(hook, "reset"):
            hook.reset()
        tc = time.time()
        frames = adapter.decode(adapter.generate(c, hook, seed))
        save_video(os.path.join(out, f"{c}.mkv"), frames)
        write_json(os.path.join(out, f"{c}.json"), {**(meta or {}), "clip": c, "seed": seed, "frames": list(frames.shape),
                                                    "wall_s": time.time() - tc, "stats": _stats(hook)})
        el = (time.time() - t0) / (i + 1)
        log(f"{os.path.basename(out)} {c} {time.time() - tc:.1f} s | {i + 1}/{len(todo)} "
            f"ETA {(len(todo) - i - 1) * el / 60:.0f} min")
    return [os.path.join(out, c) for c in clips]


def block_shape_callbacks(adapter, run_dir, seed=SEED):
    from .eval.frames import find_video, load_video
    from .eval.metrics import psnr
    from .eval.timing import time_block
    root = os.path.join(run_dir, "block_shape", "videos")

    def val_psnr(plan, clips):
        ref = generate_clips(adapter, clips, os.path.join(root, "dense"), None, seed)
        tag = "x".join(str(x) for x in plan.block_shape)
        out = generate_clips(adapter, clips, os.path.join(root, tag), make_runtime(adapter, plan), seed)
        return [psnr(load_video(find_video(p)), load_video(find_video(r))) for p, r in zip(out, ref)]

    def time_ms(plan):
        clip = adapter.clips("test")[0]
        return time_block(adapter, lambda: make_runtime(adapter, plan), clip, seed, log=log)["latency_ms"]

    return val_psnr, time_ms


def cmd_calibrate(a) -> None:
    from .planner.pipeline import STAGES, calibrate
    adapter = adapter_for(a.model)
    val_psnr, time_ms = block_shape_callbacks(adapter, a.run_dir, a.seed)
    stages = tuple(a.stages.split(",")) if a.stages else STAGES
    plan = calibrate(adapter, a.run_dir, stages=stages, val_psnr=val_psnr, time_ms=time_ms, device=a.device)
    log("stages done:", ",".join(stages), "| plan:", os.path.join(a.run_dir, "plan.npz") if plan is not None else "-")


def cmd_variants(a) -> None:
    from .planner.pipeline import calibrate_variants
    from .planner.variants import COPIED, RESOLVED
    names = tuple(a.names.split(",")) if a.names else RESOLVED + COPIED
    plans = calibrate_variants(adapter_for(a.model), a.run_dir, names=names, device=a.device)
    log("variants:", ", ".join(os.path.join(a.run_dir, "variants", f"{n}.npz") for n in plans))


def generate_baseline(adapter, baseline, run_dir: str, clips: list[str], seed: int = SEED, tag: str | None = None) -> None:
    out = os.path.join(run_dir, "videos", tag or baseline.name)
    log(f"{adapter.name} {baseline.name}: {len(clips)} clips")
    if baseline.standalone:
        baseline.generate(adapter, clips, os.path.join(run_dir, "videos", baseline.reference), seed, dense=True, log=log)
        baseline.generate(adapter, clips, out, seed, log=log)
    else:
        generate_clips(adapter, clips, out, baseline.hook(adapter, run_dir), seed,
                       {"model": adapter.name, "tag": baseline.name, "baseline": baseline.settings})


def cmd_generate(a) -> None:
    adapter = adapter_for(a.model, a.dense_backend)
    clips = a.clips.split(",") if a.clips else adapter.clips(a.split)
    if a.baseline:
        generate_baseline(adapter, baseline_for(a, adapter), a.run_dir, clips, a.seed, a.tag)
        return
    plan = None if a.dense else (a.plan or os.path.join(a.run_dir, "plan.npz"))
    tag = a.tag or ((f"dense_{a.dense_backend}" if a.dense_backend else "dense") if plan is None else "decoupattn")
    ov = overrides(a)
    hook = None
    if plan is not None:
        adapter.load()
        hook = make_runtime(adapter, plan, **ov)
    log(f"{a.model} {tag}: {len(clips)} clips")
    generate_clips(adapter, clips, os.path.join(a.run_dir, "videos", tag), hook, a.seed,
                   {"model": a.model, "tag": tag, "plan": plan, "overrides": ov})


def evaluate_tag(adapter, run_dir: str, tag: str, ref: str, clips: list[str], device: str = "cuda", lpips_net=None,
                 vbench_dir: str | None = None) -> dict:
    from .eval import metrics as M
    from .eval.frames import find_video, load_video
    window = adapter.cfg.get("eval", {}).get("ssim", "uniform")
    root = os.path.join(run_dir, "videos")
    rows = []
    for c in clips:
        pred = load_video(find_video(os.path.join(root, tag, c)))
        ref_v = load_video(find_video(os.path.join(root, ref, c)))
        r = dict(clip=c, **M.clip_metrics(pred, ref_v, window, lpips_net, device))
        rows.append(r)
        log(tag, c, json.dumps({k: round(v, 4) for k, v in r.items() if k != "clip"}))
    assert len(rows) == len(clips), (len(rows), len(clips))
    res = {"model": adapter.name, "tag": tag, "ref": ref, "ssim_window": window, "summary": M.summarize(rows),
           "clips": rows}
    if vbench_dir:
        from .eval import vbench
        mp4 = os.path.join(run_dir, "mp4", tag)
        vbench.write_mp4s(os.path.join(root, tag), mp4, clips)
        res["vbench"] = vbench.run(mp4, tag, os.path.join(run_dir, "vbench"),
                                   os.path.join(vbench_dir, "vbench", "VBench_full_info.json"), device=device)
    write_json(os.path.join(run_dir, "metrics", f"{tag}.json"), res)
    log(tag, "summary", json.dumps(res["summary"]))
    return res


def cmd_evaluate(a) -> None:
    from .eval import metrics as M
    adapter = adapter_for(a.model)
    clips = a.clips.split(",") if a.clips else adapter.clips(a.split)
    lp = None if a.no_lpips else M.lpips_model(a.device)
    evaluate_tag(adapter, a.run_dir, a.tag, a.ref, clips, a.device, lp, a.vbench)


def time_baseline(adapter, baseline, run_dir: str, clip: str, seed: int = SEED, reps: int | None = None) -> dict:
    from .eval.timing import REPS, time_model
    reps = reps or REPS
    if baseline.standalone:
        res = baseline.time(adapter, clip, seed, log=log, run_dir=run_dir, reps=reps)
        res["own_pipeline"] = True
    else:
        res = time_model(adapter, lambda: baseline.timing_hook(adapter, run_dir), clip, seed, reps=reps, log=log)
    res.update(model=adapter.name, baseline=baseline.name, settings=baseline.settings,
               timing_settings=baseline.timing_settings)
    return res


def cmd_time(a) -> None:
    from .eval.timing import REPS, time_model
    reps = a.reps or REPS
    adapter = adapter_for(a.model)
    adapter.load()
    clip = a.clip or adapter.clips("test")[0]
    if a.baseline:
        res = time_baseline(adapter, baseline_for(a, adapter), a.run_dir, clip, a.seed, reps)
        out = a.out or os.path.join(a.run_dir, "timing", f"{a.baseline}.json")
    else:
        plan = a.plan or os.path.join(a.run_dir, "plan.npz")
        ov = overrides(a)
        res = time_model(adapter, lambda: make_runtime(adapter, plan, **ov), clip, a.seed, reps=reps, log=log)
        res.update(model=a.model, plan=plan, overrides=ov)
        out = a.out or os.path.join(a.run_dir, "timing.json")
    write_json(out, res)
    log(f"speedup {res['speedup']:.3f} (dense {res['dense_ms']:.0f} ms, method {res['method_ms']:.0f} ms)")


def cmd_table1(a) -> None:
    from . import table1
    table1.main(a)


def cmd_table2(a) -> None:
    from . import table2
    table2.main(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="decoupattn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, choices=sorted(MODELS))
        p.add_argument("--run-dir", required=True)
        p.add_argument("--seed", type=int, default=SEED)
        return p

    def ablation(p):
        p.add_argument("--schedule", choices=SCHEDULES, help="cluster schedule (default: the plan's reassignment rule)")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--tied", dest="tied", action="store_true", default=None,
                       help="group the compensation by the key blocks (ablation Tied)")
        g.add_argument("--untied", dest="tied", action="store_false", help="semantic clusters even for a Tied plan")
        p.add_argument("--estimator", choices=ESTIMATORS, help="one estimator for every call (default: the plan's)")

    p = common(sub.add_parser("calibrate", help="compile the plan of a model on its calibration clips"))
    p.add_argument("--stages", help="comma-separated planner stages (default: all, in order)")
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_calibrate)
    p = common(sub.add_parser("variants", help="plans of the component-ablation variants (after calibrate)"))
    p.add_argument("--names", help="comma-separated variants (default: all)")
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_variants)
    p = common(sub.add_parser("generate", help="generate and store the videos of a split (dense, a plan or a baseline)"))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plan", help="plan .npz (default <run-dir>/plan.npz)")
    g.add_argument("--dense", action="store_true", help="dense attention (the reference)")
    g.add_argument("--baseline", choices=BASELINES, help="a baseline of Table 1, from its official code")
    p.add_argument("--dense-backend", help="dense backend instead of the configuration's (with --dense: tag "
                                           "dense_<backend>)")
    p.add_argument("--split", default="test")
    p.add_argument("--clips", help="comma-separated clip ids instead of the split")
    p.add_argument("--tag", help="output tag (default: dense / decoupattn)")
    ablation(p)
    p.set_defaults(fn=cmd_generate)
    p = common(sub.add_parser("evaluate", help="PSNR / SSIM / LPIPS of a tag against the dense outputs"))
    p.add_argument("--tag", default="decoupattn")
    p.add_argument("--ref", default="dense")
    p.add_argument("--split", default="test")
    p.add_argument("--clips")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--vbench", help="VBench checkout: also score the six VBench dimensions")
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_evaluate)
    p = common(sub.add_parser("time", help="timing protocol: dense, dense, method, dense in one process"))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plan", help="plan .npz (default <run-dir>/plan.npz)")
    g.add_argument("--baseline", choices=BASELINES, help="time a baseline of Table 1 (output timing/<name>.json)")
    p.add_argument("--clip", help="clip (default: the first test clip)")
    p.add_argument("--out", help="output json (default <run-dir>/timing.json)")
    p.add_argument("--reps", type=_positive, help="timed generations per block (default 1)")
    ablation(p)
    p.set_defaults(fn=cmd_time)
    from . import table1, table2
    table1.add_parser(sub).set_defaults(fn=cmd_table1)
    table2.add_parser(sub).set_defaults(fn=cmd_table2)
    return ap


def main(argv=None) -> None:
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
