from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch

from ..config import data_root
from ._common import add_path, upstream_path

CONFIG = "configs/wan_distilled_{kind}_480p.yaml"


def realised_density(skip: dict, similarity: dict, grid, threshold: float, anchor_rows: int) -> dict:
    F_, Hp, Wp = grid
    S = F_ * Hp * Wp
    M, N = skip["fa_block_size"]
    raw = skip["skip_lists_dict"]
    if skip.get("original_k_dim") is not None:
        ktiles = skip["original_k_dim"] - 1
    else:
        ktiles = max(t.shape[-1] for t in raw.values()) - 1
    rep_exec = math.ceil(F_ * anchor_rows * Wp / M) * M * S / S ** 2
    own = real = n = 0
    for key, t in raw.items():
        rep = {k[-1] for k, s in similarity.items() if tuple(k[:-1]) == tuple(key) and s >= threshold}
        ranges = t.to(torch.int64)[..., 1:]
        if ranges.shape[-1] % 2:
            ranges = torch.cat([ranges, torch.zeros_like(ranges[..., :1])], -1)
        ranges = ranges.view(*ranges.shape[:-1], -1, 2)
        tiles = (ranges[..., 1] - ranges[..., 0]).abs().sum(-1)[0]
        share = tiles.float().sum(-1) / (ktiles * tiles.shape[-1])
        for h in range(t.shape[1]):
            if h in rep:
                own, real = own + anchor_rows / Hp, real + rep_exec
            else:
                own, real = own + float(share[h]), real + float(tiles[h].sum()) * M * N / S ** 2
            n += 1
    return {"maps": len(raw), "calls_x_heads": n, "calibatt_density": own / n, "exec_density": real / n,
            "repetitive_share": sum(1 for s in similarity.values() if s >= threshold) / max(1, len(similarity))}


def density_of(calib_dir: str, grid, threshold: float, anchor_rows: int) -> dict:
    skip = torch.load(os.path.join(calib_dir, "skip_lists.pt"), map_location="cpu", weights_only=True)
    sim = torch.load(os.path.join(calib_dir, "similarity_map.pt"), map_location="cpu", weights_only=True)
    return realised_density(skip, sim["similarity_map"], grid, threshold, anchor_rows)


def calib_prompts(lines: list[str], picks: list[int], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[i] for i in picks) + "\n")


def calib_config(base: str, prompts: str, n: int, shift: float, out_dir: str, logs: str) -> str:
    edits = [("data_path: data/MovieGenVideoBench.txt", f'data_path: "{prompts}"'),
             ("data_num_samples: 16", f'data_num_samples: {n}\noutput_folder: "{logs}"'),
             ("epsilon_start: 0.863", f"epsilon_start: {0.863 - shift:.4f}"),
             ("epsilon_floor: 0.763", f"epsilon_floor: {0.763 - shift:.4f}"),
             ('output_dir: "calibration/wan_distilled_480p"', f'output_dir: "{out_dir}"')]
    for old, new in edits:
        if base.count(old) != 1:
            raise ValueError(f"the official calibration config has no line `{old}`")
        base = base.replace(old, new)
    return base


def model_env(adapter, s: dict) -> dict:
    env = dict(os.environ)
    local = os.path.join(data_root(), s["model_dir"])
    if "CALIBATT_MODEL_PATH" not in env and os.path.isdir(local):
        env["CALIBATT_MODEL_PATH"] = local
    env.setdefault("CALIBATT_DIT_CKPT", adapter.cfg["checkpoints"]["lightx2v"])
    env["PYTHONPATH"] = os.pathsep.join([upstream_path("LightX2V")] + [p for p in [env.get("PYTHONPATH")] if p])
    return env


def calibrate(adapter, s: dict, work: str, shift: float, picks: list[int], name: str, log=print) -> str:
    out = os.path.join(work, name)
    if os.path.exists(os.path.join(out, "skip_lists.pt")) and os.path.exists(os.path.join(out, "similarity_map.pt")):
        return out
    repo = upstream_path("ml-calibatt")
    src = os.environ.get("CALIBATT_PROMPTS") or os.path.join(repo, "data", "MovieGenVideoBench.txt")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} is missing: the official setup.sh downloads MovieGenVideoBench.txt there "
                                "(or set CALIBATT_PROMPTS to a copy)")
    with open(src, encoding="utf-8") as f:
        lines = [x.rstrip() for x in f]
    prompts = os.path.join(work, f"{name}_prompts.txt")
    calib_prompts(lines, picks, prompts)
    with open(os.path.join(repo, CONFIG.format(kind="calib")), encoding="utf-8") as f:
        cfg = calib_config(f.read(), prompts, len(picks), shift, out, os.path.join(work, f"{name}_samples"))
    cfg_path = os.path.join(work, f"{name}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg)
    log(f"calibatt: official calibration on {len(picks)} prompts, epsilon shift {shift:.4f} -> {out}")
    with open(os.path.join(work, f"{name}.log"), "w") as lg:
        subprocess.run([s.get("python") or sys.executable, "wan_distilled_run.py", "--config_path", cfg_path], cwd=repo,
                       env=model_env(adapter, s), stdout=lg, stderr=subprocess.STDOUT, check=True)
    return out


def search_shift(adapter, s: dict, work: str, log=print) -> float:
    lo, hi = s["shift_range"]
    grid = tuple(adapter.latent_grid)
    hist = []
    for _ in range(int(s["shift_iters"])):
        mid = (lo + hi) / 2
        d = density_of(calibrate(adapter, s, work, mid, s["calib_lines"][::8], f"search_d{mid:.4f}", log), grid,
                       s["repetitive_threshold"], s["anchor_rows"])["exec_density"]
        hist.append({"shift": mid, "exec_density": d})
        log(f"calibatt: epsilon shift {mid:.4f}: executed density {d:.4f}")
        lo, hi = (mid, hi) if d > s["target_density"] else (lo, mid)
    with open(os.path.join(work, "shift_search.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return min(hist, key=lambda r: abs(r["exec_density"] - s["target_density"]))["shift"]


def prepare(adapter, s: dict, work: str, log=print) -> dict:
    record = os.path.join(work, "calibration.json")
    if os.path.exists(record):
        with open(record) as f:
            return json.load(f)
    os.makedirs(work, exist_ok=True)
    shift = search_shift(adapter, s, work, log) if s.get("rematch") else float(s["epsilon_shift"])
    name = f"all{len(s['calib_lines'])}_d{shift:.4f}"
    out = calibrate(adapter, s, work, shift, s["calib_lines"], name, log)
    res = {"dir": name, "shift": shift, "prompts": len(s["calib_lines"]),
           **density_of(out, tuple(adapter.latent_grid), s["repetitive_threshold"], s["anchor_rows"])}
    with open(record + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(record + ".tmp", record)
    log(f"calibatt: executed density {res['exec_density']:.4f} over {res['maps']} calls")
    return res


class Pipeline:
    def __init__(self, adapter, s: dict, work: str):
        os.environ["FLASH_ATTENTION_SKIP_IMPORT_FA3"] = "1"
        add_path(upstream_path("LightX2V"))
        add_path(upstream_path("ml-calibatt"))
        from lightx2v import LightX2VPipeline
        from lightx2v.utils.profiler import _ProfilingContextL1
        from omegaconf import OmegaConf
        _ProfilingContextL1.__enter__ = lambda self: self
        _ProfilingContextL1.__exit__ = lambda self, *args: False
        self.s, self.work = s, work
        cfg = self.cfg = OmegaConf.load(upstream_path("ml-calibatt", CONFIG.format(kind="infer")))
        v = cfg.video
        env = model_env(adapter, s)
        if "CALIBATT_MODEL_PATH" in env:
            model_path = env["CALIBATT_MODEL_PATH"]
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(cfg["model_id"], revision="a064a6c71f5be440641209c07bf2a5ce7a2ff5e4")
        self.pipe = LightX2VPipeline(model_path=model_path, model_cls=cfg.get("model_cls", "wan2.1_distill"), task="t2v",
                                     dit_original_ckpt=env["CALIBATT_DIT_CKPT"])
        self.pipe.create_generator(attn_mode=s["attn_mode"], infer_steps=v.infer_steps, height=v.height, width=v.width,
                                   num_frames=v.num_frames, guidance_scale=v.guidance_scale,
                                   sample_shift=v.sample_shift, denoising_step_list=list(v.denoising_step_list))
        self.negative = cfg.get("negative_prompt", "")
        self.steps = v.infer_steps
        self.phases = [b.compute_phases[0] for b in self.pipe.runner.model.transformer_weights.blocks]
        self.dense_attn = [p.self_attn_1 for p in self.phases]
        self.sparse_attn = self.backend = None
        self.events: list = []
        model_cls = type(self.pipe.runner.model)
        infer, events = model_cls.infer, self.events

        def timed_infer(model, *args, **kw):
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = infer(model, *args, **kw)
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            events.append((e0, e1))
            return out
        model_cls.infer = timed_infer

    def use(self, dense: bool) -> None:
        if not dense and self.backend is None:
            from CalibAtt.calibatt import CalibAttBackend
            from CalibAtt.processors import lightx2v as processor_mod
            from lite_attention import LiteAttention
            with open(os.path.join(self.work, "calibration.json")) as f:
                calib = os.path.join(self.work, json.load(f)["dir"])
            v, runner = self.cfg.video, self.pipe.runner
            self.backend = CalibAttBackend(
                pipeline=self.pipe, processor_mod=processor_mod, cfg=False, num_frames=v.latent_num_frames,
                height_patches=v.latent_height_patches, width_patches=v.latent_width_patches,
                num_layers=len(self.phases), num_steps=v.infer_steps, num_heads=runner.model.config["num_heads"],
                fa_block_size=list(LiteAttention.get_MN(v.head_dim, torch.bfloat16)),
                output_dir=os.path.join(self.work, "logs"), device=torch.device("cuda"), verbose=False,
                skip_lists_path=os.path.join(calib, "skip_lists.pt"),
                similarity_map_path=os.path.join(calib, "similarity_map.pt"),
                repetitive_threshold=self.s["repetitive_threshold"], anchor_rows=self.s["anchor_rows"])
            self.sparse_attn = [p.self_attn_1 for p in self.phases]
        for p, a in zip(self.phases, self.dense_attn if dense else self.sparse_attn):
            p.self_attn_1 = a

    def run(self, prompt: str, seed: int, dense: bool) -> dict:
        self.use(dense)
        if not dense:
            self.backend.reset_logs()
        self.events.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            self.pipe.generate(seed=seed, prompt=prompt, negative_prompt=self.negative, save_result_path=None,
                               return_result_tensor=True)
        torch.cuda.synchronize()
        if len(self.events) != self.steps:
            raise RuntimeError(f"{len(self.events)} transformer calls timed, {self.steps} sampler steps expected")
        video = self.pipe.runner.gen_video_final
        return {"frames": (video.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(),
                "dit_ms": self.events[0][0].elapsed_time(self.events[-1][1]), "wall_s": time.time() - t0}


def stats(work: str) -> dict:
    with open(os.path.join(work, "calibration.json")) as f:
        c = json.load(f)
    return {"method": "calibatt", "exec_density": c["exec_density"], "calibatt_density": c["calibatt_density"],
            "sparse_calls": c["maps"], "dense_calls": 0, "epsilon_shift": c["shift"]}


def generate(pl: Pipeline, adapter, clips: list[str], out_dir: str, seed: int, dense: bool, log=print) -> None:
    from ..eval.frames import find_video, save_video
    todo = [c for c in clips if find_video(os.path.join(out_dir, c)) is None]
    for i, c in enumerate(todo):
        r = pl.run(adapter.prompt(c), seed, dense)
        save_video(os.path.join(out_dir, f"{c}.mkv"), r["frames"])
        with open(os.path.join(out_dir, f"{c}.json"), "w") as f:
            json.dump({"clip": c, "seed": seed, "frames": list(r["frames"].shape), "wall_s": r["wall_s"],
                       "dit_ms": r["dit_ms"], "pipeline": "LightX2V (official CalibAtt inference path)",
                       "stats": None if dense else stats(pl.work)}, f, indent=1)
        log(f"{os.path.basename(out_dir)} {c} {r['wall_s']:.1f} s | {i + 1}/{len(todo)}")


def time_rows(pl: Pipeline, adapter, clip: str, seed: int, log=print, reps: int | None = None) -> dict:
    from ..eval.timing import BLOCKS, REPS, WARMUP, summarize_blocks
    reps = reps or REPS
    prompt, rows = adapter.prompt(clip), []
    for name in BLOCKS:
        dense = name == "dense"
        for _ in range(WARMUP):
            pl.run(prompt, seed, dense)
        lat = [pl.run(prompt, seed, dense)["dit_ms"] for _ in range(reps)]
        rows.append({"row": name, "latency_ms": statistics.fmean(lat), "runs_ms": lat, "build_ms": 0.0,
                     "region": "dit_calls", "stats": None if dense else stats(pl.work)})
        log(f"timing {name}: {rows[-1]['latency_ms']:.0f} ms (mean of {reps})")
        torch.cuda.empty_cache()
    return {"clip": clip, "seed": seed, "reps": reps, "warmup": WARMUP,
            "protocol": f"LightX2V pipeline in one process: dense, dense, CalibAtt, dense; {WARMUP} untimed warm-up + "
                        f"{reps} timed generation(s) per block; speedup = mean of the three dense latencies / CalibAtt "
                        "latency; latency = first transformer call start -> last end (CUDA events)",
            "rows": rows, **summarize_blocks(rows), "gpu": torch.cuda.get_device_name()}
