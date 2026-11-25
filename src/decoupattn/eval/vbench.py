from __future__ import annotations

import glob
import json
import os

import numpy as np

from .frames import find_video, load_video

DIMS = ("subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness", "imaging_quality",
        "aesthetic_quality")


def write_mp4s(src: str, out: str, clips: list[str], fps: int = 16) -> list[str]:
    import imageio
    os.makedirs(out, exist_ok=True)
    paths = []
    for c in clips:
        p = os.path.join(out, f"{c}.mp4")
        if not os.path.exists(p):
            v = load_video(find_video(os.path.join(src, c)))
            assert v.dtype == np.uint8 and v.ndim == 4, (c, v.dtype, v.shape)
            imageio.mimwrite(p + ".tmp.mp4", v, fps=fps, quality=9, macro_block_size=1, ffmpeg_params=["-threads", "4"])
            os.replace(p + ".tmp.mp4", p)
        paths.append(p)
    return paths


def run(videos: str, name: str, out: str, full_info: str, dims=DIMS, device: str = "cuda",
        memory_cap_gib: float | None = 12.0) -> dict:
    import torch
    from vbench import VBench
    if memory_cap_gib and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(memory_cap_gib * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory, 0)
    os.makedirs(out, exist_ok=True)
    vb = VBench(torch.device(device), full_info, out)
    vb.evaluate(videos_path=videos, name=name, dimension_list=list(dims), local=True, read_frame=False,
                mode="custom_input", imaging_quality_preprocessing_mode="longer")
    return read_results(out, name)


def read_results(out: str, name: str) -> dict:
    (path,) = glob.glob(os.path.join(out, f"{name}_eval_results.json"))
    with open(path) as f:
        res = json.load(f)
    scores = {d: float(v[0]) for d, v in res.items()}
    scores["mean"] = float(np.mean([scores[d] for d in DIMS if d in scores]))
    return scores
