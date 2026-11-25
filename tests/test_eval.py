from __future__ import annotations

import contextlib
import math
import shutil
import statistics

import numpy as np
import pytest
from scipy import ndimage

from decoupattn.eval import metrics as M
from decoupattn.eval.frames import find_video, load_video, save_video
from decoupattn.eval.timing import BLOCKS, REPS, WARMUP, time_model

RNG = np.random.default_rng(1000)


def video(t=3, h=24, w=32):
    return RNG.integers(0, 256, (t, h, w, 3), dtype=np.uint8)


def test_psnr():
    a = np.zeros((2, 8, 8, 3), np.uint8)
    assert M.psnr(a + 1, a) == pytest.approx(10 * math.log10(255 ** 2))
    assert M.psnr(a, a) == pytest.approx(10 * math.log10(255 ** 2 / M.MSE_FLOOR))
    b = a.copy()
    b[0] = 4
    assert M.psnr(b, a) == pytest.approx(10 * math.log10(255 ** 2 / 8))


def ssim_reference(a, b, window):
    if window == "gaussian":
        def filt(x):
            return ndimage.gaussian_filter(x, 1.5, truncate=3.5, mode="reflect")
        win, cn = 11, 1.0
    else:
        def filt(x):
            return ndimage.uniform_filter(x, 7, mode="reflect")
        win, cn = 7, 49 / 48
    c1, c2, pad = (0.01 * 255) ** 2, (0.03 * 255) ** 2, (win - 1) // 2
    out = []
    for fa, fb in zip(a.astype(np.float64), b.astype(np.float64)):
        s = []
        for c in range(3):
            x, y = fa[..., c], fb[..., c]
            ux, uy = filt(x), filt(y)
            vx, vy, vxy = cn * (filt(x * x) - ux * ux), cn * (filt(y * y) - uy * uy), cn * (filt(x * y) - ux * uy)
            sm = (2 * ux * uy + c1) * (2 * vxy + c2) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
            s.append(sm[pad:-pad, pad:-pad].mean())
        out.append(np.mean(s))
    return np.array(out)


@pytest.mark.parametrize("window", ["gaussian", "uniform"])
def test_ssim(window):
    a = video()
    noisy = np.clip(a.astype(int) + RNG.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    assert np.allclose(M.ssim_frames(a, a, window), 1.0)
    got = M.ssim_frames(noisy, a, window)
    assert (got < 1).all()
    assert np.allclose(got, ssim_reference(noisy, a, window), atol=1e-10)


def test_clip_metrics_and_summary():
    a = video()
    r = M.clip_metrics(a, a)
    assert r["ssim"] == pytest.approx(1.0) and "lpips" not in r
    rows = [{"clip": "c0", "psnr": 30.0, "ssim": 0.9}, {"clip": "c1", "psnr": 20.0, "ssim": 0.8},
            {"clip": "c2", "psnr": 25.0, "ssim": 0.7}]
    s = M.summarize(rows)
    assert s["n"] == 3 and s["psnr"] == pytest.approx(25.0) and s["psnr_se"] == pytest.approx(5 / math.sqrt(3))
    assert (s["worst"], s["worst_id"]) == (20.0, "c1") and "lpips" not in s


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("name", ["clip.mkv", "clip"])
def test_frames_lossless(tmp_path, name):
    v = video(5, 32, 48)
    p = str(tmp_path / name)
    save_video(p, v)
    assert find_video(str(tmp_path / "clip")) == p
    assert np.array_equal(load_video(p), v)


class FakeAdapter:
    def __init__(self):
        self.timer = None
        self.log = []

    def generate(self, clip, hook, seed):
        self.log.append("method" if hook is not None else "dense")
        for i in range(4):
            with (self.timer(f"s{i}") if self.timer else contextlib.nullcontext()):
                sum(range(20000 if hook is None else 10000))


class CountingHook:
    def __init__(self):
        self.resets = self.builds = 0

    def reset(self):
        self.resets += 1

    def build_all(self):
        self.builds += 1


def test_timing_protocol():
    ad, built = FakeAdapter(), []
    res = time_model(ad, lambda: built.append(CountingHook()) or built[-1], "c0", log=lambda *_: None)
    n = WARMUP + REPS
    assert BLOCKS == ("dense", "dense", "method", "dense") and (WARMUP, REPS) == (1, 1)
    assert ad.log == ["dense"] * 2 * n + ["method"] * n + ["dense"] * n
    assert len(built) == 1 and built[0].resets == built[0].builds == n
    assert [r["row"] for r in res["rows"]] == list(BLOCKS)
    assert all(len(r["call_ms"]) == 4 and r["region"] == "dit_calls" and len(r["runs_ms"]) == REPS for r in res["rows"])
    m = res["rows"][2]
    assert res["method_ms"] == m["latency_ms"] == pytest.approx(statistics.fmean(m["runs_ms"]))
    dense = [r["latency_ms"] for r in res["rows"] if r["row"] == "dense"]
    assert len(dense) == 3 and res["dense_ms"] == pytest.approx(statistics.fmean(dense))
    assert res["speedup"] == pytest.approx(statistics.fmean(dense) / res["method_ms"])


def test_latency_as_the_sum_of_dit_calls():
    import time

    class Swapping(FakeAdapter):
        latency = "calls"

        def generate(self, clip, hook, seed):
            for i in range(4):
                if i == 2:
                    time.sleep(0.05)
                with self.timer(f"s{i}"):
                    sum(range(20000))

    from decoupattn.eval.timing import generate_timed
    from decoupattn.models.wan2_2 import Wan22
    r = generate_timed(Swapping(), None, "c0")
    assert r["region"] == "dit_call_sum" and r["latency_ms"] == pytest.approx(sum(c for _, c in r["call_ms"]))
    assert r["latency_ms"] < 50
    assert generate_timed(FakeAdapter(), None, "c0")["region"] == "dit_calls"
    assert Wan22.latency == "calls"


def test_timing_without_dit_timer():
    class Plain:
        timer = None

        def generate(self, clip, hook, seed):
            sum(range(10000))
    res = time_model(Plain(), lambda: None, "c0", reps=2, log=lambda *_: None)
    assert all(r["region"] == "generate" and r["latency_ms"] > 0 and len(r["runs_ms"]) == 2 for r in res["rows"])


def test_timing_excludes_work_between_calls():
    import time

    from decoupattn.eval.timing import generate_timed

    class Model:
        timer = None

        def generate(self, clip, hook, seed):
            for i in range(3):
                with self.timer.excluded():
                    time.sleep(0.1)
                with self.timer(f"s{i}"):
                    sum(range(20000))

    r = generate_timed(Model(), None, "c0")
    span = sum(ms for _, ms in r["call_ms"])
    assert r["excluded_ms"] >= 190 and r["latency_ms"] < span + 50
