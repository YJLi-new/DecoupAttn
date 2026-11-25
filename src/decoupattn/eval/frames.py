from __future__ import annotations

import os
import subprocess

import numpy as np

VIDEO_EXT = (".mkv", ".npy")


def save_video(path: str, video: np.ndarray, fps: int = 16) -> None:
    video = np.ascontiguousarray(video, dtype=np.uint8)
    _, H, W, _ = video.shape
    if not os.path.splitext(path)[1]:
        os.makedirs(path, exist_ok=True)
        out = [os.path.join(path, "%04d.png")]
        enc = ["-start_number", "0", "-c:v", "png", "-pix_fmt", "rgb24"]
        tmp = None
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.mkv"
        out = [tmp]
        enc = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-threads", "4"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-"] + enc + out
    subprocess.run(cmd, input=video.tobytes(), check=True)
    if tmp:
        os.replace(tmp, path)


def _size(src: str) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(r[0]), int(r[1])


def load_video(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if os.path.isdir(path):
        src = os.path.join(path, "%04d.png")
        w, h = _size(os.path.join(path, "0000.png"))
        inp = ["-start_number", "0", "-i", src]
    else:
        w, h = _size(path)
        inp = ["-i", path]
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-threads", "4"] + inp + ["-f", "rawvideo", "-pix_fmt", "rgb24",
                                                                                    "-"], capture_output=True,
                         check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def find_video(stem: str) -> str | None:
    for ext in VIDEO_EXT:
        if os.path.exists(stem + ext):
            return stem + ext
    return stem if os.path.isfile(os.path.join(stem, "0000.png")) else None
