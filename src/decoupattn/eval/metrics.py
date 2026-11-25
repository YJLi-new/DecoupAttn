from __future__ import annotations

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F

MSE_FLOOR = 1e-12


def _t(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))


def psnr(pred, ref) -> float:
    a, b = _t(pred).double(), _t(ref).double()
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(255.0 ** 2 / max(mse, MSE_FLOOR))


def _window(kind: str, device) -> tuple[torch.Tensor, float]:
    if kind == "uniform":
        w = 7
        return torch.full((1, 1, w, w), 1.0 / w ** 2, dtype=torch.float64, device=device), w * w / (w * w - 1.0)
    if kind == "gaussian":
        sigma, radius = 1.5, int(3.5 * 1.5 + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        return (g[:, None] * g[None, :])[None, None], 1.0
    raise ValueError(kind)


def _filter(t: torch.Tensor, k: torch.Tensor, shape) -> torch.Tensor:
    return F.conv2d(t.reshape(shape), k)


def ssim_frames(pred, ref, window: str = "gaussian", chunk: int = 9, device: str = "cpu") -> np.ndarray:
    k, cn = _window(window, device)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a_all, b_all = _t(pred), _t(ref)
    out = []
    for i in range(0, a_all.shape[0], chunk):
        x = a_all[i:i + chunk].to(device).permute(0, 3, 1, 2).double()
        y = b_all[i:i + chunk].to(device).permute(0, 3, 1, 2).double()
        n, c, h, w = x.shape
        f = functools.partial(_filter, k=k, shape=(n * c, 1, h, w))
        ux, uy = f(x), f(y)
        vx, vy, vxy = cn * (f(x * x) - ux * ux), cn * (f(y * y) - uy * uy), cn * (f(x * y) - ux * uy)
        sm = (2 * ux * uy + c1) * (2 * vxy + c2) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        out.append(sm.reshape(n, -1).mean(1))
    return torch.cat(out).cpu().numpy()


def lpips_model(device: str = "cuda"):
    import lpips
    return lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()


@torch.no_grad()
def lpips_frames(pred, ref, model, chunk: int = 9) -> np.ndarray:
    dev = next(model.parameters()).device
    a_all, b_all = _t(pred), _t(ref)
    out = []
    for i in range(0, a_all.shape[0], chunk):
        x = a_all[i:i + chunk].to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        y = b_all[i:i + chunk].to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        out.append(model(x, y).flatten())
    return torch.cat(out).cpu().numpy()


def clip_metrics(pred, ref, window: str = "gaussian", lpips_net=None, device: str = "cpu") -> dict:
    assert tuple(pred.shape) == tuple(ref.shape), (pred.shape, ref.shape)
    r = {"psnr": psnr(pred, ref), "ssim": float(np.mean(ssim_frames(pred, ref, window, device=device)))}
    if lpips_net is not None:
        r["lpips"] = float(np.mean(lpips_frames(pred, ref, lpips_net)))
    return r


def summarize(rows: list[dict], keys=("psnr", "ssim", "lpips")) -> dict:
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], float)
        if len(v):
            out[k] = float(v.mean())
            out[k + "_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
    if rows and "psnr" in rows[0]:
        w = min(rows, key=lambda r: r["psnr"])
        out["worst"], out["worst_id"] = w["psnr"], w["clip"]
    return out
