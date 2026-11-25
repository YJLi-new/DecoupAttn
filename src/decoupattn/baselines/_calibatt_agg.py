from __future__ import annotations

import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import types

import numpy as np
import torch

from ._calibatt_port import import_calibatt

THREADS = 4


def scaled_epsilon(x: float, c: float, g3) -> float:
    g = g3[1] + (g3[0] - g3[1]) * math.exp(-g3[2] * x)
    return 1.0 - c * (1.0 - g)


def c_grid(lo: float, hi: float, step: float) -> list[float]:
    return [round(float(x), 3) for x in np.arange(lo, hi + 1e-4, step)]


def grid_density(E: torch.Tensor, eps: list[float], agree: float, rep_heads=(), rep_density: float = 0.0) -> list[float]:
    nk = E.shape[-1]
    tot = E.sum(-1, keepdim=True)
    se, idx = E.sort(dim=-1, descending=True)
    cum = se.cumsum(-1)
    del se
    rank = torch.empty_like(idx)
    rank.scatter_(-1, idx, torch.arange(nk).expand_as(idx))
    del idx
    out = []
    for e in eps:
        kept = ((cum < tot * e).sum(-1) + 1).clamp_(max=nk)
        dh = ((rank < kept[..., None]).float().mean(0) >= agree).float().mean((1, 2))
        for h in rep_heads:
            dh[h] = rep_density
        out.append(float(dh.mean()))
    return out


def _grid_step(args):
    torch.set_num_threads(THREADS)
    samples, ti, t, ls, rep, g3, cs, agree, rep_density = args
    energies = [np.load(os.path.join(d, "block_energy.npy"), mmap_mode="r") for d in samples]
    eps = [scaled_epsilon(t, c, g3) for c in cs]
    out = []
    for li, layer in enumerate(ls):
        E = torch.from_numpy(np.stack([np.array(e[ti, li]) for e in energies])).float()
        out.append(grid_density(E, eps, agree, rep.get((t, layer), ()), rep_density))
    return ti, out


def skip_list_density(sl: torch.Tensor, rep_heads, nk: int, rep_density: float) -> float:
    sl = sl.to(torch.int64)
    counts, pairs = sl[..., 0], sl[..., 1:]
    if pairs.shape[-1] % 2:
        pairs = torch.nn.functional.pad(pairs, (0, 1))
    pairs = pairs.view(*pairs.shape[:-1], -1, 2)
    cum = (pairs[..., 1] - pairs[..., 0]).abs().cumsum(-1)
    last = ((counts - 1) // 2).clamp(min=0)
    per_row = torch.gather(cum, -1, last.unsqueeze(-1)).squeeze(-1) * (counts > 0).to(torch.int64)
    dh = per_row.sum(dim=(0, 2)).float() / (sl.shape[2] * nk)
    for h in rep_heads:
        dh[h] = rep_density
    return float(dh.mean())


def invert(target: float, cs, dens) -> float:
    order = np.argsort(dens)
    return float(np.interp(target, np.array(dens)[order], np.array(cs)[order]))


def official_masks(samples, ts, ls, c: float, g3, mdir: str, ix: dict) -> None:
    from CalibAtt.calibration.block_masks import CalibrationMixin
    F_, Hp, Wp = ix["grid"]
    stand_in = types.SimpleNamespace(num_text_tokens=0, text_tokens_position="none", calibrate_masks=False,
                                     fa_block_size_m=ix["fa_block"][0], fa_block_size_n=ix["fa_block"][1],
                                     num_frames=F_, tokens_per_frame=Hp * Wp)
    for d in samples:
        out = os.path.join(mdir, os.path.basename(d))
        os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out, "sparse_pattern_masks.pt")):
            continue
        E = np.load(os.path.join(d, "block_energy.npy"), mmap_mode="r")
        masks = {}
        for ti, t in enumerate(ts):
            eps = scaled_epsilon(t, c, g3)
            stand_in._get_scheduled_epsilon = lambda e=eps: e
            for li, layer in enumerate(ls):
                be = torch.from_numpy(np.array(E[ti, li])).float()[None]
                mask, _ = CalibrationMixin._calibrate_masks_from_block_energy(stand_in, be, 0)
                for h in range(mask.shape[1]):
                    masks[(t - ts[0], layer - ls[0], h)] = mask[:, h:h + 1].clone()
        torch.save(masks, os.path.join(out, "sparse_pattern_masks.tmp.pt"))
        os.replace(os.path.join(out, "sparse_pattern_masks.tmp.pt"), os.path.join(out, "sparse_pattern_masks.pt"))


def run_official(samples, ts, ls, c: float, g3, out: str, rep: dict, ix: dict, s: dict, log=print):
    from CalibAtt.calibration import aggregate_sparse_masks as ASM
    tag = f"c{c:.4f}"
    mdir, adir = os.path.join(out, f"masks_{tag}"), os.path.join(out, f"agg_{tag}")
    official_masks(samples, ts, ls, c, g3, mdir, ix)
    ASM.aggregate_sparse_masks(input_path=mdir, output_dir=adir, agreement_threshold=s["agreement_threshold"],
                               fa_block_size=ix["fa_block"], skip_list_merge_percentile=s["trim_percentile"],
                               gpu_device=None, verbose=True)
    shutil.rmtree(mdir)
    sd = torch.load(os.path.join(adir, "skip_lists.pt"), weights_only=True)
    sd["skip_lists_dict"] = {(t + ts[0], layer + ls[0]): v for (t, layer), v in sd["skip_lists_dict"].items()}
    path = os.path.join(out, f"skip_lists_{tag}.pt")
    torch.save(sd, path)
    nk, rep_density = ix["shape"][-1], s["anchor_rows"] / ix["grid"][1]
    per = [skip_list_density(v, rep.get(k, ()), nk, rep_density) for k, v in sd["skip_lists_dict"].items()]
    log(f"calibatt aggregation at c = {c:.4f}: executed density {float(np.mean(per)):.4f}")
    return path, float(np.mean(per)), sd.get("trim_stats", {})


def aggregate(cap: str, out: str, s: dict, log=print) -> dict:
    record = os.path.join(out, "agg.json")
    if os.path.exists(record) and os.path.lexists(os.path.join(out, "skip_lists.pt")):
        with open(record) as f:
            res = json.load(f)
        if "chosen" in res:
            return res
    import_calibatt()
    from CalibAtt.calibration.aggregate_spatial_similarities import aggregate_spatial_similarities
    torch.set_num_threads(THREADS)
    os.makedirs(out, exist_ok=True)
    samples = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(cap, "*", "block_energy.npy")))
    with open(os.path.join(samples[0], "block_energy_index.json")) as f:
        ix = json.load(f)
    ts, ls, g3, target = ix["t"], ix["l"], ix["eps"], s["target_density"]
    rep_density = s["anchor_rows"] / ix["grid"][1]
    log(f"calibatt aggregation: {len(samples)} samples, steps {ts[0]}..{ts[-1]}, layers {ls[0]}..{ls[-1]}")
    sim = aggregate_spatial_similarities(input_path=cap, output_path=os.path.join(out, "similarity_map.pt"))
    rep: dict = {}
    for (t, layer, h), value in sim.items():
        if value >= s["repetitive_threshold"]:
            rep.setdefault((t, layer), set()).add(h)
    n_rep = sum(len(v) for k, v in rep.items() if k[0] in ts and k[1] in ls)
    cs = c_grid(*s["c_grid"])
    grid_file = os.path.join(out, "grid.json")
    if os.path.exists(grid_file):
        with open(grid_file) as f:
            dens = json.load(f)["density_pretrim"]
    else:
        acc = np.zeros((len(ts), len(ls), len(cs)))
        jobs = [(samples, ti, t, ls, rep, g3, cs, s["agreement_threshold"], rep_density) for ti, t in enumerate(ts)]
        with mp.get_context("spawn").Pool(min(int(s["workers"]), len(jobs))) as pool:
            for ti, o in pool.imap_unordered(_grid_step, jobs):
                acc[ti] = np.array(o)
                log(f"calibatt grid: step {ts[ti]} done")
        dens = acc.mean((0, 1)).tolist()
        with open(grid_file, "w") as f:
            json.dump({"c": cs, "density_pretrim": dens}, f)
    c = invert(target, cs, dens)
    res = {"samples": [os.path.basename(x) for x in samples], "grid": {"c": cs, "density_pretrim": dens},
           "official_c1_density_pretrim": float(np.interp(1.0, cs, dens)), "eps_schedule": g3, "repetitive_heads": n_rep,
           "target": target, "runs": []}
    for _ in range(2):
        path, d_final, trim = run_official(samples, ts, ls, c, g3, out, rep, ix, s, log)
        d_grid = float(np.interp(c, cs, dens))
        res["runs"].append({"c": c, "density_final": d_final, "density_pretrim_grid": d_grid,
                            "skip_lists": os.path.basename(path),
                            "trim_total_extra_blocks": trim.get("total_extra_blocks")})
        if abs(d_final - target) <= s["density_tol"]:
            break
        c = invert(target - (d_final - d_grid), cs, dens)
    best = min(res["runs"], key=lambda r: abs(r["density_final"] - target))
    res.update(chosen=best["c"], chosen_skip_lists=best["skip_lists"], chosen_density=best["density_final"])
    link = os.path.join(out, "skip_lists.pt")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(best["skip_lists"], link)
    with open(record + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(record + ".tmp", record)
    log(f"calibatt: c = {best['c']:.4f}, executed density {best['density_final']:.4f}")
    return res
