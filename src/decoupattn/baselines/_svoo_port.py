from __future__ import annotations

import json
import math
import os

import torch

from ._common import BaselineHook, add_path, bhsd, seeded, unpermute_rows, upstream_path
from ._density import selected_pairs, vb_exec_pairs
from ._svg import joint_layout

LN2 = math.log(2.0)
OFFICIAL_TOP_P = {"wan": 0.90, "hunyuan": 0.88}
OFFICIAL_WINDOW = {"wan": (0.15, 0.20), "hunyuan": (0.05, 0.10)}


def import_svoo():
    add_path(upstream_path("SVOO"))
    import diffusers.models.transformers.transformer_wan as TW
    for name in ("dispatch_attention_fn", "_get_added_kv_projections"):
        if not hasattr(TW, name):
            def _stub(*_a, _n=name, **_k):
                raise NotImplementedError(f"{_n} belongs to a branch of the SVOO processor that this wrapper never runs")
            setattr(TW, name, _stub)
    import svoo.models.wan.attention as A
    return A


def processor_settings(lookup, grid, layers, s: dict, dense_steps: int) -> dict:
    return {"first_layers_fp": 0, "first_times_fp": 1001, "context_length": 0, "num_frame": grid[0],
            "frame_size": grid[1] * grid[2], "num_q_centroids": s["qc"], "num_k_centroids": s["kc"],
            "top_p_kmeans": s["top_p"], "min_kc_ratio": s["min_kc_ratio"], "ear_gamma": 1.0, "use_ear": False,
            "num_layers": layers, "kmeans_iter_init": s["iters"], "kmeans_iter_step": s["iters"],
            "zero_step_kmeans_init": False, "use_global_constraints": False, "lambda_schedule": "linear",
            "diverse_top_p_k": 0.0, "use_routing_transformer_strategy": False, "mq1": None, "mk1": None, "mq2": None,
            "mk2": None, "use_svoo": True, "start_reuse_step": dense_steps + 1, "reuse_interval": s["reuse_interval"],
            "use_dynamic_min_kc_ratio": True, "sparsity_lookup": lookup, "dynamic_min_kc_ratio_min": s["kr_min"],
            "dynamic_min_kc_ratio_max": s["kr_max"], "measure_attention_sparsity": False,
            "transformer_stage": "transformer"}


def sparse_fwd_lse(CC, q, k, v, block_map, qsz, ksz):
    fs = CC._require_flashinfer_sparse()
    B, H, S, D = q.shape
    qc, kc = qsz.shape[-1], ksz.shape[-1]
    buf = torch.empty((CC._env_int("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", 128 * 1024 * 1024),), dtype=torch.uint8,
                      device=q.device)
    w = CC.make_variable_block_sparse_attention_wrapper(
        fs, buf, backend=os.environ.get("SVOO_FLASHINFER_SPARSE_BACKEND", "auto").lower())
    iw = CC._env_int("SVOO_FLASHINFER_SPARSE_INT_WORKSPACE_BYTES", 0)
    if iw > 0:
        w.reset_workspace_buffer(float_workspace_buffer=buf,
                                 int_workspace_buffer=torch.empty((iw,), dtype=torch.uint8, device=q.device))
    w.plan(block_mask_map=block_map.reshape(B * H, qc, kc), block_row_sz=qsz.reshape(B * H, qc),
           block_col_sz=ksz.reshape(B * H, kc), num_qo_heads=B * H, num_kv_heads=B * H, head_dim=D, q_data_type=q.dtype,
           kv_data_type=k.dtype)
    o, lse = w.run(q.reshape(B * H, S, D), k.reshape(B * H, S, D), v.reshape(B * H, S, D), return_lse=True)
    return o.reshape(B, H, S, D), lse.view(B, H, S).float()


class SVOOHook(BaselineHook):
    name = "svoo"

    def __init__(self, csv: str, grid, layers: int, heads: int, head_dim: int, settings: dict, lse: bool = False,
                 warmup: bool = True):
        super().__init__(settings["dense_steps"], settings["dense_layers"], lse)
        self.A = import_svoo()
        import svoo.co_clustering as CC
        from svoo.kernels.triton.permute import apply_inverse_permutation_triton
        from svoo.sparsity import SparsityLookup
        self.CC, self.inv_perm = CC, apply_inverse_permutation_triton
        self.cfg = processor_settings(SparsityLookup(csv), grid, layers, settings, self.ds)
        self.label_builds = 0
        self.procs: dict = {}
        if warmup:
            from svoo.utils.triton_warmup import warmup_svoo_triton_kernels
            warmup_svoo_triton_kernels(model_name="Wan", num_head=heads, head_dim=head_dim, hidden_dim=heads * head_dim,
                                       seq_len=grid[0] * grid[1] * grid[2], num_q_centroids=settings["qc"],
                                       num_k_centroids=settings["kc"], dtype=torch.bfloat16, device="cuda",
                                       cfg_values=(1,), include_rmsnorm=False, include_wan_block_kernels=False,
                                       include_flashinfer_sparse=True)

    def reset(self) -> None:
        super().reset()
        self.procs, self.label_builds = {}, 0

    def _proc(self, layer: int):
        p = self.procs.get(layer)
        if p is None:
            p = self.procs[layer] = self.A.WanAttn_SAPAttn_Processor(layer_idx=layer)
            for key, val in self.cfg.items():
                setattr(p, key, val)
        return p

    def __call__(self, q, k, v, ctx):
        if self.is_dense(ctx):
            return None
        B, S, H, D = q.shape
        p = self._proc(ctx.layer)
        p._step_from_callback = p.current_inference_step = ctx.step + 1
        cached = getattr(p, "_cached_clustering_results", None)
        with seeded(q.device, ctx.layer, ctx.step), self.seg("cluster_route_permute"):
            q_p, k_p, v_p, block_map, qsz, ksz, q_idx = p.semantic_aware_permutation(bhsd(q), bhsd(k), bhsd(v), None)
        with self.seg("sparse_kernel"):
            if self.lse:
                o_p, lse_p = sparse_fwd_lse(self.CC, q_p, k_p, v_p, block_map, qsz, ksz)
                lse = unpermute_rows(lse_p, q_idx.view(B, H, S)) * LN2
            else:
                o_p = self.CC.dynamic_block_sparse_fwd_flashinfer(q_p, k_p, v_p, block_map, qsz, ksz, is_cpu=False)
            o = self.inv_perm(o_p, q_idx, dim=2)
        self.record(selected_pairs(block_map, qsz, ksz), vb_exec_pairs(block_map, qsz, ksz), H * S * S)
        self.label_builds += p._cached_clustering_results is not cached
        out = o.transpose(1, 2).reshape(B, S, H * D)
        return (out, lse[0]) if self.lse else out

    def joint(self, q, k, v, nv, ctx):
        if self.is_dense(ctx):
            return None
        B, S, H, D = q.shape
        p = self._proc(ctx.layer)
        p._step_from_callback = p.current_inference_step = ctx.step + 1
        cached = getattr(p, "_cached_clustering_results", None)
        with seeded(q.device, ctx.layer, ctx.step), self.seg("cluster_route_permute"):
            q_p, k_p, v_p, block_map, qsz, ksz, q_idx = p.semantic_aware_permutation(
                bhsd(q[:, :nv]), bhsd(k[:, :nv]), bhsd(v[:, :nv]), None)
            qj, kj, vj, bmj, qs, ks, qi = joint_layout(q_p, k_p, v_p, q_idx, block_map, qsz, ksz, q, k, v, nv)
        with self.seg("sparse_kernel"):
            o = self.inv_perm(self.CC.dynamic_block_sparse_fwd_flashinfer(qj, kj, vj, bmj, qs, ks, is_cpu=False), qi,
                              dim=2)
        self.record(selected_pairs(block_map, qsz, ksz), vb_exec_pairs(block_map, qsz, ksz), H * nv * nv)
        self.label_builds += p._cached_clustering_results is not cached
        return o.transpose(1, 2).reshape(B, S, H * D)

    def summary(self) -> dict:
        return {**super().summary(), "label_builds": int(self.label_builds)}


class Profiler:
    def __init__(self, fh, exact, threshold: float, query_samples: int, dense_steps: int, dense_layers: int,
                 branch: str = "cond"):
        self.f, self.fn, self.thr, self.qs = fh, exact, threshold, query_samples
        self.ds, self.dl, self.branch, self.n = dense_steps, dense_layers, branch, 0

    def __call__(self, q, k, v, ctx):
        if ctx.step < self.ds or ctx.layer < self.dl or ctx.branch != self.branch:
            return
        avg, per_head, st = self.fn(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), batch_size=0,
                                    threshold=self.thr, query_sample_size=self.qs)
        self.f.write(profile_lines(ctx.layer, ctx.step, self.thr, avg, per_head, st))
        self.f.flush()
        self.n += 1


def profile_lines(layer: int, step: int, threshold: float, avg: float, per_head, stats: dict) -> str:
    n = stats.get("seq_len", 0)
    ns = stats.get("sampled_seq_len", n)
    extra = f" | Query Samples: {ns}/{n}" if ns != n else ""
    head = (f"[Sparsity] Layer {layer} | Type: self | Step: {step + 1} | Threshold: {threshold:.2%} | "
            f"Avg Sparsity: {avg:.4f} | n={n}{extra}")
    lines = [head]
    lines += [f"  Head {h:2d}: Sparsity={s:.4f}" for h, s in enumerate(per_head)]
    return "\n".join(lines) + "\n"


def profile_tag(threshold: float, query_samples: int) -> str:
    return f"th{round(threshold * 100)}-{'q' + str(query_samples) if query_samples > 0 else 'qall'}"


def profile(adapter, settings: dict, work: str, seed: int, log=print) -> str:
    out = os.path.join(work, "profile.csv")
    if os.path.exists(out):
        return out
    os.environ.setdefault("SVOO_SPARSITY_USE_TRITON_PROB_COUNTS", "1")
    import_svoo()
    from svoo.sparsity import compute_exact_attention_sparsity
    from svoo.sparsity.merge import merge_sparsity_data, save_to_csv
    raw = os.path.join(work, "profile_raw")
    os.makedirs(raw, exist_ok=True)
    tag = profile_tag(settings["threshold"], settings["query_samples"])
    files = []
    for clip in adapter.clips("calib")[:settings["profile_clips"]]:
        path = os.path.join(raw, f"attention_sparsity-{clip}-{tag}.txt")
        files.append(path)
        if os.path.exists(path):
            continue
        with open(path + ".tmp", "w") as fh:
            prof = Profiler(fh, compute_exact_attention_sparsity, settings["threshold"], settings["query_samples"],
                            settings["dense_steps"], settings["dense_layers"])
            adapter.generate(clip, prof, seed)
        os.replace(path + ".tmp", path)
        log(f"svoo profile {clip}: {prof.n} calls")
    save_to_csv(merge_sparsity_data(files), out + ".tmp")
    os.replace(out + ".tmp", out)
    return out


def match(density, target: float, tol: float, iters: int, top_p0: float, window, top_p_lo: float = 0.30,
          scale_lo: float = 0.25, scale_hi: float = 4.0) -> dict:
    hist = []

    def run(top_p, scale):
        d = float(density(top_p, window[0] * scale, window[1] * scale))
        hist.append({"top_p": top_p, "window_scale": scale, "kr_min": window[0] * scale, "kr_max": window[1] * scale,
                     "exec_density": d})
        return d

    knob, d0 = "none", run(top_p0, 1.0)
    if abs(d0 - target) > tol:
        if d0 > target:
            if run(top_p_lo, 1.0) > target:
                knob, lo, hi, f = "window_scale", scale_lo, 1.0, (lambda x: run(top_p_lo, x))
            else:
                knob, lo, hi, f = "top_p", top_p_lo, top_p0, (lambda x: run(x, 1.0))
        else:
            knob, lo, hi, f = "window_scale", 1.0, scale_hi, (lambda x: run(top_p0, x))
        for _ in range(iters):
            mid = (lo + hi) / 2
            d = f(mid)
            if abs(d - target) <= tol:
                break
            lo, hi = (lo, mid) if d > target else (mid, hi)
    best = min(hist, key=lambda r: abs(r["exec_density"] - target))
    return {"knob": knob, "top_p": round(best["top_p"], 4), "kr_min": round(best["kr_min"], 4),
            "kr_max": round(best["kr_max"], 4), "exec_density": best["exec_density"], "target": target, "history": hist}


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)
