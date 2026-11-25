from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from ._common import BaselineHook, MatchedBaseline, PerBranch, add_path, bhsd, flat, seeded, upstream_path
from ._density import selected_pairs, vb_exec_pairs

UPSTREAM = "SparsePR"
CORE_KEYS = ("num_q_centroids", "num_k_centroids", "target_density", "kmeans_iter_init", "kmeans_iter_step",
             "selector_policy", "selector_budget_mode", "probe_rows", "repair_rank", "repair_ridge")


@dataclass
class _Route:
    origin_step: int
    route: object
    q_order: torch.Tensor


def import_sparsepr(cuda_ext_selector: bool = False):
    add_path(upstream_path(UPSTREAM, "src"))
    os.environ.setdefault("SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ", "1")
    os.environ.setdefault("SPARSEPR_COSMOS3_ORIGINAL_CENTROID_BACKEND", "triton_atomic_fp32")
    ext = False
    if cuda_ext_selector:
        from sparsepr.kernels.n8_extension import build_directory, load_n8_kernels
        mod = load_n8_kernels(required=False)
        if mod is None or not hasattr(mod, "svg_ear_select_budget"):
            raise RuntimeError(f"cuda_ext_selector needs SparsePR's n8_kernels extension in {build_directory()} "
                               "(build src/sparsepr/kernels/n8_ext, or point SPARSEPR_N8_CUDA_EXT_BUILD at the build)")
        ext = True
    os.environ["SPARSEPR_N8_CUDA_EXT_SELECTOR"] = "1" if ext else "0"
    import flashinfer
    from sparsepr import kmeans_utils as ku
    from sparsepr.kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
    from sparsepr.models.common import N8V6Config, N8V6Core
    ku.set_flashinfer_workspace_cache_enabled(True)
    return ku, permute_tensor_by_labels_triton, apply_inverse_permutation_triton, N8V6Config, N8V6Core


class SparsePRHook(BaselineHook):
    name = "sparsepr"

    def __init__(self, target_density: float, route_refresh_every: int = 1, flashinfer_plan_reuse: bool = True,
                 cuda_ext_selector: bool = False, text_keys: bool = False, gqa_group_size: int = 1,
                 dense_steps: int = 1, dense_layers: int = 1, lse: bool = False, **core):
        super().__init__(dense_steps, dense_layers, lse)
        self.ku, self.perm, self.inv, config, self.core_cls = import_sparsepr(cuda_ext_selector)
        self.cfg = config(target_density=float(target_density), **core)
        self.refresh, self.plan_reuse = int(route_refresh_every), bool(flashinfer_plan_reuse)
        self.wants_text, self.gqa = bool(text_keys), int(gqa_group_size)
        self.cores: dict[int, object] = {}
        self.routes: dict[int, _Route] = {}

    def reset(self) -> None:
        super().reset()
        self.cores.clear()
        self.routes.clear()
        if self.plan_reuse:
            self.ku.clear_flashinfer_plan_cache()

    def _core(self, layer: int):
        if layer not in self.cores:
            self.cores[layer] = self.core_cls(self.cfg)
        return self.cores[layer]

    def _sizes(self, route, H: int):
        c = self.cfg
        return (route.video_dynamic_map.view(1, H, c.num_q_centroids, c.num_k_centroids),
                route.q_cluster_sizes.view(1, H, c.num_q_centroids),
                route.k_cluster_sizes.view(1, H, c.num_k_centroids))

    def _record(self, dmap, qsz, ksz, probe_rows: int, tokens: int) -> None:
        H = dmap.shape[1]
        probes = probe_rows * tokens * H
        self.record(selected_pairs(dmap, qsz, ksz) + probes, vb_exec_pairs(dmap, qsz, ksz) + probes,
                    H * tokens * tokens)

    def _attend(self, q, k, v, ctx):
        _, H, S_, _ = q.shape
        layer, step = ctx.layer, ctx.step
        core, entry = self._core(layer), self.routes.get(layer)
        fresh = entry is None or (step - self.ds) % self.refresh == 0
        with self.seg("route"):
            if fresh:
                route = core.route(q, k, v, total_key_tokens=S_)
                entry = self.routes[layer] = _Route(step, route,
                                                    torch.argsort(route.q_labels, dim=-1).to(torch.int32).contiguous())
            else:
                kp, _ = self.perm(k, None, dim=2, sorted_indices=entry.route.k_sorted_indices)
                vp, _ = self.perm(v, None, dim=2, sorted_indices=entry.route.k_sorted_indices)
                route = replace(entry.route, k_permuted=kp, v_permuted=vp)
            q_p, _ = self.perm(q, None, dim=2, sorted_indices=entry.q_order)
        dmap, qsz, ksz = self._sizes(route, H)
        with self.seg("sparse_kernel"):
            key = token = None
            if self.plan_reuse:
                key, token = (layer, 0), (id(entry.route), entry.origin_step)
            o_p = self.ku.dynamic_block_sparse_fwd_flashinfer(q_p, route.k_permuted, route.v_permuted, dmap, qsz, ksz,
                                                              is_cpu=False, plan_cache_key=key, plan_cache_token=token)
            base = self.inv(o_p, entry.q_order, dim=2)
        with self.seg("repair"):
            out, info = core.repair(base, q, k, v, route)
        if route.probe_rows is not None and entry.route.probe_rows is None:
            entry.route.probe_rows, entry.route.probe_weights = route.probe_rows, route.probe_weights
        empty = q.new_empty((0,))
        entry.route.q_role_features = entry.route.k_permuted = entry.route.v_permuted = empty
        self._record(dmap, qsz, ksz, int(info["probe_rows"]), S_)
        return out

    def _attend_text(self, q, k, v, text, ctx):
        B, H, S_, D = q.shape
        kt, vt = text
        T = kt.shape[2]
        k_full, v_full = torch.cat((k, kt), 2), torch.cat((v, vt), 2)
        core = self._core(ctx.layer)
        with self.seg("route"):
            route = core.route(q, k, v, gqa_group_size=self.gqa, always_attended_keys=T, total_key_tokens=S_ + T,
                               k_full_source=k_full, v_full_source=v_full)
            if not route.permuted_includes_suffix:
                raise RuntimeError("SparsePR did not return the full [video | text] key layout")
            q_order = torch.argsort(route.q_labels, dim=-1).to(torch.int32).contiguous()
            q_p, _ = self.perm(q, None, dim=2, sorted_indices=q_order)
            from sparsepr.adapters.cosmos3 import Cosmos3N8Processor
            bmap, qs, ks = Cosmos3N8Processor._append_understanding_cluster(
                route.video_dynamic_map, route.q_cluster_sizes, route.k_cluster_sizes, batch=B, heads=H,
                understanding_tokens=T)
        with self.seg("sparse_kernel"):
            q_full = torch.cat((q_p, q.new_zeros(B, H, T, D)), 2)
            o_p = self.ku.dynamic_block_sparse_fwd_flashinfer(q_full, route.k_permuted, route.v_permuted, bmap, qs, ks,
                                                              is_cpu=False)[:, :, :S_]
            base = self.inv(o_p, q_order, dim=2)
        with self.seg("repair"):
            out, info = core.repair(base, q, k_full, v_full, route)
        self._record(*self._sizes(route, H), int(info["probe_rows"]), S_)
        return out

    def __call__(self, q, k, v, ctx):
        if self.is_dense(ctx):
            return None
        text = getattr(ctx, "text", None) if self.wants_text else None
        if self.wants_text and text is None:
            raise RuntimeError("SparsePR attends the text keys through its own always-attended cluster: the adapter "
                               "has to pass them as ctx.text = (k_text, v_text)")
        if self.lse and text is None:
            raise RuntimeError("SparsePR's repaired output has no log-sum-exp to merge text keys with: this model "
                               "needs the text_keys path (Cosmos3-Nano) or hook.joint (HunyuanVideo)")
        with seeded(q.device, ctx.layer, ctx.step):
            if text is not None:
                return flat(self._attend_text(bhsd(q), bhsd(k), bhsd(v), (bhsd(text[0]), bhsd(text[1])), ctx)), None
            return flat(self._attend(bhsd(q), bhsd(k), bhsd(v), ctx))

    def joint(self, q, k, v, nv, ctx):
        if self.is_dense(ctx):
            return None
        query, key, value = bhsd(q), bhsd(k), bhsd(v)
        H, S_ = query.shape[1], query.shape[2]
        nt = S_ - nv
        qv, kv, vv = query[:, :, :nv].contiguous(), key[:, :, :nv].contiguous(), value[:, :, :nv].contiguous()
        core = self._core(ctx.layer)
        with seeded(q.device, ctx.layer, ctx.step):
            with self.seg("route"):
                route = core.route(qv, kv, vv, always_attended_keys=nt, total_key_tokens=S_, k_full_source=key,
                                   v_full_source=value)
                if not route.permuted_includes_suffix:
                    raise RuntimeError("SparsePR did not return the full [video | text] key layout")
                q_order = torch.argsort(route.q_labels, dim=-1).to(torch.int32).contiguous()
                q_full = torch.empty((q_order.shape[0], S_), device=q.device, dtype=torch.int32)
                q_full[:, :nv] = q_order
                q_full[:, nv:] = torch.arange(nv, S_, device=q.device, dtype=torch.int32)
                q_p, _ = self.perm(query, None, dim=2, sorted_indices=q_full)
                del query, kv, vv
                vmap, qsv, ksv = self._sizes(route, H)
                bm = F.pad(vmap, (0, 2, 0, 2), value=False)
                bm[:, :, :-1, -2] = True
                bm[:, :, -2, :-1] = True
                bm[:, :, -1, -1] = True
                qs, ks = F.pad(qsv, (0, 2), value=0), F.pad(ksv, (0, 2), value=0)
                qs[:, :, -2] = nt
                ks[:, :, -2] = nt
            with self.seg("sparse_kernel"):
                o_p = self.ku.dynamic_block_sparse_fwd_flashinfer(q_p, route.k_permuted, route.v_permuted, bm, qs, ks,
                                                                  is_cpu=False)
                del q_p
                base = self.inv(o_p[:, :, :nv], q_order, dim=2)
            with self.seg("repair"):
                rep, info = core.repair(base, qv, key, value, route)
        self._record(vmap, qsv, ksv, int(info["probe_rows"]), nv)
        return flat(torch.cat((rep, o_p[:, :, nv:]), 2))


HOOK_KEYS = CORE_KEYS + ("route_refresh_every", "flashinfer_plan_reuse", "cuda_ext_selector", "text_keys",
                         "gqa_group_size")


class SparsePR(MatchedBaseline):
    name = "sparsepr"
    label = "SparsePR"
    upstream = (UPSTREAM,)
    knob = "target_density"

    def make_hook(self, adapter, settings: dict):
        kw = {k: settings[k] for k in HOOK_KEYS if k in settings}
        kw["lse"] = bool(getattr(adapter, "needs_lse", False)) and not settings.get("text_keys")

        def make():
            return SparsePRHook(**kw)
        return PerBranch(make, adapter.branches) if settings.get("per_branch") else make()


BASELINE = SparsePR
