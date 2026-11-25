from __future__ import annotations

import torch
import torch.nn.functional as F

from . import _svg as S
from ._common import BaselineHook, MatchedBaseline, PerBranch, bhsd, flat, seeded, unpermute_rows
from ._density import pseudo_key_pairs, selected_pairs, vb_exec_pairs


class EARHook(BaselineHook):
    name = "svgear"

    def __init__(self, top_p: float, num_q_centroids: int = 300, num_k_centroids: int = 1000,
                 min_kc_ratio: float = 0.05, gamma: float = 1.0, kmeans_iter_init: int = 50, kmeans_iter_step: int = 2,
                 zero_step_kmeans_init: bool = False, route_head_chunk: int = 0, **kw):
        super().__init__(**kw)
        self.svg = S.import_svg()
        self.clusters = S.Clusters(self.svg, num_q_centroids, num_k_centroids, kmeans_iter_init, kmeans_iter_step)
        self.sparse = S.VarBlockSparse(self.svg)
        self.top_p, self.min_kc_ratio, self.gamma = float(top_p), float(min_kc_ratio), float(gamma)
        self.zero_init, self.head_chunk = bool(zero_step_kmeans_init), int(route_head_chunk)

    def reset(self) -> None:
        super().reset()
        self.clusters.reset()

    def _warm_up(self, q, k, ctx) -> None:
        if self.zero_init and ctx.layer >= self.dl and ctx.step < self.ds:
            with seeded(q.device, ctx.layer, ctx.step):
                self.clusters(ctx.layer, q, k)

    def _map(self, *args):
        ku = self.svg.ku

        def one(a):
            return ku.identify_dynamic_map_estimated(*a, top_p=self.top_p, gamma=self.gamma,
                                                     min_kc_ratio=self.min_kc_ratio)
        if not self.head_chunk:
            return one(args).contiguous()
        H, g = args[0].shape[1], self.head_chunk
        return torch.cat([one(tuple(x[:, i:i + g] for x in args)) for i in range(0, H, g)], 1).contiguous()

    def _route(self, q, k, v, ctx):
        B, H, S_, D = q.shape
        cl = self.clusters
        with seeded(q.device, ctx.layer, ctx.step), self.seg("kmeans"):
            ql, qcen, qsz, kl, kcen, ksz = cl(ctx.layer, q, k)
        with self.seg("route"):
            q_p, k_p, v_p, q_idx = S.permute_qkv(self.svg, q, k, v, ql, kl)
            vcen = self.svg.ku.scatter_mean_fused(v_p.view(B, H, S_, D), ksz)
            qcen, kcen = qcen.view(B, H, cl.qc, D), kcen.view(B, H, cl.kc, D)
            bm = self._map(q_p, k_p, v_p, qsz, ksz, qcen, kcen, vcen)
        return q_p, k_p, v_p, q_idx, kcen, vcen, bm, qsz, ksz

    def _record(self, bm, qsz, ksz, total) -> None:
        self.record(selected_pairs(bm, qsz, ksz), vb_exec_pairs(bm, qsz, ksz), total, pseudo_key_pairs(bm, qsz, ksz))

    def __call__(self, q, k, v, ctx):
        if self.is_dense(ctx):
            if self.zero_init:
                self._warm_up(bhsd(q), bhsd(k), ctx)
            return None
        B, S_, H, D = q.shape
        q_p, k_p, v_p, q_idx, kcen, vcen, bm, qsz, ksz = self._route(bhsd(q), bhsd(k), bhsd(v), ctx)
        with self.seg("plan"):
            self.sparse.plan(bm, qsz, ksz, D, q.dtype)
        with self.seg("attn"):
            o_p, lse_p = S.fused_compensation(self.svg, self.sparse, q_p, k_p, v_p, kcen, vcen, bm, qsz, ksz)
            out = self.svg.inv_permute(o_p, q_idx, dim=2)
        self._record(bm, qsz, ksz, H * S_ * S_)
        if self.lse:
            return flat(out), unpermute_rows(lse_p, q_idx.view(B, H, S_))[0]
        return flat(out)

    def joint(self, q, k, v, nv, ctx):
        qv, kv, vv = bhsd(q[:, :nv]), bhsd(k[:, :nv]), bhsd(v[:, :nv])
        if self.is_dense(ctx):
            self._warm_up(qv, kv, ctx)
            return None
        H, D = q.shape[2], q.shape[3]
        q_p, k_p, v_p, q_idx, kcen, vcen, bm, qsz, ksz = self._route(qv, kv, vv, ctx)
        with self.seg("route"):
            qj, kj, vj, bmj, qs, ks, qi = S.joint_layout(q_p, k_p, v_p, q_idx, bm, qsz, ksz, q, k, v, nv)
            kcj = F.pad(kcen, (0, 0, 0, 1), value=0).contiguous()
            vcj = F.pad(vcen, (0, 0, 0, 1), value=0).contiguous()
            del qv, kv, vv, q_p, k_p, v_p
        with self.seg("plan"):
            self.sparse.plan(bmj, qs, ks, D, q.dtype)
        with self.seg("attn"):
            o, _ = S.fused_compensation(self.svg, self.sparse, qj, kj, vj, kcj, vcj, bmj, qs, ks)
            del qj, kj, vj
            o = self.svg.inv_permute(o, qi, dim=2)
        self._record(bm, qsz, ksz, H * nv * nv)
        return flat(o)


HOOK_KEYS = ("top_p", "num_q_centroids", "num_k_centroids", "min_kc_ratio", "gamma", "kmeans_iter_init",
             "kmeans_iter_step", "zero_step_kmeans_init", "route_head_chunk")


class SVGEAR(MatchedBaseline):
    name = "svgear"
    label = "SVG-EAR"
    upstream = ("Sparse-VideoGen",)
    knob = "top_p"

    def make_hook(self, adapter, settings: dict):
        kw = {k: settings[k] for k in HOOK_KEYS if k in settings}
        kw["lse"] = bool(getattr(adapter, "needs_lse", False))

        def make():
            return EARHook(**kw)
        return PerBranch(make, adapter.branches) if settings.get("per_branch") else make()


BASELINE = SVGEAR
