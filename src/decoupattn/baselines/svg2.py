from __future__ import annotations

from . import _svg as S
from ._common import LN2, BaselineHook, MatchedBaseline, PerBranch, bhsd, flat, seeded, unpermute_rows
from ._density import selected_pairs, vb_exec_pairs


class SVG2Hook(BaselineHook):
    name = "svg2"

    def __init__(self, top_p: float, num_q_centroids: int = 300, num_k_centroids: int = 1000,
                 min_kc_ratio: float = 0.10, kmeans_iter_init: int = 50, kmeans_iter_step: int = 2,
                 zero_step_kmeans_init: bool = False, **kw):
        super().__init__(**kw)
        self.svg = S.import_svg()
        self.clusters = S.Clusters(self.svg, num_q_centroids, num_k_centroids, kmeans_iter_init, kmeans_iter_step)
        self.sparse = S.VarBlockSparse(self.svg)
        self.top_p, self.min_kc_ratio, self.zero_init = float(top_p), float(min_kc_ratio), bool(zero_step_kmeans_init)

    def reset(self) -> None:
        super().reset()
        self.clusters.reset()

    def _warm_up(self, q, k, ctx) -> None:
        if self.zero_init and ctx.layer >= self.dl and ctx.step < self.ds:
            with seeded(q.device, ctx.layer, ctx.step):
                self.clusters(ctx.layer, q, k)

    def _route(self, q, k, v, ctx):
        H, D = q.shape[1], q.shape[3]
        with seeded(q.device, ctx.layer, ctx.step), self.seg("kmeans"):
            ql, qcen, qsz, kl, kcen, ksz = self.clusters(ctx.layer, q, k)
        with self.seg("route"):
            bm = self.svg.ku.identify_dynamic_map(qcen.view(1, H, -1, D), kcen.view(1, H, self.clusters.kc, D), qsz,
                                                  ksz, self.top_p, self.min_kc_ratio)
            q_p, k_p, v_p, q_idx = S.permute_qkv(self.svg, q, k, v, ql, kl)
        return q_p, k_p, v_p, q_idx, bm, qsz, ksz

    def __call__(self, q, k, v, ctx):
        if self.is_dense(ctx):
            if self.zero_init:
                self._warm_up(bhsd(q), bhsd(k), ctx)
            return None
        B, S_, H, D = q.shape
        q_p, k_p, v_p, q_idx, bm, qsz, ksz = self._route(bhsd(q), bhsd(k), bhsd(v), ctx)
        lse = None
        with self.seg("sparse_kernel"):
            self.sparse.plan(bm, qsz, ksz, D, q.dtype)
            if self.lse:
                o_p, lse_p = self.sparse.run_lse(q_p, k_p, v_p)
                o_p = o_p.reshape(B, H, S_, D)
                lse = unpermute_rows(lse_p.view(B, H, S_).float(), q_idx.view(B, H, S_)) * LN2
            else:
                o_p = self.sparse.run(q_p, k_p, v_p)
            out = self.svg.inv_permute(o_p, q_idx, dim=2)
        self.record(selected_pairs(bm, qsz, ksz), vb_exec_pairs(bm, qsz, ksz), H * S_ * S_)
        return (flat(out), lse[0]) if self.lse else flat(out)

    def joint(self, q, k, v, nv, ctx):
        qv, kv, vv = bhsd(q[:, :nv]), bhsd(k[:, :nv]), bhsd(v[:, :nv])
        if self.is_dense(ctx):
            self._warm_up(qv, kv, ctx)
            return None
        H = q.shape[2]
        q_p, k_p, v_p, q_idx, bm, qsz, ksz = self._route(qv, kv, vv, ctx)
        with self.seg("route"):
            qj, kj, vj, bmj, qs, ks, qi = S.joint_layout(q_p, k_p, v_p, q_idx, bm, qsz, ksz, q, k, v, nv)
            del qv, kv, vv, q_p, k_p, v_p
        with self.seg("sparse_kernel"):
            self.sparse.plan(bmj, qs, ks, q.shape[3], q.dtype)
            o = self.sparse.run(qj, kj, vj)
            del qj, kj, vj
            o = self.svg.inv_permute(o, qi, dim=2)
        self.record(selected_pairs(bm, qsz, ksz), vb_exec_pairs(bm, qsz, ksz), H * nv * nv)
        return flat(o)


HOOK_KEYS = ("top_p", "num_q_centroids", "num_k_centroids", "min_kc_ratio", "kmeans_iter_init", "kmeans_iter_step",
             "zero_step_kmeans_init")


class SVG2(MatchedBaseline):
    name = "svg2"
    label = "SVG2"
    upstream = ("Sparse-VideoGen",)
    knob = "top_p"

    def make_hook(self, adapter, settings: dict):
        kw = {k: settings[k] for k in HOOK_KEYS if k in settings}
        kw["lse"] = bool(getattr(adapter, "needs_lse", False))

        def make():
            return SVG2Hook(**kw)
        return PerBranch(make, adapter.branches) if settings.get("per_branch") else make()


BASELINE = SVG2
