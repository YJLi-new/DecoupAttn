from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch

from .clustering import ClusterCache
from .compensation import CENTROID, FULL
from .constants import BUILD_STEP, CLUSTERS, DENSE_LAYERS, DENSE_STEPS, KMEANS_ITERS
from .layout import exec_density, pseudo_density, real_density, tiling_for_plan
from .plan import EST_FULL, KIND_DENSE


class _RefPlan:
    def __init__(self, mask, til):
        self.density = real_density(mask, til)
        self.exec_density = exec_density(mask, til)
        self.pseudo_density = pseudo_density(mask, til)


def _sdpa(q, k, v):
    o = torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1, 2) for x in (q, k, v)))
    return o.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


class DecoupRuntime:
    def __init__(self, plan, grid, *, dense=None, dense_heads=None, dense_lse=None, backend="kernels", device="cuda",
                 fast=True, lse=False, mixed="subset", schedule=None, tied=None, estimator=None,
                 cluster_all_layers=False, keep_plans="all", free_plans_at_end=False, last_branch="cond",
                 clamp_steps=False, lean=False, lean_hmax=40, keep_subsets=True, clusters=CLUSTERS,
                 kmeans_iters=KMEANS_ITERS, head_dim=128, head_cap=0, dense_head_cap=0, head_chunk_gib=None,
                 label_free=False):
        assert backend in ("kernels", "reference") and mixed in ("subset", "overwrite") and keep_plans in ("all", "row")
        assert estimator in (None, FULL, CENTROID), estimator
        self.plan, self.backend, self.ref = plan, backend, backend == "reference"
        self.til = tiling_for_plan(plan, grid, device)
        self.masks = torch.from_numpy(np.asarray(plan.mask, dtype=bool))
        self.kind = np.asarray(plan.kind).astype(np.int64)
        self.step_row = [int(x) for x in plan.step_row]
        self.est = np.asarray(plan.estimator)
        self.dense = dense or (self._dense_ref if self.ref else _sdpa)
        self._dense_heads, self._dense_lse = dense_heads, dense_lse
        self.fast, self.lse, self.mixed, self.force_est = fast and not lse, lse, mixed, estimator
        self.keep_plans, self.free_end, self.last_branch = keep_plans, free_plans_at_end, last_branch
        self.clamp, self.keep_subsets = clamp_steps, keep_subsets
        self.head_dim, self.head_cap, self.dense_cap, self.chunk_gib = head_dim, head_cap, dense_head_cap, head_chunk_gib
        if schedule is None:
            schedule = plan.reassign if plan.reassign in ("reassign", "rebuild") else "frozen"
        if tied is None:
            tied = str(plan.extra.get("abl_group", "")) == "tile"
        self.schedule, self.tied = schedule, tied
        self.cache = ClusterCache(self.til, schedule, tied, clusters, kmeans_iters, BUILD_STEP)
        self.lean_drop = not tied and ((lean and schedule == "reassign") or (label_free and schedule != "frozen"))
        if lean and not self.ref:
            from .kernels.block_sparse import set_scratch_heads
            set_scratch_heads(lean_hmax)
        self.any_sparse = (self.kind[DENSE_STEPS:] != KIND_DENSE).any(axis=(0, 2))
        self.cluster_layer = np.ones_like(self.any_sparse) if cluster_all_layers else self.any_sparse
        self.plans, self.chunk_plans, self.splits = {}, {}, {}
        self.reset()

    def reset(self):
        self.cache.reset()
        self.subsets, self.ev = {}, []
        self.stats = {"sparse_calls": 0, "dense_calls": 0, "density": [], "exec_density": [], "pseudo_density": [],
                      "call_exec_density": [], "cluster_builds": 0, "dense_head_calls": 0}

    def split(self, s, l):
        sp = self.splits.get((s, l))
        if sp is None:
            o = self.kind[s, l]
            sp = self.splits[(s, l)] = (np.nonzero(o == KIND_DENSE)[0], np.nonzero(o != KIND_DENSE)[0])
        return sp

    @staticmethod
    def _run_heads(sh):
        return np.concatenate([sh, sh]) if len(sh) == 1 else sh

    def estimator(self, s, l):
        if self.force_est is not None:
            return self.force_est
        return FULL if int(self.est[s, l]) == EST_FULL else CENTROID

    def _make_plan(self, m):
        m = m.to(self.til.device)
        if self.ref:
            return _RefPlan(m, self.til), m
        from .kernels.block_sparse import SpatialPlan
        return SpatialPlan(m, self.til, D=self.head_dim), m

    def n_chunks(self, m) -> int:
        nh = m.shape[0]
        tot = int((m.long() * self.til.n_real.to(m.device)[None, None, :]).sum()) if self.chunk_gib else 0
        n = max(1, min(nh // 2, math.ceil(tot * 4 / (self.chunk_gib * 2 ** 30)))) if tot else 1
        return max(n, -(-nh // self.head_cap)) if self.head_cap else n

    def _keep_row(self, row):
        if self.keep_plans == "row":
            self.plans = {k: v for k, v in self.plans.items() if k[0] == row}
            self.chunk_plans = {k: v for k, v in self.chunk_plans.items() if k[0] == row}

    def _plan_entry(self, s, l):
        row = self.step_row[s]
        p = self.plans.get((row, l))
        if p is None:
            self._keep_row(row)
            m = self.masks[row, l]
            if self.mixed == "subset":
                m = m[torch.from_numpy(self._run_heads(self.split(s, l)[1]))]
            nch = self.n_chunks(m) if self.mixed == "subset" else 1
            if nch > 1:
                p = self.plans[(row, l)] = (_RefPlan(m.to(self.til.device), self.til), None, nch)
            else:
                p = self.plans[(row, l)] = self._make_plan(m) + (1,)
        return p

    def static_plan(self, s, l):
        return self._plan_entry(s, l)[:2]

    def _chunk_plan(self, s, l, ci, nch, hs):
        ck = (self.step_row[s], l, ci, nch)
        p = self.chunk_plans.get(ck)
        if p is None:
            self._keep_row(ck[0])
            p = self.chunk_plans[ck] = self._make_plan(self.masks[ck[0], l][torch.from_numpy(hs)])
        return p

    @staticmethod
    def _chunks(rh, nch):
        return enumerate(np.array_split(np.arange(len(rh)), nch))

    def _missing(self, s) -> list[int]:
        row, out = self.step_row[s], []
        for l in range(DENSE_LAYERS, self.masks.shape[1]):
            if not len(self.split(s, l)[1]):
                continue
            p = self.plans.get((row, l))
            if p is None or (p[2] > 1 and any((row, l, ci, p[2]) not in self.chunk_plans for ci in range(p[2]))):
                out.append(l)
        return out

    def _build(self, s, layers):
        self._keep_row(self.step_row[s])
        for l in layers:
            nch = self._plan_entry(s, l)[2]
            if nch > 1:
                rh = self._run_heads(self.split(s, l)[1])
                for ci, idx in self._chunks(rh, nch):
                    self._chunk_plan(s, l, ci, nch, rh[idx])

    def build_all(self, steps=None):
        last = self.kind.shape[0] if steps is None else steps
        for s in range(DENSE_STEPS, last if self.keep_plans == "all" else min(last, DENSE_STEPS + 1)):
            self._build(s, self._missing(s))

    def build_step(self, ctx, around=contextlib.nullcontext):
        s = min(ctx.step, self.kind.shape[0] - 1)
        if s < DENSE_STEPS:
            return
        todo = self._missing(s)
        if todo:
            with around():
                self._build(s, todo)

    def _event(self):
        if self.til.device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def cluster_ms(self) -> dict:
        if self.ev and not isinstance(self.ev[0][2], float):
            torch.cuda.synchronize()
        ms = [(s, a.elapsed_time(b) if not isinstance(a, float) else 1e3 * (b - a)) for s, _, a, b in self.ev]
        out = {"total": sum(x for _, x in ms)}
        for s in sorted({x[0] for x in ms}):
            out[f"step{s}"] = sum(x for st, x in ms if st == s)
        return out

    def summary(self) -> dict:
        st = self.stats
        mean = (lambda x: float(np.mean(x)) if x else float("nan"))
        return {"exec_density": mean(st["call_exec_density"]), "sparse_calls": st["sparse_calls"],
                "dense_calls": st["dense_calls"], "cluster_builds": st["cluster_builds"],
                "sparse_exec_density": mean(st["exec_density"]), "sparse_real_density": mean(st["density"]),
                "pseudo_density": mean(st["pseudo_density"]), "cluster_ms": self.cluster_ms()["total"]}

    def _dense_ref(self, q, k, v):
        from .kernels.reference import dense_ref
        return dense_ref(q, k, v, range(q.shape[2])).to(q.dtype).reshape(1, q.shape[1], -1)

    def dense_heads(self, q, k, v, idx):
        if not self.dense_cap:
            return self._dense_heads_once(q, k, v, idx)
        parts = [self._dense_heads_once(q, k, v, c) for c in np.array_split(idx, -(-len(idx) // self.dense_cap))]
        return torch.cat([p[0] for p in parts], 2), (torch.cat([p[1] for p in parts], 0) if self.lse else None)

    def _dense_heads_once(self, q, k, v, idx):
        S, D = q.shape[1], q.shape[3]
        if self.lse:
            if self._dense_lse is not None:
                o, ls = self._dense_lse(*(x[:, :, idx].contiguous() for x in (q, k, v)))
            else:
                from .kernels.reference import dense_ref
                o, ls = dense_ref(q, k, v, idx.tolist(), return_lse=True)
            return o.reshape(1, S, len(idx), D), ls
        if self._dense_heads is not None:
            o = self._dense_heads(q, k, v, idx)
        else:
            it = torch.from_numpy(idx).to(q.device)
            o = self.dense(*(x.index_select(2, it) for x in (q, k, v)))
        return o.reshape(1, S, len(idx), D), None

    def _sub_clusters(self, key, rh):
        cl = self.cache.clusters(*key)
        sk = (key, tuple(int(h) for h in rh))
        sub = self.subsets.get(sk)
        if sub is None:
            sub = cl.subset(rh)
            if self.keep_subsets:
                self.subsets[sk] = sub
        return sub

    def _decoup(self, qs, ks, vs, sp, cl, m, est):
        if self.ref:
            from .kernels.reference import decoup_ref
            o, ls = decoup_ref(qs, ks, vs, m, cl.lab, cl.C, self.til, range(qs.shape[2]), estimator=est,
                               return_lse=True)
            o = o.to(qs.dtype).reshape(1, qs.shape[1], -1)
            return (o, ls) if self.lse else o
        from .attention import decoupled_attention
        return decoupled_attention(qs, ks, vs, sp, cl, m, self.til, est, self.fast, self.lse)

    @torch.no_grad()
    def __call__(self, q, k, v, ctx):
        s, l = ctx.step, ctx.layer
        if self.clamp:
            s = min(s, self.kind.shape[0] - 1)
        key = (l, ctx.branch)
        if l >= DENSE_LAYERS and self.cluster_layer[l]:
            e0 = self._event()
            if self.cache.update(k, s, l, ctx.branch):
                for sk in [x for x in self.subsets if x[0] == key]:
                    del self.subsets[sk]
                self.ev.append((s, l, e0, self._event()))
                self.stats["cluster_builds"] += 1
        try:
            return self._call(q, k, v, s, l, key)
        finally:
            if self.lean_drop:
                self.cache.drop(*key)
                for sk in [x for x in self.subsets if x[0] == key]:
                    del self.subsets[sk]

    def after_call(self, ctx):
        if self.free_end and ctx.step >= self.kind.shape[0] - 1 and ctx.branch == self.last_branch:
            self.plans.clear()
            self.chunk_plans.clear()
            if self.til.device.type == "cuda":
                torch.cuda.empty_cache()

    def _call(self, q, k, v, s, l, key):
        st = self.stats
        if s < DENSE_STEPS or l < DENSE_LAYERS:
            st["dense_calls"] += 1
            return None
        dh, sh = self.split(s, l)
        S, H, D = q.shape[1], q.shape[2], q.shape[3]
        if len(sh) == 0:
            st["dense_calls"] += 1
            st["dense_head_calls"] += H
            st["call_exec_density"].append(1.0)
            return None
        est = self.estimator(s, l)
        if self.mixed == "overwrite":
            return self._call_overwrite(q, k, v, s, l, key, dh, est)
        rh = self._run_heads(sh)
        full = len(rh) == H and len(dh) == 0
        if full:
            qs, ks, vs = q, k, v
        else:
            it = torch.from_numpy(rh).to(q.device)
            qs, ks, vs = (x.index_select(2, it) for x in (q, k, v))
        sp, m, nch = self._plan_entry(s, l)
        if nch <= 1:
            r = self._decoup(qs, ks, vs, sp, self._sub_clusters(key, rh), m, est)
        else:
            r = self._chunked(s, l, key, rh, qs, ks, vs, est, nch)
            st["chunked_calls"] = st.get("chunked_calls", 0) + 1
        o, ls = r if self.lse else (r, None)
        nsh, nd = len(sh), len(dh)
        st["sparse_calls"] += 1
        st["density"].append(sp.density * nsh / H + nd / H)
        st["exec_density"].append(sp.exec_density * nsh / H + nd / H)
        st["call_exec_density"].append(st["exec_density"][-1])
        st["pseudo_density"].append(sp.pseudo_density * nsh / H)
        st["dense_head_calls"] += nd
        if full:
            return r
        del qs, ks, vs
        out = torch.empty(1, S, H, D, dtype=q.dtype, device=q.device)
        ish = torch.from_numpy(sh).to(q.device)
        out[:, :, ish] = o.view(1, S, len(rh), D)[:, :, :nsh]
        lse_all = None
        if self.lse:
            lse_all = torch.empty(H, S, dtype=torch.float32, device=q.device)
            lse_all[ish] = ls[:nsh]
        if nd:
            idh = torch.from_numpy(dh).to(q.device)
            od, ld = self.dense_heads(q, k, v, dh)
            out[:, :, idh] = od.to(q.dtype)
            if self.lse:
                lse_all[idh] = ld.float()
        out = out.view(1, S, H * D)
        return (out, lse_all) if self.lse else out

    def _chunked(self, s, l, key, rh, qs, ks, vs, est, nch):
        S, D = qs.shape[1], qs.shape[3]
        outs, lses = [], []
        for ci, idx in self._chunks(rh, nch):
            hs = rh[idx]
            pc, mc = self._chunk_plan(s, l, ci, nch, hs)
            it = torch.from_numpy(idx).to(qs.device)
            qc, kc, vc = (x.index_select(2, it) for x in (qs, ks, vs))
            r = self._decoup(qc, kc, vc, pc, self._sub_clusters(key, hs), mc, est)
            o, ls = r if self.lse else (r, None)
            outs.append(o.view(1, S, len(idx), D))
            lses.append(ls)
            del qc, kc, vc
        o = torch.cat(outs, 2).reshape(1, S, len(rh) * D)
        return (o, torch.cat(lses, 0)) if self.lse else o

    def _call_overwrite(self, q, k, v, s, l, key, dh, est):
        st = self.stats
        S, H, D = q.shape[1], q.shape[2], q.shape[3]
        sp, m = self.static_plan(s, l)
        cl = self.cache.clusters(*key)
        r = self._decoup(q, k, v, sp, cl, m, est)
        st["sparse_calls"] += 1
        st["density"].append(sp.density)
        st["exec_density"].append(sp.exec_density)
        st["pseudo_density"].append(sp.pseudo_density)
        if len(dh):
            o, ls = r if self.lse else (r, None)
            idx = torch.from_numpy(dh).to(q.device)
            od, ld = self.dense_heads(q, k, v, dh)
            o = o.view(1, S, H, D).clone()
            o[:, :, idx] = od.to(o.dtype)
            if self.lse:
                ls = ls.clone()
                ls[idx] = ld.to(ls.dtype)
            nd, nt = len(dh), float(self.til.nt)
            st["exec_density"][-1] = (sp.exec_density * H - nd / nt + nd) / H
            st["density"][-1] = (sp.density * H - nd / nt + nd) / H
            st["dense_head_calls"] += nd
            o = o.view(1, S, H * D)
            r = (o, ls) if self.lse else o
        st["call_exec_density"].append(st["exec_density"][-1])
        return r


def runtime_for(adapter, plan, **overrides) -> DecoupRuntime:
    from .models.base import ModelAdapter

    kw = dict(adapter.cfg.get("runtime", {}))
    kw.update(overrides)
    if adapter.needs_lse:
        kw.setdefault("lse", True)
        kw.setdefault("dense_lse", adapter.dense_lse)
    if type(adapter).dense_heads is not ModelAdapter.dense_heads:
        kw.setdefault("dense_heads", adapter.dense_heads)
    kw.setdefault("head_dim", adapter.head_dim)
    kw.setdefault("last_branch", tuple(adapter.branches)[-1])
    return DecoupRuntime(plan, adapter.latent_grid, dense=adapter.dense, **kw)
