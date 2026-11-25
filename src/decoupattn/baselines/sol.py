from __future__ import annotations

import os
from typing import ClassVar

import torch
import torch.nn.functional as F

from ._common import BaselineHook, MatchedBaseline, add_path, bhsd, flat, upstream_path
from ._density import threshold_exact_pairs

UPSTREAM = "Sana"
BLOCK = 64
LOG2E = 1.4426950408889634


def import_sol():
    add_path(upstream_path(UPSTREAM, "techniques", "sparse_backends"))
    import sol_attn_backend
    return sol_attn_backend


def install_model_morton(adapter, backend, grid) -> int:
    tr = adapter.tr
    done = 0
    for t in (list(tr.experts) if hasattr(tr, "experts") else [tr]):
        if getattr(t, "_sol_morton", None) is not None:
            if t._sol_morton != tuple(grid):
                raise RuntimeError(f"a Morton order of grid {t._sol_morton} is already installed")
            continue
        cache: dict = {}

        def perm_inv(dev, cache=cache):
            if dev not in cache:
                cache[dev] = backend._morton3d_perm(tuple(grid), dev)
            return cache[dev]

        def active():
            return bool(getattr(getattr(adapter, "hook", None), "model_morton", False))

        def rope_forward(hidden_states, orig=t.rope.forward, perm_inv=perm_inv):
            r = orig(hidden_states)
            if not active():
                return r
            if isinstance(r, tuple):
                return tuple(x.index_select(1, perm_inv(x.device)[0]) for x in r)
            return r.index_select(2, perm_inv(r.device)[0])

        def pre(_m, args, perm_inv=perm_inv):
            return (args[0].index_select(1, perm_inv(args[0].device)[0]),) + tuple(args[1:]) if active() else None

        def post(_m, _a, out, perm_inv=perm_inv):
            return out.index_select(1, perm_inv(out.device)[1]) if active() else None

        t.rope.forward = rope_forward
        t.blocks[0].register_forward_pre_hook(pre)
        t.blocks[-1].register_forward_hook(post)
        t._sol_morton = tuple(grid)
        done += 1
    return done


class SolHook(BaselineHook):
    name = "sol"

    def __init__(self, grid, tau: float = 1.0, thresh_type: str = "diag", kernel: str = "cute", morton: str = "call",
                 account: bool = True, text_sink: bool = False, **kw):
        super().__init__(**kw)
        os.environ.setdefault("SOL_ATTN_STRICT", "1")
        self.backend = import_sol()
        if kernel == "cute":
            self.sol = self.backend._load_sol_attn()
        elif kernel == "triton_ref":
            from sol_attn.triton_ref import fwd
            self.sol = fwd.sol_attn
        else:
            raise ValueError(f"unknown Sol-Attn kernel {kernel}")
        if morton not in ("call", "model", "none"):
            raise ValueError(f"unknown Morton mode {morton}")
        self.kernel, self.grid, self.tau, self.thresh = kernel, tuple(int(x) for x in grid), float(tau), thresh_type
        self.model_morton, self.per_call = morton == "model", morton == "call"
        self.account, self.wants_text = bool(account), bool(text_sink)
        self._prepare = self._perm = None

    def _order(self, device):
        if self._perm is None or self._perm[0].device != device:
            self._perm = self.backend._morton3d_perm(self.grid, device)
        return self._perm

    def _run(self, q, k, v, **sink):
        kw = {"tau": self.tau, "thresh_type": self.thresh}
        if self.kernel == "cute":
            kw["kv_splits"] = self.backend._resolve_kv_splits(q, "auto")
        return self.sol(q, k, v, **kw, **sink)

    def _exact_pairs(self, q, k, v):
        if self._prepare is None:
            tma = True
            if self.kernel == "triton_ref":
                from sol_attn.triton_ref import fwd
                tma = fwd._use_tma(q.device)
            if tma:
                from sol_attn.preprocess import prepare
                self._prepare = lambda q, k, v, scale: prepare(q, k, v, scale=scale, tau=self.tau,
                                                               thresh_type=self.thresh)
            else:
                self._prepare = lambda q, k, v, scale: fwd.prepare_ptr(q, k, v, scale=scale, tau=self.tau,
                                                                       thresh_type=self.thresh, tokens=q.shape[1])
        _, T, H, D = q.shape
        scale = D ** -0.5
        kc, _, thr = self._prepare(q, k, v, scale)
        n = -(-T // BLOCK)
        ln = torch.full((n,), float(BLOCK), device=q.device)
        ln[-1] = T - (n - 1) * BLOCK
        qbar = F.pad(q[0], (0, 0, 0, 0, 0, n * BLOCK - T)).view(n, BLOCK, H, D).float().sum(1) / ln[:, None, None]
        score = torch.einsum("nhd,mhd->hnm", qbar, kc[0, :n].float()) * (scale * LOG2E)
        return threshold_exact_pairs(score, thr.reshape(-1)[: n * H].view(n, H).transpose(0, 1), BLOCK)

    def _record(self, q, k, v) -> None:
        H, T = q.shape[2], q.shape[1]
        total = H * T * T
        if self.account:
            ex = self._exact_pairs(q, k, v)
            self.record(ex, ex, total, total - ex.float())
        else:
            self.calls["sparse"] += 1

    def __call__(self, q, k, v, ctx):
        if self.is_dense(ctx):
            return None
        B, S_, H, D = q.shape
        if self.lse and not self.wants_text:
            raise RuntimeError("Sol-Attn returns no log-sum-exp to merge text keys with: this model needs the "
                               "text_sink path (Cosmos3-Nano) or hook.joint (HunyuanVideo)")
        inv = None
        if self.per_call:
            perm, inv = self._order(q.device)
            q, k, v = (x.index_select(1, perm) for x in (q, k, v))
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        text = getattr(ctx, "text", None) if self.wants_text else None
        if self.wants_text:
            if text is None:
                raise RuntimeError("Sol-Attn attends the text keys through its exact KV sink: the adapter has to pass "
                                   "them as ctx.text = (k_text, v_text)")
            kt, vt = text
            T = kt.shape[1]
            qj = torch.cat([q, q.new_zeros(B, T, H, D)], 1).contiguous()
            with self.seg("sol"):
                o = self._run(qj, torch.cat([k, kt], 1).contiguous(), torch.cat([v, vt], 1).contiguous(),
                              sink_start=S_, sink_tokens=T)[:, :S_]
        else:
            with self.seg("sol"):
                o = self._run(q, k, v)
        self._record(q, k, v)
        o = o.reshape(B, S_, H, D)
        if inv is not None:
            o = o.index_select(1, inv)
        o = o.reshape(B, S_, H * D)
        return (o, None) if self.wants_text else o

    def joint(self, q, k, v, nv, ctx):
        if self.is_dense(ctx):
            return None
        valid = torch.ones(1, q.shape[1], dtype=torch.bool, device=q.device)
        with self.seg("sol"):
            o = self.backend._run_mmdit_sol_attn_bthd(bhsd(q), bhsd(k), bhsd(v), video_len=nv, key_valid=valid,
                                                      tau=self.tau, thresh_type=self.thresh, kv_splits="auto",
                                                      grid=self.grid, morton=False)
        self._record(q[:, :nv].contiguous(), k[:, :nv].contiguous(), v[:, :nv].contiguous())
        return flat(o)


HOOK_KEYS = ("tau", "thresh_type", "kernel", "morton", "account", "text_sink")


class Sol(MatchedBaseline):
    name = "sol"
    label = "Sol-Attn"
    upstream = (UPSTREAM,)
    knob = "tau"
    knob_sign = -1
    timing_settings: ClassVar[dict] = {"account": False}

    def make_hook(self, adapter, settings: dict):
        kw = {k: settings[k] for k in HOOK_KEYS if k in settings}
        hook = SolHook(adapter.latent_grid, lse=bool(getattr(adapter, "needs_lse", False)), **kw)
        if hook.model_morton:
            getattr(adapter, "ensure_loaded", lambda: None)()
            install_model_morton(adapter, hook.backend, hook.grid)
        return hook


BASELINE = Sol
