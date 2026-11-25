from __future__ import annotations

from ._common import Baseline, BaselineHook, add_path, bhsd, flat, upstream_path
from ._density import block_topk_exec_pairs

UPSTREAM = "piecewise-sparse-attention"


class PISAHook(BaselineHook):
    name = "pisa"

    def __init__(self, density: float = 0.2, block_size: int = 64, use_bias: bool = True, **kw):
        super().__init__(**kw)
        add_path(upstream_path(UPSTREAM))
        from piecewise_attn import piecewise_sparse_attention_hyd
        self.f = piecewise_sparse_attention_hyd
        self.density, self.block, self.use_bias = float(density), int(block_size), bool(use_bias)

    def _record(self, tokens: int, heads: int) -> None:
        total = heads * tokens * tokens
        self.record(self.density * total, block_topk_exec_pairs(tokens, heads, self.block, self.density), total,
                    (1.0 - self.density) * total)

    def __call__(self, q, k, v, ctx):
        if self.is_dense(ctx):
            return None
        S_, H = q.shape[1], q.shape[2]
        if self.lse:
            o, lse = self.f(bhsd(q), bhsd(k), bhsd(v), self.density, self.block, use_bias=self.use_bias,
                            return_lse=True)
        else:
            o = self.f(bhsd(q), bhsd(k), bhsd(v), self.density, self.block, use_bias=self.use_bias)
        self._record(S_, H)
        return (flat(o), lse[0]) if self.lse else flat(o)

    def joint(self, q, k, v, nv, ctx):
        if self.is_dense(ctx):
            return None
        o = self.f(bhsd(q), bhsd(k), bhsd(v), self.density, self.block, use_bias=self.use_bias, sink_idx=-1)
        self._record(q.shape[1], q.shape[2])
        return flat(o)


class PISA(Baseline):
    name = "pisa"
    label = "PISA"
    upstream = (UPSTREAM,)

    def hook(self, adapter, run_dir: str | None = None):
        s = self.settings
        return PISAHook(s["density"], s["block_size"], s["use_bias"], lse=bool(getattr(adapter, "needs_lse", False)))


BASELINE = PISA
