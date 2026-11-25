from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

_COMPILED = {}


def _ln_mod(x, scale, shift):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6) * (1 + scale) + shift


def _add_ln_mod(x, pe, scale, shift):
    x = x + pe
    return x, _ln_mod(x, scale, shift)


def _res_ln_mod(x, g, o, scale, shift):
    x = x + g * o
    return x, _ln_mod(x, scale, shift)


def _res_add_ln_mod(x, g, o, pe, scale, shift):
    x = x + g * o
    x = x + pe
    return x, _ln_mod(x, scale, shift)


def _res(x, g, o):
    return x + g * o


def compiled(fn):
    if fn not in _COMPILED:
        import torch._inductor.config as ic
        ic.emulate_precision_casts = True
        _COMPILED[fn] = torch.compile(fn, fullgraph=True, dynamic=False)
    return _COMPILED[fn]


def pin_autotuning(pins: str | None = None) -> None:
    from torch._inductor.runtime import triton_heuristics as TH
    table = {}
    if pins and os.path.exists(pins):
        with open(pins) as f:
            table = json.load(f)

    def cfg(c):
        return {**c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}

    def autotune_to_one_config(self, *args, **kwargs):
        fn = getattr(self.fn, "__name__", None) or getattr(getattr(self.fn, "fn", None), "__name__", "?")
        want = table.get(f"{os.path.basename(getattr(self, 'filename', None) or '?')}:{fn}")
        hit = [lch for lch in self.launchers if want is not None and cfg(lch.config) == want]
        if not hit:
            print(f"[dr_fast] no pinned configuration for {fn}: first of {len(self.launchers)}", file=sys.stderr)
        self.launchers = (hit or self.launchers)[:1]

    TH.CachingAutotuner.autotune_to_one_config = autotune_to_one_config


class FastNet:
    def __init__(self, net, compile: bool = True):
        self.net, self.compile = net, compile
        self._cache = {}
        self._checked = False

    def _f(self, fn):
        return compiled(fn) if self.compile else fn

    @staticmethod
    def _mod(sb, emb, ada):
        sh, sc, g = (sb.adaLN_modulation(emb) + ada).chunk(3, dim=1)
        return sh.unsqueeze(0), sc.unsqueeze(0), g.unsqueeze(0)

    @staticmethod
    def _self_attn(att, h, rope):
        q, k, v = att.cal_qkv(h, None, None, rope_emb=rope)
        return att.cal_attn(q, k, v)

    def _tables(self, inp, S, B, D):
        key = (S, B, D)
        if key not in self._cache:
            pe = inp["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"].reshape(S, B, D)
            self._cache[key] = (pe.contiguous(), inp["rope_emb_L_1_1_D"])
        return self._cache[key]

    def check_cross_attention(self, ctx: torch.Tensor) -> None:
        assert not bool(ctx.any()), "cross-attention context is not zero: FastNet is not exact for this model"
        for blk in self.net.blocks.values():
            att = blk.blocks[1].block.attn
            assert att.to_k[0].bias is None and att.to_v[0].bias is None and att.to_out[0].bias is None
        self._checked = True

    def __call__(self, x, timesteps, **cond):
        net = self.net
        inp = net.forward_before_blocks(x=x, timesteps=timesteps, **cond)
        xx = inp["x"]
        T, Hh, Ww, B, D = xx.shape
        S = T * Hh * Ww
        pe, rope = self._tables(inp, S, B, D)
        emb, ada = inp["affline_emb_B_D"], inp["adaln_lora_B_3D"]
        if not self._checked:
            self.check_cross_attention(inp["crossattn_emb"])
        xs = self._blocks(xx.reshape(S, B, D), emb, ada, pe, rope)
        x_B_T_H_W_D = xs.reshape(T, Hh, Ww, B, D).permute(3, 0, 1, 2, 4)
        return net.decoder_head(x_B_T_H_W_D=x_B_T_H_W_D, emb_B_D=emb, crossattn_emb=None,
                                origin_shape=inp["original_shape"], crossattn_mask=None, adaln_lora_B_3D=ada)

    forward = __call__

    def _blocks(self, xs, emb, ada, pe, rope):
        add_ln, res_ln, res_add_ln, res = (self._f(f) for f in (_add_ln_mod, _res_ln_mod, _res_add_ln_mod, _res))
        pending = None
        for blk in self.net.blocks.values():
            fa, _, mlp = blk.blocks
            sh, sc, g = self._mod(fa, emb, ada)
            if pending is None:
                xs, hn = add_ln(xs, pe, sc, sh)
            else:
                xs, hn = res_add_ln(xs, pending[0], pending[1], pe, sc, sh)
            o = self._self_attn(fa.block.attn, hn, rope)
            sh3, sc3, g3 = self._mod(mlp, emb, ada)
            xs, hn = res_ln(xs, g, o, sc3, sh3)
            ff = mlp.block
            pending = (g3, ff.layer2(ff.activation(ff.layer1(hn))))
        return res(xs, pending[0], pending[1])
