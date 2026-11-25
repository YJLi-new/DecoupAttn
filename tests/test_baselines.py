from __future__ import annotations

import contextlib
import math
import os
import sys
import types
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from decoupattn.baselines import METHODS, _common, _density, get_baseline, pisa, sol, sparsepr, svg2, svgear
from decoupattn.baselines import _svg as svg_glue
from decoupattn.context import CallContext
from decoupattn.models import MODELS

MINE = ("svg2", "svgear", "sol", "pisa", "sparsepr")
MATCHED = ("svg2", "svgear", "sol", "sparsepr")
H, S, D = 2, 48, 8


def qkv(s=S, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(1, s, H, D, generator=g) for _ in range(3))


def masked_attention(q, k, v, mask=None):
    logits = torch.einsum("shd,thd->hst", q[0], k[0]) * D ** -0.5
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    out = torch.einsum("hst,thd->shd", torch.softmax(logits, -1), v[0])
    return out.reshape(1, q.shape[1], -1), torch.logsumexp(logits, -1)


def ctx(step=1, layer=1, **kw):
    c = CallContext(step=step, layer=layer)
    for k_, v_ in kw.items():
        setattr(c, k_, v_)
    return c


@pytest.mark.parametrize("method", MINE)
@pytest.mark.parametrize("model", list(MODELS))
def test_settings_of_every_model(method, model):
    b = get_baseline(method, model)
    assert b.name == method and b.label == METHODS[method][1]
    s = b.settings
    if method in MATCHED:
        lo, hi = s["match_range"]
        assert lo <= s[b.knob] <= hi
        assert s["match_band"][0] < 0.2 < s["match_band"][1]
        assert s["match_scheme"] in ("interp", "bisect")
    else:
        assert s["density"] == pytest.approx(0.2)


def test_overrides_replace_single_settings():
    assert get_baseline("svg2", "cp25", top_p=0.5).settings["top_p"] == 0.5
    assert get_baseline("svg2", "cp25").settings["top_p"] != 0.5


@pytest.mark.parametrize("method", MINE)
def test_upstream_sources(method):
    for name in get_baseline(method, "cp25").upstream:
        src = _common.load_source(name)
        assert len(src["commit"]) == 40 and method in src["used_by"]
        assert src["url"].startswith("https://") and src["license"]
        for p in src["patches"]:
            assert os.path.isfile(os.path.join(_common.SOURCES_DIR, name, p))


def test_model_specific_paths():
    assert get_baseline("sol", "wan21").settings["morton"] == "model"
    assert get_baseline("sol", "nano").settings["text_sink"] and get_baseline("sparsepr", "nano").settings["text_keys"]
    for m in ("svg2", "svgear", "sparsepr"):
        assert get_baseline(m, "nano").settings["per_branch"]
    assert get_baseline("sparsepr", "dr").settings["route_refresh_every"] > 1


def test_first_step_and_first_layer_stay_dense():
    h = _common.BaselineHook()
    assert h.is_dense(ctx(0, 5)) and h.is_dense(ctx(3, 0)) and not h.is_dense(ctx(1, 1))
    h.record(10, 20, 100, 5)
    s = h.summary()
    assert (s["dense_calls"], s["sparse_calls"]) == (2, 1)
    assert (s["real_density"], s["exec_density"], s["pseudo_density"]) == pytest.approx((0.1, 0.2, 0.05))
    h.reset()
    assert h.summary()["exec_density"] is None


def test_unpermute_rows_and_layouts():
    x = torch.randn(1, S, H, D)
    assert torch.equal(_common.flat(_common.bhsd(x)), x.reshape(1, S, H * D))
    idx = torch.stack([torch.randperm(S) for _ in range(H)])
    vals = torch.randn(H, S)
    assert torch.equal(_common.unpermute_rows(torch.gather(vals, 1, idx), idx), vals)


def test_seeded_stream_is_reproducible_and_leaves_the_global_one():
    torch.manual_seed(7)
    expect = torch.rand(3)
    torch.manual_seed(7)
    with _common.seeded(torch.device("cpu"), 2, 3):
        a = torch.rand(3)
    after = torch.rand(3)
    with _common.seeded(torch.device("cpu"), 2, 3):
        b = torch.rand(3)
    assert torch.equal(a, b) and torch.equal(after, expect)


def test_density_accounting():
    bm = torch.tensor([[[[True, False], [True, True]]]])
    qsz, ksz = torch.tensor([[[100, 200]]]), torch.tensor([[[50, 150]]])
    assert int(_density.selected_pairs(bm, qsz, ksz)) == 100 * 50 + 200 * 200
    assert int(_density.vb_exec_pairs(bm, qsz, ksz)) == 128 * 96 + 256 * 288
    assert int(_density.pseudo_key_pairs(bm, qsz, ksz)) == 100
    assert _density.block_topk(640, 64, 0.2) == (10, 2)
    assert _density.block_topk_exec_pairs(640, 3, 64, 0.2) == 3 * 640 * 128
    score = torch.zeros(1, 4, 4)
    score[0, 0, 3] = 2.0
    exact = _density.threshold_exact_pairs(score, torch.ones(1, 4), 64)
    assert int(exact) == (10 + 1) * 64 * 64


@pytest.mark.parametrize("scheme", ["interp", "bisect"])
@pytest.mark.parametrize("sign", [1, -1])
def test_match_density_reaches_the_band(scheme, sign):
    def density(x):
        return 0.05 + 0.4 * x if sign > 0 else 0.45 - 0.4 * x
    saved = []
    r = _common.match_density(density, 0.0, 1.0, 0.9, sign, iters=12, tried=None, save=saved.append, scheme=scheme)
    assert r["in_band"] and 0.195 <= density(r["chosen"]) <= 0.205
    assert len(saved) == len(r["tried"])
    again = _common.match_density(density, 0.0, 1.0, 0.9, sign, iters=12, tried=r["tried"], scheme=scheme)
    assert again["chosen"] == r["chosen"] and len(again["tried"]) == len(r["tried"])


def test_match_density_keeps_the_closest_point_out_of_band():
    r = _common.match_density(lambda x: 0.3, 0.0, 1.0, 0.5, 1, iters=3)
    assert not r["in_band"] and r["exec_density"] == 0.3 and len(r["tried"]) <= 3


class _Counting(_common.BaselineHook):
    name = "counting"

    def __call__(self, q, k, v, c):
        if self.is_dense(c):
            return None
        self.record(1, 2, 10)
        return q


def test_per_branch_keeps_one_state_per_branch():
    h = _common.PerBranch(_Counting, ("cond", "uncond"))
    for br in ("cond", "uncond", "cond"):
        h(None, None, None, ctx(branch=br))
    assert h.m["cond"].calls["sparse"] == 2 and h.m["uncond"].calls["sparse"] == 1
    s = h.summary()
    assert s["sparse_calls"] == 3 and s["exec_density"] == pytest.approx(0.2) and s["per_branch"]
    h.reset()
    assert h.summary()["sparse_calls"] == 0


class _Adapter:
    name, branches, needs_lse, latent_grid = "fake", ("cond",), False, (3, 4, 4)

    def __init__(self, densities):
        self.densities, self.generated = densities, []

    def clips(self, split):
        return [f"{split}_{i}" for i in range(5)]

    def generate(self, clip, hook, seed):
        hook.record(0, self.densities(hook.value) * 1000, 1000)
        self.generated.append((clip, hook.value))


def test_rematch_writes_the_knob_the_hook_then_uses(tmp_path):
    class B(_common.MatchedBaseline):
        name, knob = "svg2", "top_p"

        def make_hook(self, adapter, settings):
            h = _common.BaselineHook()
            h.value = settings["top_p"]
            return h

    b = B("cp1", match_clips=2)
    a = _Adapter(lambda x: 0.4 * x)
    assert b.hook(a, str(tmp_path)).value == b.settings["top_p"]
    res = b.rematch(a, str(tmp_path), log=lambda *_: None)
    assert res["in_band"] and {c for c, _ in a.generated} == {"calib_0", "calib_1"}
    assert b.hook(a, str(tmp_path)).value == res["chosen"] == pytest.approx(0.5, abs=0.02)
    n = len(a.generated)
    b.rematch(a, str(tmp_path), log=lambda *_: None)
    assert len(a.generated) == n


def _token_mask(bm, qsz, ksz):
    rows = torch.repeat_interleave(torch.arange(len(qsz)), qsz)
    cols = torch.repeat_interleave(torch.arange(len(ksz)), ksz)
    return bm[rows][:, cols]


class _Wrapper:
    def __init__(self, workspace, backend="auto"):
        self.planned = None

    def plan(self, block_mask_map, block_row_sz, block_col_sz, **_):
        self.planned = (block_mask_map, block_row_sz, block_col_sz)

    def run(self, q, k, v, return_lse=False):
        bm, qs, ks = self.planned
        outs, lses = [], []
        for h in range(q.shape[0]):
            mask = _token_mask(bm[h], qs[h], ks[h]) if bm is not None else None
            o, lse = masked_attention(q[h][None, :, None], k[h][None, :, None], v[h][None, :, None],
                                      None if mask is None else mask[None])
            outs.append(o[0])
            lses.append(lse[0] / math.log(2))
        o = torch.stack(outs)
        return (o, torch.stack(lses)) if return_lse else o


def _kmeans(x, n_clusters, max_iters=100, init_centroids=None, **_):
    b, n, d = x.shape
    labels = torch.stack([(torch.arange(n) * 5 + 3 * i) % n_clusters for i in range(b)])
    labels = torch.where(labels == n_clusters - 1, 0, labels)
    sizes = torch.stack([torch.bincount(l_, minlength=n_clusters) for l_ in labels])
    cen = torch.zeros(b, n_clusters, d).index_put_((torch.arange(b)[:, None].expand(b, n), labels), x, accumulate=True)
    return labels, cen / sizes.clamp(min=1)[..., None], sizes, max_iters


def _permute(t, labels, dim=2, sorted_indices=None):
    b, h, s, d = t.shape
    if sorted_indices is None:
        sorted_indices = torch.argsort(labels, dim=-1, stable=True).to(torch.int32)
    idx = sorted_indices.long().view(b, h, s, 1).expand(b, h, s, d)
    return torch.gather(t, 2, idx), sorted_indices


def _inv_permute(t, sorted_indices, dim=2):
    b, h, s, d = t.shape
    return torch.empty_like(t).scatter_(2, sorted_indices.long().view(b, h, s, 1).expand(b, h, s, d), t)


def _top_map(qcen, kcen, share):
    score = torch.einsum("bhqd,bhkd->bhqk", qcen, kcen)
    keep = max(1, round(share * score.shape[-1]))
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, score.topk(keep, -1).indices, True)


class _FusedKernel:
    def __getitem__(self, grid):
        def run(*, Q, K_centroids, V_centroids, block_mask_map, block_col_sz, O_flash, LSE_flash, QC_INDPTR, LSE_final,
                Out, sm_scale, **_):
            bm, ksz, ptr = block_mask_map.flatten(0, 1), block_col_sz.flatten(0, 1), QC_INDPTR.flatten(0, 1)
            for h in range(Q.shape[0]):
                cid = torch.bucketize(torch.arange(Q.shape[1]), ptr[h, 1:].long(), right=True)
                pl = Q[h] @ K_centroids[h].T * sm_scale + torch.log(ksz[h].float())
                pl = pl.masked_fill(bm[h][cid] | (ksz[h] == 0), float("-inf"))
                lf = LSE_flash[h] * math.log(2)
                tot = torch.logaddexp(lf, torch.logsumexp(pl, -1))
                Out[h] = torch.exp(lf - tot)[:, None] * O_flash[h] + torch.exp(pl - tot[:, None]) @ V_centroids[h]
                LSE_final[h] = tot
        return run


def _fake_svg(calls):
    def kmeans(x, n_clusters, max_iters=100, init_centroids=None, **kw):
        calls.append((max_iters, init_centroids is not None))
        return _kmeans(x, n_clusters, max_iters, init_centroids)

    def scatter_mean(src, size):
        _, h, s, d = src.shape
        lab = torch.stack([torch.repeat_interleave(torch.arange(size.shape[-1]), size[0, i]) for i in range(h)])
        out = torch.zeros(h, size.shape[-1], d).index_put_((torch.arange(h)[:, None].expand(h, s), lab), src[0],
                                                           accumulate=True)
        return (out / size[0].clamp(min=1)[..., None])[None]

    ku = types.SimpleNamespace(
        batch_kmeans_Euclid=kmeans, scatter_mean_fused=scatter_mean, _fused_qc_kernel_opt=_FusedKernel(),
        identify_dynamic_map=lambda qc, kc, qsz, ksz, p, min_kc_ratio=0: _top_map(qc, kc, p),
        identify_dynamic_map_estimated=lambda q, k, v, qsz, ksz, qc, kc, vc, top_p, gamma, min_kc_ratio:
            _top_map(qc, kc, top_p))
    return types.SimpleNamespace(ku=ku, patch_ctx=contextlib.nullcontext, permute=_permute, inv_permute=_inv_permute,
                                 wrapper_cls=_Wrapper)


@pytest.fixture
def svg_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(svg_glue, "import_svg", lambda: _fake_svg(calls))
    monkeypatch.setattr(svg_glue.VarBlockSparse, "FLOAT_WS", 16)
    return calls


def _cluster_reference(q, k, v, qc, kc, share, pseudo):
    outs, lses, vcens = [], [], None
    ql, qcen, qsz, _ = _kmeans(_common.bhsd(q)[0], qc)
    kl, kcen, ksz, _ = _kmeans(_common.bhsd(k)[0], kc)
    bm = _top_map(qcen[None], kcen[None], share)[0]
    if pseudo:
        vcens = torch.zeros(H, kc, D).index_put_((torch.arange(H)[:, None].expand(H, S), kl), _common.bhsd(v)[0],
                                                 accumulate=True) / ksz.clamp(min=1)[..., None]
    for h in range(H):
        kh, vh, mask = k[0, :, h], v[0, :, h], bm[h][ql[h]][:, kl[h]]
        if pseudo:
            un = ~bm[h][ql[h]] & (ksz[h] > 0)
            logits = torch.cat([(q[0, :, h] @ kh.T * D ** -0.5).masked_fill(~mask, float("-inf")),
                                (q[0, :, h] @ kcen[h].T * D ** -0.5 + torch.log(ksz[h].float())).masked_fill(
                                    ~un, float("-inf"))], 1)
            vals = torch.cat([vh, vcens[h]])
        else:
            logits, vals = (q[0, :, h] @ kh.T * D ** -0.5).masked_fill(~mask, float("-inf")), vh
        outs.append(torch.softmax(logits, -1) @ vals)
        lses.append(torch.logsumexp(logits, -1))
    return torch.stack(outs, 1).reshape(1, S, H * D), torch.stack(lses), (bm, qsz, ksz)


def test_svg2_matches_the_routed_reference(svg_calls):
    q, k, v = qkv()
    hook = svg2.SVG2Hook(top_p=0.5, num_q_centroids=4, num_k_centroids=6, lse=True)
    assert hook(q, k, v, ctx(0, 1)) is None and hook(q, k, v, ctx(1, 0)) is None
    out, lse = hook(q, k, v, ctx(1, 1))
    ref, ref_lse, (bm, qsz, ksz) = _cluster_reference(q, k, v, 4, 6, 0.5, pseudo=False)
    assert torch.allclose(out, ref, atol=1e-5) and torch.allclose(lse, ref_lse, atol=1e-5)
    s = hook.summary()
    assert s["real_density"] == pytest.approx(float(_density.selected_pairs(bm, qsz, ksz)) / (H * S * S))
    assert s["exec_density"] >= s["real_density"] and (s["dense_calls"], s["sparse_calls"]) == (2, 1)


def test_svg2_warm_starts_within_a_clip_and_restarts_after_reset(svg_calls):
    q, k, v = qkv()
    hook = svg2.SVG2Hook(top_p=1.0, num_q_centroids=4, num_k_centroids=6, kmeans_iter_init=50, kmeans_iter_step=2)
    out = hook(q, k, v, ctx(1, 1))
    assert torch.allclose(out, masked_attention(q, k, v)[0], atol=1e-5)
    hook(q, k, v, ctx(2, 1))
    hook.reset()
    hook(q, k, v, ctx(1, 1))
    assert [c for c in svg_calls[::2]] == [(50, False), (2, True), (50, False)]


def test_svgear_adds_one_pseudo_key_per_unselected_cluster(svg_calls):
    q, k, v = qkv(seed=1)
    hook = svgear.EARHook(top_p=0.5, num_q_centroids=4, num_k_centroids=6, lse=True, route_head_chunk=1)
    out, lse = hook(q, k, v, ctx(1, 1))
    ref, ref_lse, (bm, qsz, ksz) = _cluster_reference(q, k, v, 4, 6, 0.5, pseudo=True)
    assert torch.allclose(out, ref, atol=1e-5) and torch.allclose(lse, ref_lse, atol=1e-5)
    s = hook.summary()
    assert s["pseudo_density"] == pytest.approx(float(_density.pseudo_key_pairs(bm, qsz, ksz)) / (H * S * S))


@pytest.mark.parametrize("cls,mod", [(svg2.SVG2Hook, svg2), (svgear.EARHook, svgear)])
def test_svg_joint_keeps_the_text_tokens_exact(svg_calls, cls, mod):
    q, k, v = qkv(S + 5, seed=2)
    hook = cls(top_p=1.0, num_q_centroids=4, num_k_centroids=6, zero_step_kmeans_init=True)
    assert hook.joint(q, k, v, S, ctx(0, 1)) is None
    out = hook.joint(q, k, v, S, ctx(1, 1))
    assert torch.allclose(out, masked_attention(q, k, v)[0], atol=1e-5)
    assert svg_calls[0][1] is False and svg_calls[2][1] is True


def test_matched_baselines_build_per_branch_hooks(svg_calls):
    a = _Adapter(None)
    a.branches, a.needs_lse = ("cond", "uncond"), True
    hook = get_baseline("svg2", "nano").hook(a)
    assert isinstance(hook, _common.PerBranch) and all(h.lse for h in hook.m.values())
    assert not isinstance(get_baseline("svgear", "cp25").hook(a), _common.PerBranch)


def _fake_sol_backend(log):
    def kernel(q, k, v, *, tau, thresh_type, kv_splits=None, sink_tokens=0, sink_start=None):
        log.append({"tau": tau, "kv_splits": kv_splits, "sink": (sink_start, sink_tokens)})
        return masked_attention(q, k, v)[0].view(q.shape)

    def morton(grid, device):
        perm = torch.randperm(grid[0] * grid[1] * grid[2], generator=torch.Generator().manual_seed(3))
        return perm, torch.argsort(perm)

    def mmdit(q, k, v, *, video_len, key_valid, tau, thresh_type, kv_splits, grid, morton):
        log.append({"mmdit": video_len, "morton": morton})
        qs, ks, vs = (x.permute(0, 2, 1, 3) for x in (q, k, v))
        return masked_attention(qs, ks, vs)[0].view(qs.shape).permute(0, 2, 1, 3)

    return types.SimpleNamespace(_load_sol_attn=lambda: kernel, _resolve_kv_splits=lambda q, mode: 1,
                                 _morton3d_perm=morton, _run_mmdit_sol_attn_bthd=mmdit)


@pytest.fixture
def sol_log(monkeypatch):
    log = []
    monkeypatch.setattr(sol, "import_sol", lambda: _fake_sol_backend(log))
    pre = types.ModuleType("sol_attn.preprocess")

    def prepare(q, k, v, *, scale, tau, thresh_type):
        n = -(-q.shape[1] // 64)
        kc = F.pad(k, (0, 0, 0, 0, 0, n * 64 - k.shape[1])).view(1, n, 64, *k.shape[2:]).mean(2)
        return kc, None, torch.full((1, n, q.shape[2]), 1e9)

    pre.prepare = prepare
    monkeypatch.setitem(sys.modules, "sol_attn", types.ModuleType("sol_attn"))
    monkeypatch.setitem(sys.modules, "sol_attn.preprocess", pre)
    return log


def test_sol_orders_tokens_per_call_and_counts_exact_tiles(sol_log):
    q, k, v = qkv(256)
    hook = sol.SolHook((4, 8, 8), tau=1.2)
    assert hook(q, k, v, ctx(0, 3)) is None
    out = hook(q, k, v, ctx(1, 1))
    assert torch.allclose(out, masked_attention(q, k, v)[0], atol=1e-5)
    assert sol_log[0]["tau"] == 1.2 and sol_log[0]["kv_splits"] == 1
    assert hook.summary()["exec_density"] == pytest.approx(10 / 16)
    quiet = sol.SolHook((4, 8, 8), tau=1.2, account=False)
    quiet(q, k, v, ctx(1, 1))
    assert quiet.summary()["sparse_calls"] == 1 and quiet.summary()["exec_density"] is None


def test_sol_timing_hook_runs_without_density_accounting(sol_log):
    b = get_baseline("sol", "cp25")
    a = _Adapter(None)
    a.latent_grid = (4, 8, 8)
    assert b.hook(a).account and not b.timing_hook(a).account
    assert b.settings["account"]
    q, k, v = qkv(256)
    hook = b.timing_hook(a)
    hook(q, k, v, ctx(1, 1))
    assert hook.summary()["exec_density"] is None


def test_sol_text_sink_and_joint_paths(sol_log):
    q, k, v = qkv(256)
    kt, vt = torch.randn(1, 7, H, D), torch.randn(1, 7, H, D)
    hook = sol.SolHook((4, 8, 8), text_sink=True, lse=True)
    out, lse = hook(q, k, v, ctx(text=(kt, vt)))
    assert lse is None and sol_log[-1]["sink"] == (256, 7)
    assert torch.allclose(out, masked_attention(q, torch.cat([k, kt], 1), torch.cat([v, vt], 1))[0], atol=1e-5)
    with pytest.raises(RuntimeError):
        hook(q, k, v, ctx())
    with pytest.raises(RuntimeError):
        sol.SolHook((4, 8, 8), lse=True)(q, k, v, ctx())
    qj, kj, vj = qkv(256 + 9, seed=4)
    joint = sol.SolHook((4, 8, 8), morton="none").joint(qj, kj, vj, 256, ctx())
    assert torch.allclose(joint, masked_attention(qj, kj, vj)[0], atol=1e-5)
    assert sol_log[-1] == {"mmdit": 256, "morton": False}


def test_sol_model_level_morton_is_active_only_under_its_hook(sol_log):
    class Block(torch.nn.Module):
        def forward(self, x):
            self.seen = x
            return x + 0

    class Rope(torch.nn.Module):
        def forward(self, x):
            return torch.arange(12.0).view(1, 1, 12, 1)

    tr = torch.nn.Module()
    tr.rope, tr.blocks = Rope(), torch.nn.ModuleList([Block(), Block()])
    a = _Adapter(None)
    a.tr, a.latent_grid, a.hook = tr, (1, 3, 4), None
    hook = get_baseline("sol", "wan21").hook(a)
    assert hook.model_morton and not hook.per_call and tr._sol_morton == (1, 3, 4)
    perm, _ = hook.backend._morton3d_perm((1, 3, 4), None)
    x = torch.randn(1, 12, 4)

    def run():
        return tr.blocks[1](tr.blocks[0](x)), tr.rope(x)

    out, rope = run()
    assert torch.equal(tr.blocks[0].seen, x) and torch.equal(rope.flatten(), torch.arange(12.0))
    a.hook = hook
    out, rope = run()
    assert torch.equal(tr.blocks[0].seen, x[:, perm]) and torch.equal(out, x)
    assert torch.equal(rope.flatten(), perm.float())
    get_baseline("sol", "wan21").hook(a)
    assert torch.equal(run()[0], x) and torch.equal(tr.blocks[0].seen, x[:, perm])


@pytest.fixture
def pisa_log(monkeypatch, tmp_path):
    log = []

    def fn(q, k, v, density=0.1, block_size=64, scale=None, use_bias=False, sink_idx=None, return_lse=False):
        log.append({"density": density, "block": block_size, "use_bias": use_bias, "sink_idx": sink_idx})
        qs, ks, vs = (x.permute(0, 2, 1, 3) for x in (q, k, v))
        o, lse = masked_attention(qs, ks, vs)
        o = o.view(qs.shape).permute(0, 2, 1, 3)
        return (o, lse[None]) if return_lse else o

    mod = types.ModuleType("piecewise_attn")
    mod.piecewise_sparse_attention_hyd = fn
    monkeypatch.setitem(sys.modules, "piecewise_attn", mod)
    monkeypatch.setattr(pisa, "upstream_path", lambda *a: str(tmp_path))
    return log


def test_pisa_passes_the_settings_and_returns_the_lse_when_needed(pisa_log):
    q, k, v = qkv(128)
    a = _Adapter(None)
    hook = get_baseline("pisa", "dr", density=0.5).hook(a)
    assert hook(q, k, v, ctx(0, 1)) is None
    out = hook(q, k, v, ctx())
    ref, ref_lse = masked_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)
    assert pisa_log[-1] == {"density": 0.5, "block": 64, "use_bias": False, "sink_idx": None}
    assert hook.summary()["exec_density"] == pytest.approx(0.5)
    a.needs_lse = True
    out, lse = get_baseline("pisa", "nano").hook(a)(q, k, v, ctx())
    assert torch.allclose(lse, ref_lse, atol=1e-5) and pisa_log[-1]["use_bias"] is True
    get_baseline("pisa", "hy").hook(a).joint(q, k, v, 100, ctx())
    assert pisa_log[-1]["sink_idx"] == -1


@dataclass
class _Cfg:
    target_density: float = 0.22
    num_q_centroids: int = 4
    num_k_centroids: int = 6
    kmeans_iter_init: int = 25
    kmeans_iter_step: int = 2
    selector_policy: str = "svg_ear_value"
    selector_budget_mode: str = "fixed_density"
    probe_rows: int = 2
    repair_rank: int = 16
    repair_ridge: float = 0.1


@dataclass
class _RouteOut:
    q_labels: torch.Tensor
    k_sorted_indices: torch.Tensor
    k_permuted: torch.Tensor
    v_permuted: torch.Tensor
    video_dynamic_map: torch.Tensor
    q_cluster_sizes: torch.Tensor
    k_cluster_sizes: torch.Tensor
    permuted_includes_suffix: bool = False
    probe_rows: torch.Tensor | None = None
    probe_weights: torch.Tensor | None = None
    q_role_features: torch.Tensor | None = None


ROUTES: list = []


class _Core:
    def __init__(self, cfg):
        self.cfg = cfg

    def route(self, q, k, v, *, total_key_tokens, **_):
        ROUTES.append(self)
        c = self.cfg
        ql, _, qsz, _ = _kmeans(q[0], c.num_q_centroids)
        kl, _, ksz, _ = _kmeans(k[0], c.num_k_centroids)
        kp, k_idx = _permute(k, kl)
        vp, _ = _permute(v, None, sorted_indices=k_idx)
        full = torch.ones(q.shape[1], c.num_q_centroids, c.num_k_centroids, dtype=torch.bool)
        return _RouteOut(ql, k_idx, kp, vp, full, qsz, ksz)

    def repair(self, base, q, k, v, route):
        route.probe_rows = route.probe_weights = torch.zeros(1)
        return base, {"probe_rows": self.cfg.probe_rows}


@pytest.fixture
def spr(monkeypatch):
    def fwd(q, k, v, bm, qsz, ksz, is_cpu=False, plan_cache_key=None, plan_cache_token=None):
        w = _Wrapper(None)
        w.plan(bm[0], qsz[0], ksz[0])
        return w.run(q[0], k[0], v[0])[None]

    state = {"cleared": 0}
    ku = types.SimpleNamespace(dynamic_block_sparse_fwd_flashinfer=fwd,
                               clear_flashinfer_plan_cache=lambda: state.__setitem__("cleared", state["cleared"] + 1))
    monkeypatch.setattr(sparsepr, "import_sparsepr", lambda ext=False: (ku, _permute, _inv_permute, _Cfg, _Core))
    ROUTES.clear()
    return state


@pytest.mark.parametrize("refresh,routes", [(1, 3), (4, 1)])
def test_sparsepr_refreshes_or_reuses_its_route(spr, refresh, routes):
    hook = sparsepr.SparsePRHook(0.15, route_refresh_every=refresh, num_q_centroids=4, num_k_centroids=6, probe_rows=2)
    assert hook.cfg.target_density == 0.15
    for step in (1, 2, 3):
        q, k, v = qkv(seed=step)
        assert hook(q, k, v, ctx(0, 1)) is None
        out = hook(q, k, v, ctx(step, 1))
        assert torch.allclose(out, masked_attention(q, k, v)[0], atol=1e-5)
    assert len(ROUTES) == routes
    s = hook.summary()
    assert s["exec_density"] > 1.0 and s["real_density"] == pytest.approx(1.0 + 2 / S)
    cleared = spr["cleared"]
    hook.reset()
    assert not hook.cores and not hook.routes and spr["cleared"] == cleared + 1


def test_sparsepr_needs_a_text_path_on_models_that_merge_text_keys(spr):
    q, k, v = qkv()
    with pytest.raises(RuntimeError):
        sparsepr.SparsePRHook(0.15, lse=True)(q, k, v, ctx())
    with pytest.raises(RuntimeError):
        sparsepr.SparsePRHook(0.15, text_keys=True)(q, k, v, ctx())
    a = _Adapter(None)
    a.branches, a.needs_lse = ("cond", "uncond"), True
    hook = get_baseline("sparsepr", "nano").hook(a)
    assert isinstance(hook, _common.PerBranch) and hook.wants_text and hook.m["cond"].gqa == 4
