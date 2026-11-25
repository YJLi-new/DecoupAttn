from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from decoupattn.config import load_config, repo_path
from decoupattn.models import adapter_class
from decoupattn.planner.pipeline import n_clips

VIDEO = ("cp25", "wan21", "wan22", "hy", "nano")
TEST_CLIPS = {"cp25": 32, "wan21": 32, "wan22": 32, "hy": 8, "nano": 32}
STEPS = {"cp25": 4, "wan21": 4, "wan22": 4, "hy": 50, "nano": 35}


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DECOUPATTN_DATA_ROOT", str(tmp_path))
    return tmp_path


def adapter(key, backend="torch"):
    cfg = load_config(key)
    cfg["model"]["dense_backend"] = backend
    return adapter_class(key)(cfg)


@pytest.mark.parametrize("key", VIDEO)
def test_config_matches_adapter(key, data_root):
    cfg = load_config(key)
    A = adapter_class(key)
    att, plan, smp = cfg["attention"], cfg["plan"], cfg["sampler"]
    assert cfg["model"]["key"] == A.name == key and cfg["model"]["dense_backend"] == "fa3"
    assert A.steps == smp["steps"] == STEPS[key]
    assert tuple(att["latent_grid"]) == A.latent_grid
    assert (att["n_layers"], att["n_heads"], att["head_dim"]) == (A.n_layers, A.n_heads, A.head_dim)
    assert plan["layout"] == A.layout and tuple(plan["rows"]) == A.rows and tuple(smp["branches"]) == A.branches
    assert (tuple(plan["block_shape"]) if "block_shape" in plan else None) == A.block_shape
    assert A.needs_lse == ("text_merge" in att)
    assert cfg["eval"]["ssim"] == "uniform"
    for v in list(cfg["checkpoints"].values()) + [cfg["data"]["clips"], cfg["data"]["work"]]:
        assert v.startswith(str(data_root)) and "@DATA_ROOT@" not in v


@pytest.mark.parametrize("key", VIDEO)
def test_sensitivity_sources(key):
    cfg = load_config(key)
    A = adapter_class(key)
    ps = cfg["planner"]["sensitivity"]
    assert ps["kind"] == "injection" and "path" not in ps
    steps = [s for a, b in ps["step_groups"] for s in range(a, b + 1)]
    assert steps == list(range(1, A.steps))
    from decoupattn.planner import sensitivity
    from decoupattn.planner.units import Units
    sensitivity.check_step_groups(Units(A.layout, A.steps, A.rows, A.n_layers, 1), ps["step_groups"])
    if A.steps == 4:
        assert ps["step_groups"] == [[1, 1], [2, 2], [3, 3]]


@pytest.mark.parametrize("key", VIDEO)
def test_splits(key):
    cfg = load_config(key)
    with open(repo_path(cfg["data"]["splits"])) as f:
        sp = json.load(f)
    a = adapter(key)
    assert sp["seed"] == 1000
    assert len(a.clips("test")) == TEST_CLIPS[key]
    assert len(a.clips("val")) == 8
    calib = a.clips("calib")
    assert 0 < len(calib) <= 16
    assert all(len(calib) >= n_clips(a, k) for k in ("distortion_clips", "sens_clips"))
    sets = [set(calib), set(a.clips("val")), set(a.clips("test"))]
    assert sum(map(len, sets)) == len(set().union(*sets))


def test_imports_without_gpu_packages():
    code = (
        "import sys, importlib.abc\n"
        "BLOCK = ('diffusers', 'transformers', 'triton', 'flash_attn', 'flash_attn_interface', 'flashinfer',\n"
        "         'fastvideo', 'cosmos_predict2', 'cosmos_oss', 'safetensors', 'PIL', 'peft')\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, B())\n"
        "from decoupattn.models import adapter_class\n"
        "for k in ('cp25', 'wan21', 'wan22', 'hy', 'nano'):\n"
        "    adapter_class(k)\n"
        "import decoupattn.models._wanapi, decoupattn.models._wanapi_wan, decoupattn.models._hunyuan\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def _ref(q, k, v):
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / q.shape[-1] ** 0.5
    o = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v.float())
    return o.flatten(2, 3), torch.logsumexp(s, -1)[0]


class Rec:
    def __init__(self, fn=None):
        self.fn, self.calls = fn, []

    def __call__(self, q, k, v, ctx):
        self.calls.append((ctx.step, ctx.layer, ctx.branch, tuple(q.shape), tuple(k.shape)))
        return None if self.fn is None else self.fn(q, k, v)


class FakeWanAttn(torch.nn.Module):
    def __init__(self, H, D):
        super().__init__()
        self.heads = H
        self.to_q, self.to_k, self.to_v = (torch.nn.Linear(H * D, H * D) for _ in range(3))
        self.norm_q = torch.nn.RMSNorm(H * D)
        self.norm_k = torch.nn.RMSNorm(H * D)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(H * D, H * D), torch.nn.Dropout(0.0)])
        self.add_k_proj = None


def test_wan_processor_routes_through_hook():
    from decoupattn.models._wanapi_wan import WanSelfAttn, rope_torch
    torch.manual_seed(0)
    a = adapter("wan21", backend="sdpa")
    H, D, S = 2, 8, 12
    attn = FakeWanAttn(H, D)
    x = torch.randn(1, S, H * D)
    freqs = torch.polar(torch.ones(1, 1, S, D // 2, dtype=torch.float64), torch.rand(1, 1, S, D // 2).double())
    proc = WanSelfAttn(a, 7)
    q = rope_torch(attn.norm_q(attn.to_q(x)).unflatten(2, (H, -1)), freqs)
    k = rope_torch(attn.norm_k(attn.to_k(x)).unflatten(2, (H, -1)), freqs)
    v = attn.to_v(x).unflatten(2, (H, -1))
    dense = attn.to_out[0](F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
                           .transpose(1, 2).flatten(2, 3))
    with torch.no_grad():
        assert torch.allclose(proc(attn, x, rotary_emb=freqs), dense, atol=1e-5)
        rec = Rec()
        a.ctx.at(step=2)
        with a.hooked(rec):
            assert torch.allclose(proc(attn, x, rotary_emb=freqs), dense, atol=1e-5)
        assert rec.calls == [(2, 7, "cond", (1, S, H, D), (1, S, H, D))]
        with a.hooked(Rec(lambda q_, k_, v_: torch.zeros(1, S, H * D))):
            out = proc(attn, x, rotary_emb=freqs)
    assert torch.allclose(out, attn.to_out[0].bias.expand_as(out))


def test_cp25_compute_attention_routes_through_hook():
    import types

    from decoupattn.models.cosmos_predict2_5 import _hook_compute_attention
    torch.manual_seed(0)
    a = adapter("cp25", backend="sdpa")
    H, D, S = 2, 8, 10
    sa = types.SimpleNamespace(output_proj=torch.nn.Linear(H * D, 5), output_dropout=torch.nn.Identity())
    fn = types.MethodType(_hook_compute_attention(a, 3), sa)
    q, k, v = (torch.randn(1, S, H, D) for _ in range(3))
    with torch.no_grad():
        dense = sa.output_proj(_ref(q, k, v)[0])
        assert torch.allclose(fn(q, k, v), dense, atol=1e-5)
        rec = Rec(lambda q_, k_, v_: _ref(q_, k_, v_)[0])
        with a.hooked(rec):
            assert torch.allclose(fn(q, k, v), dense, atol=1e-5)
    assert rec.calls[0][1] == 3


def test_wan22_experts_by_step():
    from decoupattn.models.wan2_2 import ExpertPair, expert_index

    class Expert(torch.nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].attn1 = FakeWanAttn(1, 4)

        def forward(self, x):
            return self.tag

    a = adapter("wan22")
    a.tr = ExpertPair(a, Expert("high"), Expert("low"))
    assert [expert_index(s) for s in range(4)] == [0, 0, 1, 1]
    for s, tag in ((0, "high"), (1, "high"), (2, "low"), (3, "low")):
        a.ctx.at(step=s)
        assert a.tr(None) == tag and a.expert(s) == expert_index(s)
        assert a.out_proj(0, s) is a.tr.experts[expert_index(s)].blocks[0].attn1.to_out[0].weight


def test_hunyuan_joint_merges_text_exactly():
    from decoupattn.models import _hunyuan as HJ
    torch.manual_seed(0)
    a = adapter("hy")
    nv, nt, pad, H, D = 24, 5, 3, 2, 8
    S = nv + nt + pad
    qkv = torch.randn(1, S, 3, H, D)
    mask = torch.zeros(1, S, dtype=torch.bool)
    mask[0, :nv + nt] = True
    joint = HJ.Joint(a, nv)
    q, k, v = (t[:, :nv + nt] for t in qkv.unbind(2))
    dense = _ref(q, k, v)[0]
    out0 = joint(qkv, mask)
    assert torch.allclose(out0[:, :nv + nt].flatten(2, 3), dense, atol=1e-5) and not out0[:, nv + nt:].any()
    rec = Rec(lambda q_, k_, v_: _ref(q_, k_, v_))
    a.ctx.at(step=4, layer=31)
    with a.hooked(rec):
        out = joint(qkv, mask)
    assert rec.calls == [(4, 31, "cond", (1, nv, H, D), (1, nv, H, D))]
    assert torch.allclose(out[:, :nv + nt].flatten(2, 3), dense, atol=1e-5) and not out[:, nv + nt:].any()
    with a.hooked(Rec(lambda q_, k_, v_: _ref(q_, k_, v_)[0])), pytest.raises(TypeError):
        joint(qkv, mask)


def test_hunyuan_merge_chunks_are_exact(monkeypatch):
    from decoupattn.models import _hunyuan as HJ
    torch.manual_seed(1)
    nv, nt, H, D = 40, 6, 2, 8
    q, k, v = (torch.randn(1, nv + nt, H, D) for _ in range(3))
    o_v, l_v = _ref(q[:, :nv], k[:, :nv], v[:, :nv])
    one = HJ.merge_text(o_v, l_v, q, k, v, nv, "torch")
    monkeypatch.setattr(HJ, "MERGE_CHUNK", 7)
    assert torch.equal(HJ.merge_text(o_v, l_v, q, k, v, nv, "torch"), one)


class FakeC3Attn(torch.nn.Module):
    def __init__(self, Hq, Hk, D):
        super().__init__()
        self.num_attention_heads, self.num_key_value_heads, self.head_dim = Hq, Hk, D
        lin = lambda o: torch.nn.Linear(Hq * D, o)
        self.to_q, self.add_q_proj = lin(Hq * D), lin(Hq * D)
        self.to_k, self.to_v, self.add_k_proj, self.add_v_proj = (lin(Hk * D) for _ in range(4))
        self.norm_q = self.norm_k = self.norm_added_q = self.norm_added_k = torch.nn.RMSNorm(D)
        self.k_norm_und_for_gen = torch.nn.RMSNorm(D)
        self.to_out, self.to_add_out = torch.nn.Linear(Hq * D, Hq * D), torch.nn.Linear(Hq * D, Hq * D)


def test_nano_processor_merges_text_exactly():
    from decoupattn.models.cosmos3_nano import C3Attn
    torch.manual_seed(0)
    a = adapter("nano")
    Hq, Hk, D, T, S = 4, 2, 8, 5, 18
    attn = FakeC3Attn(Hq, Hk, D)
    und, gen = torch.randn(T, Hq * D), torch.randn(S, Hq * D)
    rot = [torch.randn(n, D) for n in (T, T, S, S)]
    proc = C3Attn(a, 9)
    with torch.no_grad():
        u0, g0 = proc(attn, und, gen, rot)
        rec = Rec(lambda q_, k_, v_: _ref(q_, k_, v_))
        a.ctx.at(step=3, branch="uncond")
        with a.hooked(rec):
            u1, g1 = proc(attn, und, gen, rot)
    assert rec.calls == [(3, 9, "uncond", (1, S, Hq, D), (1, S, Hq, D))]
    assert torch.equal(u0, u1) and torch.allclose(g0, g1, atol=1e-5)


def test_attend_defaults_to_dense():
    a = adapter("wan21", backend="torch")
    q, k, v = (torch.randn(1, 6, 2, 4) for _ in range(3))
    assert torch.allclose(a.attend(q, k, v), _ref(q, k, v)[0], atol=1e-5)
    with a.hooked(Rec()):
        assert torch.allclose(a.attend(q, k, v), _ref(q, k, v)[0], atol=1e-5)
    assert a.hook is None


def test_prompt_index_and_build(tmp_path):
    with open(repo_path("data/prompts/moviegenbench.json")) as f:
        idx = json.load(f)
    ids = {c: n for split in idx["splits"].values() for c, n in split.items()}
    assert len(ids) == len(set(ids.values())) == 56
    for key in VIDEO:
        a = adapter(key)
        assert set(a.clips("calib") + a.clips("val") + a.clips("test")) <= set(ids)
        assert load_config(key)["data"]["clips"].endswith("prompts/moviegenbench.json")
    src = tmp_path / "mgb.txt"
    src.write_text("".join(f"prompt {i}\n" for i in range(1003)), encoding="utf-8")
    out = tmp_path / "prompts.json"
    r = subprocess.run([sys.executable, repo_path("scripts/build_prompts.py"), "--src", str(src), "--out", str(out)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    prompts = json.loads(out.read_text(encoding="utf-8"))
    assert all(prompts[c] == f"prompt {n}" for c, n in ids.items())
    a = adapter("hy")
    a.cfg["data"]["clips"] = str(out)
    assert a.prompt("test_00") == f"prompt {ids['test_00']}"
