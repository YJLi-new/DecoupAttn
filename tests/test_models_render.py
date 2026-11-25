from __future__ import annotations

import sys

import pytest
import torch

from decoupattn.config import load_config
from decoupattn.context import CallContext
from decoupattn.models import adapter_class
from decoupattn.models._newrenderers import nearest_frames, to_uint8

KEYS = ("t25", "dr", "alaya", "unirelight", "cp1")
GPU_PACKAGES = ("flash_attn_interface", "flash_attn", "flashinfer", "cosmos_predict1", "cosmos_transfer2", "diffsynth",
                "transformer_engine")
TEST_CLIPS = {"t25": 39, "dr": 39, "alaya": 39, "unirelight": 39, "cp1": 39}


def adapter(key):
    return adapter_class(key)(load_config(key, resolve=False))


@pytest.mark.parametrize("key", KEYS)
def test_import_without_gpu_packages(key):
    adapter_class(key)
    assert not [m for m in GPU_PACKAGES if m in sys.modules]


@pytest.mark.parametrize("key", KEYS)
def test_config_agrees_with_adapter(key):
    cfg, cls = load_config(key, resolve=False), adapter_class(key)
    assert cfg["model"]["key"] == cls.name == key
    assert cfg["model"]["family"] == cls.family
    s, at, pl = cfg["sampler"], cfg["attention"], cfg["plan"]
    assert s["steps"] == cls.steps
    assert tuple(s["branches"]) == cls.branches
    assert (at["n_layers"], at["n_heads"], at["head_dim"]) == (cls.n_layers, cls.n_heads, cls.head_dim)
    assert tuple(at["latent_grid"]) == cls.latent_grid
    assert pl["layout"] == cls.layout and tuple(pl["rows"]) == cls.rows
    assert (tuple(pl["block_shape"]) if "block_shape" in pl else None) == cls.block_shape
    assert all(0 < r < cls.steps for r in cls.rows)
    if "frames" in vars(cls):
        assert s["frames"] == cls.frames


@pytest.mark.parametrize("key", KEYS)
def test_splits(key):
    a = adapter(key)
    calib, val, test = a.clips("calib"), a.clips("val"), a.clips("test")
    assert (len(calib), len(val), len(test)) == (16, 8, TEST_CLIPS[key])
    assert not (set(calib) & set(val) or set(calib) & set(test) or set(val) & set(test))


def test_planner_and_runtime_tables(tmp_path):
    import inspect

    from decoupattn.planner.pipeline import sens_source
    from decoupattn.runtime import DecoupRuntime
    opts = set(inspect.signature(DecoupRuntime).parameters)
    for key in KEYS:
        cfg, a = load_config(key, resolve=False), adapter(key)
        assert set(cfg["runtime"]) <= opts, key
        src = cfg["planner"]["sensitivity"]
        groups = src["step_groups"]
        steps = [s for s0, s1 in groups for s in range(s0, s1 + 1)]
        assert steps == list(range(1, cls_steps(key)))
        from decoupattn.planner import sensitivity
        from decoupattn.planner.units import Units
        A = adapter_class(key)
        sensitivity.check_step_groups(Units(A.layout, A.steps, A.rows, A.n_layers, 1), groups)
        assert sens_source(a, str(tmp_path))["path"] == f"{tmp_path}/sensitivity/J_s1000.jsonl", key


def cls_steps(key):
    return adapter_class(key).steps


def test_route_hook_and_dense_fallback():
    a = adapter("t25")
    a.cfg["model"]["dense_backend"] = "sdpa"
    q, k, v = (torch.randn(1, 64, 2, 8) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    a.hook = lambda *_: None
    assert torch.allclose(a.route(q, k, v), ref.transpose(1, 2).flatten(2, 3))
    a.hook = lambda q_, k_, v_, ctx: torch.zeros(1, 64, 16)
    assert a.route(q, k, v).abs().max() == 0


def test_unirelight_final_call_uses_last_step_plan():
    a = adapter("unirelight")
    seen = []
    a.hook = lambda q, k, v, ctx: seen.append(ctx.step) or torch.zeros(1, q.shape[1], q.shape[2] * q.shape[3])
    a.ctx = CallContext()
    q = torch.zeros(1, 4, 2, 8)
    for step in (0, 34, 35):
        a.ctx.step = step
        a.route(q, q, q)
        assert a.ctx.step == step
    assert seen == [0, 34, 34]


def test_nearest_frames_and_uint8():
    m = nearest_frames(57, 93)
    assert len(m) == 93 and m[0] == 0 and m[-1] == 56 and m == sorted(m) and len(set(m)) == 57
    assert nearest_frames(57, 81)[40] == 28
    v = torch.tensor([-1.0, 0.0, 1.0, 2.0]).view(1, 4, 1, 1).expand(3, 4, 1, 1)
    assert to_uint8(v)[:, 0, 0, 0].tolist() == [0, 128, 255, 255]


def test_dr_env_params_and_fast_path(tmp_path):
    import json

    from decoupattn.models import _dr_fast
    from decoupattn.models.diffusion_renderer import env_params
    spec = {"spec": {"hdri": {"path": "/server/hdri/park_2k.hdr", "rot_deg": 35.0, "strength": 1.5}}}
    (tmp_path / "render.json").write_text(json.dumps(spec))
    ep = env_params(str(tmp_path), "/data/hdri")
    assert ep == {"hdri": "/data/hdri/park_2k.hdr", "env_flip": False, "env_rot": 125.0, "env_strength": 1.5,
                  "pose_file": str(tmp_path / "transforms.json")}
    assert load_config("dr", resolve=False)["model"]["forward"] == "fast"
    assert not [m for m in GPU_PACKAGES if m in sys.modules]


def test_dr_native_dense_heads():
    a = adapter("dr")
    a.cfg["model"]["dense_backend"] = "native"
    q, k, v = (torch.randn(1, 32, 4, 8) for _ in range(3))
    sd = torch.nn.functional.scaled_dot_product_attention
    a.native = lambda q_, k_, v_: sd(*(t.transpose(1, 2) for t in (q_, k_, v_))).transpose(1, 2).reshape(1, 32, -1)
    full = a.dense(q, k, v).view(1, 32, 4, 8)
    import numpy as np
    out = a.dense_heads(q, k, v, np.array([0, 2]))
    assert out.shape == (1, 32, 16) and torch.equal(out, full[:, :, [0, 2]].reshape(1, 32, 16))


def test_stop_after_steps():
    from decoupattn.models._newrenderers import Stop
    a = adapter("alaya")
    a.loaded = True
    a.begin(None, steps=2)
    a.ctx.step = 1
    with a.dit_call("s1_cond"):
        pass
    a.ctx.step = 2
    with pytest.raises(Stop):
        a.dit_call("s2_cond")
    a.begin(None)
    assert a.stop_at is None
