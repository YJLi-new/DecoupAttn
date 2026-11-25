from __future__ import annotations

import json
import math
import os
import sys
import types

import pytest
import torch

from decoupattn.baselines import _calibatt_agg as A
from decoupattn.baselines import _calibatt_port as P
from decoupattn.baselines import _calibatt_wan as W
from decoupattn.baselines import _svoo_port as SP
from decoupattn.baselines import get_baseline
from decoupattn.context import CallContext

MODELS = ["cp1", "cp25", "wan21", "wan22", "hy", "nano", "t25", "dr", "alaya", "unirelight"]
TILES = (4, 8)


class Adapter:
    name, latent_grid, n_layers, n_heads, head_dim, steps = "fake", (2, 10, 4), 3, 2, 8, 3
    branches, needs_lse = ("cond",), False

    def __init__(self):
        self.generated, self.outputs = [], []

    def clips(self, split):
        return ["train_00", "train_01", "train_02"]

    def generate(self, clip, hook=None, seed=1000, steps=None):
        self.generated.append(clip)
        for step in range(self.steps):
            for layer in range(self.n_layers):
                q = torch.randn(1, 80, self.n_heads, self.head_dim)
                self.outputs.append(((step, layer), hook(q, q, q, CallContext(step=step, layer=layer))))


def module(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, m)
    parent, _, leaf = name.rpartition(".")
    if parent in sys.modules:
        monkeypatch.setattr(sys.modules[parent], leaf, m, raising=False)
    return m


class Lite:
    def __init__(self, enable_skipping=True, max_batch_size=1):
        self._ktiles, self._skip_list = None, [None]

    @staticmethod
    def get_MN(head_dim, dtype):
        return TILES

    def initialize_ktiles(self, seq_len, head_dim, dtype):
        self._ktiles = -(-seq_len // TILES[1])

    def calc_percentage_per_head(self, t):
        return t.float()


class Backend:
    def __init__(self, **kw):
        self.kw, self.calls = kw, []
        self.fa_block_size_m, self.fa_block_size_n = kw["fa_block_size"]
        self.anchor_rows, self.repeat_anchor_strategy = kw["anchor_rows"], kw["repeat_anchor_strategy"]
        self.lite_attention_modules = kw["processor_mod"].install(self, kw["pipeline"])
        self.skip_lists_dict, self.repetitive_heads_per_layer = {}, {}
        self.calibrate_masks = bool(kw.get("calibrate_masks"))
        self.current_timestep = self.current_layer = 0
        self.device = kw["device"]
        if kw.get("skip_lists_path"):
            self.load_skip_lists(kw["skip_lists_path"])

    def load_skip_lists(self, path):
        d = torch.load(path, weights_only=False)
        self.skip_lists_dict = dict(d["skip_lists_dict"])
        self.repetitive_heads_per_layer = {k: (v, None) for k, v in d.get("repetitive", {}).items()}

    def _calc_block_energy_from_qk(self, q, k, softmax_scale=None, causal=False):
        nq, nk = (-(-q.shape[1] // b) for b in (self.fa_block_size_m, self.fa_block_size_n))
        return torch.ones(q.shape[0], q.shape[2], nq, nk)

    def compute_attention(self, q, k, v):
        self.calls.append((self.current_timestep, self.current_layer))
        if self.calibrate_masks:
            self._calc_block_energy_from_qk(q, k)
        return torch.zeros_like(q)

    def reset_logs(self):
        pass

    def update_output_dir(self, d):
        self.output_dir = d

    def save_deferred_masks(self):
        torch.save({}, os.path.join(self.output_dir, "spatial_similarities.pt"))


@pytest.fixture
def calibatt(monkeypatch):
    module(monkeypatch, "lite_attention", LiteAttention=Lite)
    module(monkeypatch, "CalibAtt")
    module(monkeypatch, "CalibAtt.calibatt", CalibAttBackend=Backend, flash_attn_func_fa3=None)
    monkeypatch.setattr(P, "import_calibatt", lambda: None)


def settings(model="cp1", **over):
    return {**get_baseline("calibatt", model).settings, **over}


def skip_file(path, shares, repetitive=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {k: torch.tensor(v)[None, :, None].expand(1, len(v), 20).clone() for k, v in shares.items()}
    torch.save({"skip_lists_dict": d, "repetitive": repetitive or {}}, path)


@pytest.mark.parametrize("model", MODELS)
def test_calibatt_settings(model):
    b = get_baseline("calibatt", model)
    s = b.settings
    start, floor, decay = P.epsilon_of(s)
    assert 0 < floor < start < 1 and decay > 0
    assert 0 < s["target_density"] < 1 and s["anchor_rows"] >= 1 and isinstance(s["dense_first"], bool)
    assert b.standalone == (model == "wan21")
    if b.standalone:
        assert b.reference != "dense" and len(set(s["calib_lines"])) == len(s["calib_lines"]) > 0
    else:
        assert s["calib_clips"] >= 0 and len(A.c_grid(*s["c_grid"])) > 1


@pytest.mark.parametrize("model", MODELS)
def test_svoo_settings(model):
    b = get_baseline("svoo", model)
    s = b.settings
    assert 0 < s["top_p"] <= 1 and 0 < s["kr_min"] <= s["kr_max"] < 1
    assert s["official"] in SP.OFFICIAL_TOP_P and not b.needs_match()
    assert get_baseline("svoo", model, rematch=True).needs_match()
    cfg = SP.processor_settings(None, (2, 4, 4), 5, s, s["dense_steps"])
    assert cfg["start_reuse_step"] == s["dense_steps"] + 1 and cfg["use_svoo"] and not cfg["use_ear"]
    assert (cfg["dynamic_min_kc_ratio_min"], cfg["dynamic_min_kc_ratio_max"]) == (s["kr_min"], s["kr_max"])


def test_anchor_rows_and_padding():
    for rows, anchors in ((30, 5), (44, 5), (7, 3)):
        cnt = P.anchor_row_counts(rows, anchors)
        assert sum(cnt) == rows and len(cnt) == anchors and max(cnt) - min(cnt) <= 1 and cnt == sorted(cnt, reverse=True)
    e = torch.arange(24.0).view(2, 3, 4)
    p = P.pad_energy(e, 5, 6)
    assert p.shape == (2, 5, 6) and torch.equal(p[:, :3, :4], e) and float(p[..., 4:].sum()) == 0
    assert torch.equal(p[:, 3:], p[:, :2])


@pytest.mark.parametrize("dense_first", [True, False])
def test_calibatt_infer_dense_guard_and_density(calibatt, tmp_path, dense_first):
    ad = Adapter()
    first = 1 if dense_first else 0
    shares = {(t, layer): [0.25, 0.75] for t in range(first, 3) for layer in range(first, 3)}
    rep = {(2, 2): [1]}
    skip_file(str(tmp_path / "agg" / "skip_lists.pt"), shares, rep)
    hook = P.infer_hook(ad, settings(dense_first=dense_first), str(tmp_path / "agg"), device="cpu")
    hook.reset()
    ad.generate("c", hook)
    dense = [key for key, out in ad.outputs if out is None]
    assert sorted(dense) == sorted(k for k in [(t, layer) for t in range(3) for layer in range(3)] if k not in shares)
    assert all(out.shape == (1, 80, 16) for key, out in ad.outputs if out is not None)
    assert sorted(hook.be.calls) == sorted(shares)
    st = hook.summary()
    want = ((len(shares) - 1) * 0.5 + (0.25 + 5 / 10) / 2) / len(shares)
    assert st["sparse_calls"] == len(shares) and st["dense_calls"] == 9 - len(shares)
    assert st["exec_density"] == pytest.approx(want)


def test_calibatt_capture_resumes(calibatt, tmp_path):
    ad, cap = Adapter(), str(tmp_path / "capture")
    s = settings(dense_first=True)
    P.capture(ad, s, cap, ["train_00", "train_01"], 1000, device="cpu", log=lambda *_: None)
    assert ad.generated == ["train_00", "train_01"] and all(out is not None or key[0] == 0 or key[1] == 0
                                                            for key, out in ad.outputs)
    import numpy as np
    E = np.load(os.path.join(cap, "train_00_s1000", "block_energy.npy"))
    with open(os.path.join(cap, "train_00_s1000", "block_energy_index.json")) as f:
        ix = json.load(f)
    assert E.shape == (2, 2, 2, 20, 10) and ix["t"] == [1, 2] and ix["l"] == [1, 2] and tuple(ix["shape"]) == E.shape
    assert os.path.exists(os.path.join(cap, "train_00_s1000", "spatial_similarities.pt"))
    assert not os.path.exists(os.path.join(cap, "train_01_s1000", "spatial_similarities.pt"))
    P.capture(ad, s, cap, ["train_00", "train_01", "train_02"], 1000, device="cpu", log=lambda *_: None)
    assert ad.generated == ["train_00", "train_01", "train_02"]


def test_calibatt_joint_model(calibatt, tmp_path):
    ad = Adapter()
    skip_file(str(tmp_path / "agg" / "skip_lists.pt"), {(1, 1): [0.5, 0.5]})
    hook = P.infer_hook(ad, settings("hy"), str(tmp_path / "agg"), device="cpu")
    assert hook.be.kw["max_text_tokens"] > 0 and hook.be.kw["text_tokens_position"] == "end"
    q = torch.randn(1, 90, 2, 8)
    assert hook.joint(q, q, q, 80, CallContext(step=0, layer=1)) is None
    assert hook.joint(q, q, q, 80, CallContext(step=1, layer=1)).shape == (1, 90, 16)
    assert hook.summary()["exec_density"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        hook(q[:, :80], q[:, :80], q[:, :80], CallContext(step=1, layer=1))


def test_calibatt_prepare_idempotent(calibatt, tmp_path, monkeypatch):
    b, ad, run, calls = get_baseline("calibatt", "cp25", calib_clips=2), Adapter(), str(tmp_path), []

    def aggregate(cap, out, s, log=print):
        calls.append(sorted(os.listdir(cap)))
        skip_file(os.path.join(out, "skip_lists.pt"), {(t, layer): [0.2, 0.2] for t in range(3) for layer in range(3)})
    monkeypatch.setattr(A, "aggregate", aggregate)
    with pytest.raises(FileNotFoundError):
        b.hook(ad, run)
    b.prepare(ad, run, device="cpu", log=lambda *_: None)
    assert ad.generated == ["train_00", "train_01"] and calls == [["train_00_s1000", "train_01_s1000"]]
    b.prepare(ad, run, device="cpu", log=lambda *_: None)
    assert ad.generated == ["train_00", "train_01"] and len(calls) == 1
    with pytest.raises(NotImplementedError):
        get_baseline("calibatt", "wan21").hook(ad, run)


def test_scaled_epsilon():
    g3 = (0.9, 0.8, 1.0)
    assert A.scaled_epsilon(0, 1.0, g3) == pytest.approx(0.9) and A.scaled_epsilon(50, 1.0, g3) == pytest.approx(0.8)
    assert 1 - A.scaled_epsilon(1, 2.0, g3) == pytest.approx(2 * (1 - A.scaled_epsilon(1, 1.0, g3)))


def test_grid_density():
    E = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).view(2, 1, 1, 4)
    same = E[:1].expand(2, 1, 1, 4)
    assert A.grid_density(same, [0.3, 0.75, 1.0], 0.5) == pytest.approx([0.25, 0.75, 1.0])
    loose, strict = A.grid_density(E, [0.3, 0.75], 0.5), A.grid_density(E, [0.3, 0.75], 0.9)
    assert loose == sorted(loose) and all(a >= b for a, b in zip(loose, strict)) and strict[0] == 0
    two = torch.rand(3, 2, 5, 8)
    d = A.grid_density(two, [0.5], 0.5, rep_heads={1}, rep_density=0.125)[0]
    assert d == pytest.approx((A.grid_density(two[:, :1], [0.5], 0.5)[0] + 0.125) / 2)


def test_skip_list_density_and_invert():
    sl = torch.tensor([[2, 0, 3, 0, 0], [4, 0, 1, 2, 4]]).view(1, 1, 2, 5)
    assert A.skip_list_density(sl, (), 6, 0.0) == pytest.approx(0.5)
    two = torch.cat([sl, torch.zeros_like(sl)], 1)
    assert A.skip_list_density(two, (), 6, 0.0) == pytest.approx(0.25)
    assert A.skip_list_density(two, {1}, 6, 0.3) == pytest.approx(0.4)
    assert A.invert(0.3, [1.0, 2.0, 3.0], [0.4, 0.2, 0.1]) == pytest.approx(1.5)


def test_aggregate_is_idempotent(tmp_path, monkeypatch):
    out = tmp_path / "agg"
    out.mkdir()
    (out / "skip_lists.pt").write_bytes(b"")
    (out / "agg.json").write_text(json.dumps({"chosen": 1.2, "chosen_density": 0.2}))
    monkeypatch.setattr(A, "import_calibatt", lambda: pytest.fail("a finished aggregation must not run again"))
    assert A.aggregate(str(tmp_path / "capture"), str(out), settings())["chosen"] == 1.2


def test_aggregate_matches_density(tmp_path, monkeypatch):
    import numpy as np
    torch.manual_seed(0)
    cap, out, target = tmp_path / "capture", tmp_path / "agg", 0.3
    for i in range(3):
        d = cap / f"train_0{i}_s1000"
        d.mkdir(parents=True)
        np.save(d / "block_energy.npy", torch.rand(2, 2, 2, 6, 16).pow(6).numpy().astype(np.float16))
        (d / "block_energy_index.json").write_text(json.dumps(
            {"t": [1, 2], "l": [1, 2], "shape": [2, 2, 2, 6, 16], "fa_block": [4, 8], "grid": [1, 10, 4], "S": 40,
             "eps": [0.9, 0.8, 1.0]}))

    class Mixin:
        @staticmethod
        def _calibrate_masks_from_block_energy(obj, be, head_offset):
            se, idx = be.sort(-1, descending=True)
            kept = ((se.cumsum(-1) < be.sum(-1, keepdim=True) * obj._get_scheduled_epsilon()).sum(-1) + 1).clamp(max=16)
            rank = torch.empty_like(idx).scatter_(-1, idx, torch.arange(16).expand_as(idx))
            return rank < kept[..., None], None

    def aggregate_sparse_masks(input_path, output_dir, agreement_threshold, fa_block_size, skip_list_merge_percentile,
                               gpu_device=None, verbose=False):
        files = [torch.load(os.path.join(input_path, d, "sparse_pattern_masks.pt")) for d in sorted(os.listdir(input_path))]
        lists = {}
        for key in files[0]:
            vote = torch.stack([f[key].float() for f in files]).mean(0) >= agreement_threshold
            sl = torch.zeros(1, 1, vote.shape[2], 1 + 2 * 16, dtype=torch.int32)
            for r, row in enumerate(vote[0, 0]):
                cols = torch.nonzero(row).flatten().tolist()
                sl[0, 0, r, 0] = 2 * len(cols)
                for j, c in enumerate(cols):
                    sl[0, 0, r, 1 + 2 * j], sl[0, 0, r, 2 + 2 * j] = c, c + 1
            lists.setdefault(key[:2], []).append(sl)
        os.makedirs(output_dir, exist_ok=True)
        torch.save({"skip_lists_dict": {k: torch.cat(v, 1) for k, v in lists.items()}},
                   os.path.join(output_dir, "skip_lists.pt"))

    def aggregate_spatial_similarities(input_path, output_path):
        torch.save({"similarity_map": {}}, output_path)
        return {}

    module(monkeypatch, "CalibAtt")
    module(monkeypatch, "CalibAtt.calibration", aggregate_sparse_masks=types.SimpleNamespace(
        aggregate_sparse_masks=aggregate_sparse_masks))
    module(monkeypatch, "CalibAtt.calibration.block_masks", CalibrationMixin=Mixin)
    module(monkeypatch, "CalibAtt.calibration.aggregate_spatial_similarities",
           aggregate_spatial_similarities=aggregate_spatial_similarities)
    monkeypatch.setattr(A, "import_calibatt", lambda: None)
    s = settings(target_density=target, workers=1, c_grid=[0.2, 6.0, 0.2])
    res = A.aggregate(str(cap), str(out), s, log=lambda *_: None)
    assert abs(res["chosen_density"] - target) < 0.05
    assert res["chosen_density"] == min((r["density_final"] for r in res["runs"]), key=lambda d: abs(d - target))
    sd = torch.load(out / "skip_lists.pt")["skip_lists_dict"]
    assert sorted(sd) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert not any(p.startswith("masks_") for p in os.listdir(out))
    again = A.aggregate(str(cap), str(out), s, log=lambda *_: pytest.fail("ran again"))
    assert again["chosen"] == res["chosen"]


def test_realised_density():
    full = torch.tensor([2, 0, 2]).expand(1, 1, 4, 3)
    half = torch.tensor([2, 0, 1]).expand(1, 1, 4, 3)
    skip = {"fa_block_size": [4, 8], "skip_lists_dict": {(0, 0): torch.cat([full, half], 1)}}
    d = W.realised_density(skip, {}, (1, 4, 4), 0.87, 1)
    assert d["exec_density"] == pytest.approx(0.75) and d["calibatt_density"] == pytest.approx(0.75)
    d = W.realised_density(skip, {(0, 0, 0): 0.9, (0, 0, 1): 0.1}, (1, 4, 4), 0.87, 1)
    assert d["exec_density"] == pytest.approx((0.25 + 0.5) / 2) and d["repetitive_share"] == pytest.approx(0.5)


def test_calibration_config_and_prompts(tmp_path):
    base = ('data_path: data/MovieGenVideoBench.txt\ndata_num_samples: 16\ncalibatt:\n  epsilon_start: 0.863\n'
            '  epsilon_floor: 0.763\n  epsilon_decay: 1.410\n  aggregation:\n    output_dir: "calibration/wan_distilled_480p"\n')
    cfg = W.calib_config(base, "/p/prompts.txt", 64, 0.135, "/p/out", "/p/samples")
    assert "0.7280" in cfg and "0.6280" in cfg and "/p/prompts.txt" in cfg and "/p/out" in cfg and "1.410" in cfg
    assert "data_num_samples: 64" in cfg and "MovieGenVideoBench" not in cfg
    with pytest.raises(ValueError):
        W.calib_config(base.replace("epsilon_floor: 0.763", "epsilon_floor: 0.5"), "a", 64, 0.1, "b", "c")
    W.calib_prompts(["zero", "one", "two", "three"], [3, 1], str(tmp_path / "p" / "prompts.txt"))
    assert (tmp_path / "p" / "prompts.txt").read_text().splitlines() == ["three", "one"]


def test_wan_prepare_idempotent(tmp_path, monkeypatch):
    b, ad, runs = get_baseline("calibatt", "wan21"), Adapter(), []

    def calibrate(adapter, s, work, shift, picks, name, log=print):
        runs.append((shift, len(picks)))
        os.makedirs(os.path.join(work, name))
        torch.save({"fa_block_size": [4, 8], "skip_lists_dict": {(0, 0): torch.tensor([2, 0, 1]).expand(1, 2, 20, 3)}},
                   os.path.join(work, name, "skip_lists.pt"))
        torch.save({"similarity_map": {}}, os.path.join(work, name, "similarity_map.pt"))
        return os.path.join(work, name)
    monkeypatch.setattr(W, "calibrate", calibrate)
    b.prepare(ad, str(tmp_path), log=lambda *_: None)
    assert runs == [(b.settings["epsilon_shift"], len(b.settings["calib_lines"]))]
    rec = json.loads((tmp_path / "baselines" / "calibatt" / "calibration.json").read_text())
    assert 0 < rec["exec_density"] <= 1 and W.stats(b.work_dir(str(tmp_path)))["exec_density"] == rec["exec_density"]
    b.prepare(ad, str(tmp_path), log=lambda *_: None)
    assert len(runs) == 1 and ad.generated == []


def test_svoo_match_picks_the_knob():
    cases = [(lambda p, lo, hi: 0.2 * p + 0.5 * hi, "top_p"),
             (lambda p, lo, hi: 0.3 * p + hi, "window_scale"),
             (lambda p, lo, hi: 0.1 * p + 0.25 * hi, "window_scale")]
    for fn, knob in cases:
        r = SP.match(fn, 0.20, 0.004, 8, 0.90, (0.15, 0.20))
        assert r["knob"] == knob and abs(r["exec_density"] - 0.20) <= 0.004
        assert fn(r["top_p"], r["kr_min"], r["kr_max"]) == pytest.approx(r["exec_density"], abs=1e-3)
        assert r["kr_min"] / r["kr_max"] == pytest.approx(0.75, abs=2e-3)
    r = SP.match(lambda p, lo, hi: 0.2, 0.20, 0.004, 8, 0.90, (0.15, 0.20))
    assert (r["knob"], r["top_p"], r["kr_max"], len(r["history"])) == ("none", 0.90, 0.20, 1)


def test_svoo_knobs_prefer_the_rematch(tmp_path):
    b = get_baseline("svoo", "cp25")
    assert b.knobs(str(tmp_path))["top_p"] == b.settings["top_p"]
    SP.write_json(b.match_file(str(tmp_path)), {"top_p": 0.41, "kr_min": 0.1, "kr_max": 0.2})
    k = b.knobs(str(tmp_path))
    assert (k["top_p"], k["kr_min"], k["kr_max"]) == (0.41, 0.1, 0.2) and k["qc"] == b.settings["qc"]
    with pytest.raises(FileNotFoundError):
        b.hook(Adapter(), str(tmp_path))


def test_svoo_profile_is_idempotent(tmp_path, monkeypatch):
    def exact(q, k, batch_size=0, threshold=0.95, query_sample_size=0):
        return 0.5, [0.4, 0.6], {"seq_len": q.shape[2]}

    def save_to_csv(data, path):
        with open(path, "w") as f:
            f.write("\n".join(data))
    module(monkeypatch, "svoo")
    module(monkeypatch, "svoo.sparsity", compute_exact_attention_sparsity=exact)
    module(monkeypatch, "svoo.sparsity.merge", merge_sparsity_data=lambda files: [os.path.basename(f) for f in files],
           save_to_csv=save_to_csv)
    monkeypatch.setattr(SP, "import_svoo", lambda: None)
    b, ad = get_baseline("svoo", "cp1"), Adapter()
    csv = SP.profile(ad, b.settings, str(tmp_path), 1000, log=lambda *_: None)
    assert ad.generated == ad.clips("calib")[: b.settings["profile_clips"]] and os.path.exists(csv)
    assert all(out is None for _, out in ad.outputs)
    raw = sorted(os.listdir(tmp_path / "profile_raw"))
    assert len(raw) == b.settings["profile_clips"]
    text = (tmp_path / "profile_raw" / raw[0]).read_text()
    assert text.count("[Sparsity]") == 4 and text.count("Head") == 8
    assert SP.profile(ad, b.settings, str(tmp_path), 1000) == csv and len(ad.generated) == b.settings["profile_clips"]


def test_svoo_hook_dense_guard_and_density(monkeypatch):
    class Processor:
        def __init__(self, layer_idx):
            self.layer_idx, self._cached_clustering_results = layer_idx, None

        def semantic_aware_permutation(self, q, k, v, _):
            B, H, S, _ = q.shape
            if self._cached_clustering_results is None:
                self._cached_clustering_results = object()
            block_map = torch.zeros(B, H, 2, 4, dtype=torch.bool)
            block_map[..., :2] = True
            qsz = torch.full((B, H, 2), S // 2)
            ksz = torch.full((B, H, 4), S // 4)
            return q, k, v, block_map, qsz, ksz, torch.arange(S).expand(B, H, S)

    module(monkeypatch, "svoo")
    module(monkeypatch, "svoo.co_clustering",
           dynamic_block_sparse_fwd_flashinfer=lambda q, k, v, bm, qsz, ksz, is_cpu=False: torch.ones_like(q))
    module(monkeypatch, "svoo.kernels")
    module(monkeypatch, "svoo.kernels.triton")
    module(monkeypatch, "svoo.kernels.triton.permute", apply_inverse_permutation_triton=lambda o, idx, dim=2: o)
    module(monkeypatch, "svoo.sparsity", SparsityLookup=lambda csv: {"csv": csv})
    monkeypatch.setattr(SP, "import_svoo", lambda: types.SimpleNamespace(WanAttn_SAPAttn_Processor=Processor))
    s = get_baseline("svoo", "cp1").settings
    hook = SP.SVOOHook("profile.csv", (2, 10, 4), 3, 2, 8, s, warmup=False)
    ad = Adapter()
    for _ in range(2):
        hook.reset()
        ad.outputs = []
        ad.generate("c", hook)
        st = hook.summary()
        assert [key for key, out in ad.outputs if out is None] == [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]
        assert all(out.shape == (1, 80, 16) for _, out in ad.outputs if out is not None)
        assert (st["sparse_calls"], st["dense_calls"], st["label_builds"]) == (4, 5, 2)
        assert st["real_density"] == pytest.approx(0.5) and st["exec_density"] >= st["real_density"]
    p = hook._proc(1)
    assert p.current_inference_step == 3 and p.start_reuse_step == 2 and math.isclose(p.top_p_kmeans, s["top_p"])
    q = torch.randn(1, 90, 2, 8)
    assert hook.joint(q, q, q, 80, CallContext(step=0, layer=1)) is None
    assert hook.joint(q, q, q, 80, CallContext(step=1, layer=1)).shape == (1, 90, 16)
