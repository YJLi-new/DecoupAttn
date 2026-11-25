from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from decoupattn.config import REPO_ROOT, load_config
from decoupattn.models import MODELS

spec_ = importlib.util.spec_from_file_location("fetch_models", os.path.join(REPO_ROOT, "scripts", "fetch_models.py"))
fm = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(fm)
SPEC = fm.load_spec()
ROOT = "/data-root"


@pytest.fixture(autouse=True)
def data_root(monkeypatch):
    monkeypatch.setenv("DECOUPATTN_DATA_ROOT", ROOT)


def under(path: str, base: str) -> bool:
    return os.path.normpath(path) == os.path.normpath(base) or os.path.normpath(path).startswith(base.rstrip("/") + "/")


def test_every_model_has_an_entry():
    for key in MODELS:
        assert key in SPEC, key
        assert SPEC[key].get("weights"), key


@pytest.mark.parametrize("key", list(MODELS))
def test_repos_agree_with_the_model_config(key):
    model = load_config(key)["model"]
    repos = SPEC[key].get("repos", [])
    if "repo" not in model:
        assert not repos
        return
    match = [r for r in repos if under(model["repo"], os.path.join(ROOT, r["dest"]))]
    assert match, model["repo"]
    assert match[0]["url"] == model["repo_url"]
    assert match[0]["commit"]
    if "commit" in model:
        assert match[0]["commit"] == model["commit"]


@pytest.mark.parametrize("key", list(MODELS))
def test_checkpoint_paths_are_fetched_or_derived(key):
    m = SPEC[key]
    dests = [os.path.join(ROOT, w["dest"]) for w in m["weights"] if w["dest"]]
    dests += [os.path.join(ROOT, d) for d in m.get("derived", [])]
    for name, path in load_config(key)["checkpoints"].items():
        if not path:
            continue
        path = os.path.dirname(path) if "{" in path else path
        assert any(under(path, d) or under(d, path) for d in dests), (key, name, path)


def test_weight_entries_are_well_formed():
    for key, m in SPEC.items():
        for w in m["weights"]:
            assert "/" in w["repo"] and w.get("gib", 0) > 0, (key, w["repo"])
            assert not os.path.isabs(w["dest"]) and ".." not in w["dest"].split("/"), (key, w["dest"])
        for r in m.get("repos", []):
            assert r["url"].startswith("https://github.com/") and not os.path.isabs(r["dest"]), r
            for p in r.get("patches", []):
                assert p["old"] and p["new"] and p["old"] != p["new"], p


def test_dry_run_prints_commands_and_fetches_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DECOUPATTN_DATA_ROOT", str(tmp_path))

    def forbidden(*a, **k):
        raise AssertionError("a dry run must not run anything")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert fm.main(["--model", "all", "--optional", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "git clone https://github.com/" in out and "git -C" in out
    assert "hf download " in out and "--local-dir" in out and "--include" in out
    assert "convert_hunyuan_transformer.py" in out
    assert list(tmp_path.iterdir()) == []


def test_dry_run_without_data_root_keeps_the_variable(monkeypatch, capsys):
    monkeypatch.delenv("DECOUPATTN_DATA_ROOT")
    assert fm.main(["--model", "cp25", "--weights", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"$DECOUPATTN_DATA_ROOT/checkpoints/' in out and "git clone" not in out


def test_list_shows_every_model(capsys):
    assert fm.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(key in out for key in MODELS)


def test_present_repo_is_patched_once_and_not_cloned(tmp_path, monkeypatch):
    r = {"url": "https://github.com/org/x", "commit": "abc1234", "dest": "repos/x",
         "patches": [{"file": "m.py", "old": "import a\n", "new": "try:\n    import a\nexcept ImportError:\n    a = None\n"}]}
    dest = tmp_path / "repos" / "x"
    (dest / ".git").mkdir(parents=True)
    (dest / "m.py").write_text("import a\nimport b\n")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda c, **k: calls.append(c) or subprocess.CompletedProcess(c, 0, "abc1234ff\n"))
    assert fm.fetch_repo(r, str(tmp_path), dry=False)
    assert fm.fetch_repo(r, str(tmp_path), dry=False)
    assert all("clone" not in c for c in calls)
    assert (dest / "m.py").read_text().count("except ImportError") == 1


def test_completed_weights_are_skipped(tmp_path, monkeypatch):
    w = {"repo": "org/model", "dest": "checkpoints/model", "include": ["*.json"]}
    d = tmp_path / "checkpoints" / "model"
    d.mkdir(parents=True)
    (d / fm.MARKER).write_text(json.dumps([fm.entry(w)]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert fm.fetch_weights(w, str(tmp_path), dry=False)
