from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

import pytest

from decoupattn.baselines import METHODS, _common
from decoupattn.config import REPO_ROOT

spec_ = importlib.util.spec_from_file_location("fetch_baselines",
                                               os.path.join(REPO_ROOT, "scripts", "fetch_baselines.py"))
fb = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(fb)
SOURCES = fb.load_sources()
MINE = ("Sparse-VideoGen", "SparsePR", "Sana", "piecewise-sparse-attention")


@pytest.mark.parametrize("name", MINE)
def test_sources_are_pinned_and_complete(name):
    s = SOURCES[name]
    assert s["name"] == name and len(s["commit"]) == 40 and int(s["commit"], 16) >= 0
    assert s["url"].startswith("https://github.com/") and s["license"]
    assert s["used_by"] and all(m in METHODS for m in s["used_by"])
    for p in fb.patch_files(s):
        assert os.path.isfile(p)


def test_the_wrappers_find_the_checkout_the_script_makes(tmp_path, monkeypatch):
    monkeypatch.setenv("DECOUPATTN_DATA_ROOT", str(tmp_path))
    dest = fb.destination(SOURCES["SparsePR"], str(tmp_path))
    with pytest.raises(FileNotFoundError):
        _common.upstream_path("SparsePR")
    os.makedirs(dest)
    assert _common.upstream_path("SparsePR", "src") == os.path.join(dest, "src")


def test_dry_run_prints_the_commands_and_fetches_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DECOUPATTN_DATA_ROOT", str(tmp_path))
    assert fb.main(["--dry-run", "--name", "piecewise-sparse-attention", "--name", "Sparse-VideoGen"]) == 0
    out = capsys.readouterr().out
    for name in ("piecewise-sparse-attention", "Sparse-VideoGen"):
        assert SOURCES[name]["url"] in out and SOURCES[name]["commit"] in out
    assert "git -C" in out and "apply" in out and "0001-return-lse.patch" in out
    assert "submodule update" in out and "flashinfer" in out
    assert "SparsePR" not in out and not os.listdir(tmp_path)


def test_dry_run_without_a_data_root_keeps_the_variable(monkeypatch, capsys):
    monkeypatch.delenv("DECOUPATTN_DATA_ROOT", raising=False)
    assert fb.main(["--dry-run", "--name", "Sana"]) == 0
    assert '"$DECOUPATTN_DATA_ROOT/baselines/Sana"' in capsys.readouterr().out


def test_list_and_unknown_names(capsys):
    assert fb.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in MINE)
    with pytest.raises(SystemExit):
        fb.main(["--name", "no-such-repository", "--dry-run"])


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
                    *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_fetch_checks_out_the_commit_and_applies_patches_once(tmp_path, capsys):
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "--quiet")
    git(up, "config", "uploadpack.allowReachableSHA1InWant", "true")
    (up / "kernel.py").write_text("def f():\n    return 1\n")
    git(up, "add", "kernel.py")
    git(up, "commit", "--quiet", "-m", "first")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=up, check=True, capture_output=True,
                            text=True).stdout.strip()
    (up / "kernel.py").write_text("def f(lse=False):\n    return 1\n")
    spec_dir = tmp_path / "specs" / "Fake"
    spec_dir.mkdir(parents=True)
    patch = subprocess.run(["git", "diff"], cwd=up, check=True, capture_output=True, text=True).stdout
    (spec_dir / "0001-lse.patch").write_text(patch)
    git(up, "commit", "--quiet", "-am", "later work that must not be fetched")
    (spec_dir / "source.toml").write_text(
        f'name = "Fake"\nurl = "{up.as_posix()}"\ncommit = "{commit}"\ndest = "fake-dest"\nlicense = "MIT"\n'
        'submodules = false\npatches = ["0001-lse.patch"]\nused_by = ["pisa"]\n'
        f'[[requires]]\npath = "third_party/dep"\nurl = "{up.as_posix()}"\ncommit = "{commit}"\n')
    src = fb.load_sources(str(tmp_path / "specs"))["Fake"]
    root = str(tmp_path / "data")
    assert fb.state(src, root) == "missing"
    assert fb.fetch(src, root, dry=False)
    dest = tmp_path / "data" / "baselines" / "fake-dest"
    assert (dest / "kernel.py").read_text() == "def f(lse=False):\n    return 1\n"
    assert (dest / "third_party" / "dep" / "kernel.py").read_text() == "def f():\n    return 1\n"
    assert fb.state(src, root) == "present"
    assert fb.fetch(src, root, dry=False)
    assert (dest / "kernel.py").read_text() == "def f(lse=False):\n    return 1\n"
    assert "applied" not in capsys.readouterr().out.split("present")[-1]

    other = {**src, "commit": "0" * 40}
    assert not fb.fetch(other, root, dry=False)
    assert fb.state(src, root) == "present"
