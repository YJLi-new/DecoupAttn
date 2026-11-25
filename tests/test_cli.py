from __future__ import annotations

import os
import subprocess
import sys

import pytest

from decoupattn.cli import parser

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def run(*args):
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "decoupattn", *args], capture_output=True, text=True, env=env, check=False)


def test_help():
    r = run("--help")
    assert r.returncode == 0
    for cmd in ("calibrate", "generate", "evaluate", "time", "table1", "table2"):
        assert cmd in r.stdout


@pytest.mark.parametrize("cmd", ["calibrate", "variants", "generate", "evaluate", "time", "table1", "table2"])
def test_command_help(cmd):
    r = run(cmd, "--help")
    assert r.returncode == 0 and "--model" in r.stdout and "--run-dir" in r.stdout


def test_parse():
    a = parser().parse_args(["generate", "--model", "dr", "--run-dir", "R", "--dense"])
    assert (a.model, a.run_dir, a.dense, a.plan, a.split, a.seed) == ("dr", "R", True, None, "test", 1000)
    a = parser().parse_args(["time", "--model", "alaya", "--run-dir", "R", "--plan", "p.npz"])
    assert (a.plan, a.clip, a.reps) == ("p.npz", None, None)
    assert parser().parse_args(["time", "--model", "alaya", "--run-dir", "R", "--reps", "2"]).reps == 2
    with pytest.raises(SystemExit):
        parser().parse_args(["generate", "--model", "unknown", "--run-dir", "R"])
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            parser().parse_args(["time", "--model", "alaya", "--run-dir", "R", "--reps", bad])


def test_ablation_flags():
    from decoupattn.cli import overrides
    a = parser().parse_args(["generate", "--model", "t25", "--run-dir", "R", "--schedule", "rebuild", "--untied",
                             "--estimator", "Centroid"])
    assert overrides(a) == {"schedule": "rebuild", "tied": False, "estimator": "Centroid"}
    a = parser().parse_args(["time", "--model", "t25", "--run-dir", "R", "--tied"])
    assert overrides(a) == {"tied": True}
    assert overrides(parser().parse_args(["generate", "--model", "t25", "--run-dir", "R"])) == {}
