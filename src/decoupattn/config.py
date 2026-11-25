from __future__ import annotations

import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "models")
DATA_ROOT_TOKEN = "@DATA_ROOT@"


def data_root() -> str:
    root = os.environ.get("DECOUPATTN_DATA_ROOT")
    if not root:
        raise RuntimeError("set DECOUPATTN_DATA_ROOT to the data root (see README.md)")
    return root


def _resolve(x):
    if isinstance(x, dict):
        return {k: _resolve(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_resolve(v) for v in x]
    if isinstance(x, str) and DATA_ROOT_TOKEN in x:
        return x.replace(DATA_ROOT_TOKEN, data_root())
    return x


def load_config(model: str, path: str | None = None, resolve: bool = True) -> dict:
    with open(path or os.path.join(CONFIG_DIR, f"{model}.toml"), "rb") as f:
        cfg = tomllib.load(f)
    return _resolve(cfg) if resolve else cfg


def repo_path(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
