from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

KIND_DENSE, KIND_STATIC, KIND_STATIC_DYNAMIC, KIND_COMPONLY = 0, 1, 2, 3
EST_CENTROID, EST_FULL = 0, 1

_KEYS = {
    "mask": "mask",
    "step_row": "step_index",
    "block_shape": "tile",
    "query_tiling": "qtile",
    "kind": "oc_opt",
    "k_per_head": "K_per_head",
    "estimator": "oc_comp",
    "choice": "choice",
    "rows": "rows",
    "row_weight": "row_weight",
    "reassign": "upr_refresh",
    "dynamic_blocks": "oc_kd",
}


@dataclass
class Plan:
    mask: np.ndarray
    step_row: np.ndarray
    block_shape: tuple[int, int, int]
    kind: np.ndarray
    k_per_head: np.ndarray
    estimator: np.ndarray
    query_tiling: str = "frame"
    choice: np.ndarray | None = None
    rows: np.ndarray | None = None
    row_weight: np.ndarray | None = None
    reassign: str | None = None
    dynamic_blocks: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.kind.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.kind.shape[1])

    @property
    def n_heads(self) -> int:
        return int(self.kind.shape[2])

    def heads(self, step: int, layer: int) -> tuple[np.ndarray, np.ndarray]:
        k = self.kind[step, layer]
        return np.nonzero(k == KIND_DENSE)[0], np.nonzero(k != KIND_DENSE)[0]

    def call_mask(self, step: int, layer: int) -> np.ndarray:
        return self.mask[int(self.step_row[step]), layer]

    def full_estimator(self, step: int, layer: int) -> bool:
        return bool(self.estimator[step, layer] == EST_FULL)

    def save(self, path: str) -> None:
        kw = {}
        for attr, key in _KEYS.items():
            val = getattr(self, attr)
            if val is None:
                continue
            kw[key] = np.array(val) if isinstance(val, (str, tuple)) else val
        kw.update(self.extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **kw)
        os.replace(tmp, path)


def load_plan(path: str) -> Plan:
    z = np.load(path)
    kw = {}
    for attr, key in _KEYS.items():
        if key in z.files:
            kw[attr] = z[key]
    kw["block_shape"] = tuple(int(x) for x in kw["block_shape"])
    kw["query_tiling"] = str(kw.get("query_tiling", "frame"))
    if "reassign" in kw:
        kw["reassign"] = str(kw["reassign"])
    known = set(_KEYS.values())
    kw["extra"] = {k: z[k] for k in z.files if k not in known}
    return Plan(**kw)
