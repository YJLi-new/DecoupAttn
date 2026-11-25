from __future__ import annotations

import importlib

from ._common import Baseline

METHODS = {
    "calibatt": ("calibatt", "CalibAtt"),
    "svoo": ("svoo", "SVOO"),
    "svg2": ("svg2", "SVG2"),
    "svgear": ("svgear", "SVG-EAR"),
    "sol": ("sol", "Sol-Attn"),
    "pisa": ("pisa", "PISA"),
    "sparsepr": ("sparsepr", "SparsePR"),
}


def get_baseline(method: str, model: str, **overrides) -> Baseline:
    mod, _ = METHODS[method]
    cls = importlib.import_module(f"{__name__}.{mod}").BASELINE
    return cls(model, **overrides)


__all__ = ["METHODS", "Baseline", "get_baseline"]
