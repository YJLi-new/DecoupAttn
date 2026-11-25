from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def step_rows(steps: int, rows) -> np.ndarray:
    return np.array([int(np.argmin([abs(s - r) for r in rows])) for s in range(steps)], np.int64)


def row_weights(steps: int, rows) -> np.ndarray:
    si = step_rows(steps, rows)
    return np.bincount(si[1:], minlength=len(rows)).astype(np.float64)


def make_units(n_layers: int, n_heads: int, n_rows: int) -> np.ndarray:
    return np.array([(r, l, h) for r in range(n_rows) for l in range(1, n_layers) for h in range(n_heads)])


def k_blocks(theta: float, n_blocks: int) -> int:
    return max(1, round(theta * n_blocks))


def thirds(n_layers: int) -> list[tuple[int, int]]:
    return [(1 + i * (n_layers - 1) // 3, (i + 1) * (n_layers - 1) // 3) for i in range(3)]


def cluster_cache(til, reassign, tied: bool = False):
    from ..clustering import ClusterCache

    schedule = reassign if isinstance(reassign, str) else ("reassign" if reassign else "frozen")
    return ClusterCache(til, schedule, tied)


@dataclass
class Units:
    layout: str
    steps: int
    rows: tuple
    n_layers: int
    n_heads: int
    branches: tuple = ("cond",)
    units: np.ndarray = field(init=False)

    def __post_init__(self):
        self.rows = tuple(int(r) for r in self.rows)
        if self.layout == "steps":
            assert self.rows == (1, 2, 3) and self.steps == 4, (self.rows, self.steps)
        self.units = make_units(self.n_layers, self.n_heads, len(self.rows))

    @classmethod
    def of(cls, adapter) -> Units:
        return cls(adapter.layout, adapter.steps, adapter.rows, adapter.n_layers, adapter.n_heads, tuple(adapter.branches))

    def __len__(self) -> int:
        return len(self.units)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_calls(self) -> int:
        return self.n_rows * (self.n_layers - 1)

    @property
    def step_row(self) -> np.ndarray:
        return np.arange(4) if self.layout == "steps" else step_rows(self.steps, self.rows)

    @property
    def row_weight(self) -> np.ndarray:
        return row_weights(self.steps, self.rows)

    @property
    def weights(self) -> np.ndarray:
        if self.layout == "steps":
            return np.ones(len(self.units))
        return self.row_weight[self.units[:, 0]]

    @property
    def calls(self) -> np.ndarray:
        return self.units[:, 0] * (self.n_layers - 1) + (self.units[:, 1] - 1)

    @property
    def unit_steps(self) -> np.ndarray:
        return np.asarray(self.rows)[self.units[:, 0]]

    @property
    def observed_steps(self) -> int:
        return max(self.rows) + 1

    @property
    def measured_branches(self) -> tuple:
        return ("cond",) if self.layout == "steps" else tuple(self.branches)

    def table_row(self, step: int, branch: str) -> int | None:
        if step not in self.rows or branch not in self.measured_branches:
            return None
        return self.rows.index(step)
