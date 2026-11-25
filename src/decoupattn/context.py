from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallContext:
    step: int = 0
    layer: int = 0
    branch: str = "cond"
    sigma: float = 1.0
    text: tuple | None = None

    def at(self, step: int | None = None, layer: int | None = None, branch: str | None = None) -> CallContext:
        if step is not None:
            self.step = step
        if layer is not None:
            self.layer = layer
        if branch is not None:
            self.branch = branch
        return self
