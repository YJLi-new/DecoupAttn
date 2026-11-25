from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext

import numpy as np
import torch

from ..constants import SEED
from ..context import CallContext

Hook = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, CallContext], "torch.Tensor | None"]


class ModelAdapter(ABC):
    name: str = ""
    family: str = "video"
    steps: int = 4
    layout: str = "steps"
    rows: tuple[int, ...] = (1, 2, 3)
    branches: tuple[str, ...] = ("cond",)
    n_layers: int = 0
    n_heads: int = 0
    head_dim: int = 128
    latent_grid: tuple[int, int, int] = (0, 0, 0)
    block_shape: tuple[int, int, int] | None = None
    needs_lse: bool = False
    latency: str = "span"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ctx = CallContext()
        self.timer: Callable[[str], AbstractContextManager] | None = None

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        pass

    @abstractmethod
    def clips(self, split: str) -> list[str]:
        pass

    @abstractmethod
    def generate(self, clip: str, hook: Hook | None = None, seed: int = SEED, steps: int | None = None) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def out_proj(self, layer: int, step: int = 0) -> torch.Tensor:
        pass

    @abstractmethod
    def dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pass

    @contextmanager
    def dit_timer(self, tag: str):
        hook = getattr(self, "hook", None)
        build = getattr(hook, "build_step", None)
        if build is not None:
            build(self.ctx, getattr(self.timer, "excluded", nullcontext))
        with self.timer(tag) if self.timer is not None else nullcontext():
            yield
        after = getattr(hook, "after_call", None)
        if after is not None:
            after(self.ctx)

    def dense_lse(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"{self.name} does not merge attention outputs through the log-sum-exp")

    def dense_heads(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.dense(q[:, :, idx], k[:, :, idx], v[:, :, idx])

    def expert(self, step: int) -> int:
        return 0

    def conditions(self, clip: str) -> dict[str, np.ndarray]:
        return {}
