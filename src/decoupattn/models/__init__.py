from __future__ import annotations

import importlib

from .base import Hook, ModelAdapter

MODELS = {
    "cp1": "cosmos_predict1:CosmosPredict1",
    "cp25": "cosmos_predict2_5:CosmosPredict25",
    "wan21": "wan2_1:Wan21",
    "wan22": "wan2_2:Wan22",
    "hy": "hunyuan_video:HunyuanVideo",
    "nano": "cosmos3_nano:Cosmos3Nano",
    "t25": "cosmos_transfer2_5:CosmosTransfer25",
    "dr": "diffusion_renderer:DiffusionRenderer",
    "alaya": "alaya_renderer:AlayaRenderer",
    "unirelight": "unirelight:UniRelight",
}


def adapter_class(model: str) -> type[ModelAdapter]:
    mod, cls = MODELS[model].split(":")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), cls)


def get_adapter(model: str, cfg: dict | None = None) -> ModelAdapter:
    from ..config import load_config

    return adapter_class(model)(cfg if cfg is not None else load_config(model))


__all__ = ["MODELS", "Hook", "ModelAdapter", "adapter_class", "get_adapter"]
