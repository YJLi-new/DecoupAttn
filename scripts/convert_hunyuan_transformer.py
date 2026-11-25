from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
OFFICIAL = os.path.join("hunyuan-video-t2v-720p", "transformers", "mp_rank_00_model_states.pt")


def main():
    from decoupattn.config import load_config

    cfg = load_config("hy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(cfg["checkpoints"]["transformer"]), OFFICIAL))
    ap.add_argument("--out", default=cfg["checkpoints"]["transformer"])
    a = ap.parse_args()
    weights = os.path.join(a.out, "diffusion_pytorch_model.safetensors")
    if os.path.exists(weights):
        print(f"{weights} exists")
        return

    import torch
    from diffusers.loaders.single_file_utils import convert_hunyuan_video_transformer_to_diffusers
    from safetensors.torch import save_file

    from decoupattn.models import _hunyuan

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sd = torch.load(a.src, map_location="cpu", weights_only=True)
    sd = sd.get("module", sd)
    n_src = len(sd)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in convert_hunyuan_video_transformer_to_diffusers(sd).items()}

    _hunyuan.dcm_path(cfg["model"]["repo"])
    import fastvideo.models.hunyuan_hf.modeling_hunyuan as MH

    cfg_path = os.path.join(cfg["checkpoints"]["dcm_hy"], "transformer", "config.json")
    with open(cfg_path) as f:
        tcfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    with torch.device("meta"):
        model = MH.HunyuanVideoTransformer3DModel(**tcfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise SystemExit(f"conversion does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")

    os.makedirs(a.out, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(a.out, "config.json"))
    save_file(sd, weights + ".tmp")
    os.replace(weights + ".tmp", weights)
    print(f"{n_src} -> {len(sd)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
