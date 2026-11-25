from __future__ import annotations

import json
import os

from ..constants import SEED
from . import _svoo_port as P
from ._common import Baseline, PerBranch


class SVOO(Baseline):
    name = "svoo"
    label = "SVOO"
    upstream = ("SVOO",)

    def profile_csv(self, run_dir: str) -> str:
        return os.path.join(self.work_dir(run_dir), "profile.csv")

    def match_file(self, run_dir: str) -> str:
        return os.path.join(self.work_dir(run_dir), "match.json")

    def knobs(self, run_dir: str) -> dict:
        s = dict(self.settings)
        path = self.match_file(run_dir)
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            s.update(top_p=m["top_p"], kr_min=m["kr_min"], kr_max=m["kr_max"])
        return s

    def needs_match(self) -> bool:
        s = self.settings
        return bool(s.get("rematch")) or any(key not in s for key in ("top_p", "kr_min", "kr_max"))

    def _hook(self, adapter, csv: str, settings: dict, warmup: bool = True):
        def make():
            return P.SVOOHook(csv, tuple(adapter.latent_grid), adapter.n_layers, adapter.n_heads, adapter.head_dim,
                              settings, lse=adapter.needs_lse, warmup=warmup)
        if settings.get("per_branch") and len(adapter.branches) > 1:
            return PerBranch(make, adapter.branches)
        return make()

    def prepare(self, adapter, run_dir: str, device: str = "cuda", log=print) -> None:
        work = self.work_dir(run_dir)
        os.makedirs(work, exist_ok=True)
        csv = P.profile(adapter, self.settings, work, SEED, log)
        if not self.needs_match() or os.path.exists(self.match_file(run_dir)):
            return
        s = self.settings
        clips = adapter.clips("calib")[:s["match_clips"]]

        def density(top_p, kr_min, kr_max):
            hook = self._hook(adapter, csv, {**s, "top_p": top_p, "kr_min": kr_min, "kr_max": kr_max}, warmup=False)
            vals = []
            for clip in clips:
                hook.reset()
                adapter.generate(clip, hook, SEED)
                vals.append(hook.summary()["exec_density"])
            d = sum(vals) / len(vals)
            log(f"svoo match top_p {top_p:.4f} window [{kr_min:.4f}, {kr_max:.4f}]: executed density {d:.4f}")
            return d

        family = s["official"]
        res = P.match(density, s["target_density"], s["match_tol"], s["match_iters"], P.OFFICIAL_TOP_P[family],
                      P.OFFICIAL_WINDOW[family])
        P.write_json(self.match_file(run_dir), {**res, "clips": clips, "seed": SEED, "profile": csv})

    def hook(self, adapter, run_dir: str):
        csv = self.profile_csv(run_dir)
        if not os.path.exists(csv):
            raise FileNotFoundError(f"{csv} is missing: run the baseline's prepare step first")
        return self._hook(adapter, csv, self.knobs(run_dir))


BASELINE = SVOO
