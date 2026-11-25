from __future__ import annotations

import os

from ..constants import SEED
from ._common import Baseline


class CalibAtt(Baseline):
    name = "calibatt"
    label = "CalibAtt"
    upstream = ("ml-calibatt",)

    def __init__(self, model: str, **overrides):
        super().__init__(model, **overrides)
        if self.settings.get("pipeline") == "official":
            self.standalone, self.reference = True, "calibatt_dense"
            self.upstream = ("ml-calibatt", "LightX2V")
        self._pipeline = self._run_dir = None

    def agg_dir(self, run_dir: str) -> str:
        return os.path.join(self.work_dir(run_dir), "agg")

    def prepare(self, adapter, run_dir: str, device: str = "cuda", log=print) -> None:
        work, self._run_dir = self.work_dir(run_dir), run_dir
        if self.standalone:
            from . import _calibatt_wan as W
            W.prepare(adapter, self.settings, work, log)
            return
        from . import _calibatt_agg as A
        from . import _calibatt_port as P
        if os.path.lexists(os.path.join(self.agg_dir(run_dir), "skip_lists.pt")):
            return
        clips = adapter.clips("calib")
        clips = clips[: self.settings["calib_clips"]] if self.settings["calib_clips"] else clips
        P.capture(adapter, self.settings, os.path.join(work, "capture"), clips, SEED, device, log)
        A.aggregate(os.path.join(work, "capture"), self.agg_dir(run_dir), self.settings, log)

    def hook(self, adapter, run_dir: str):
        if self.standalone:
            raise NotImplementedError("CalibAtt on this model runs in its official pipeline: use generate() and time()")
        from . import _calibatt_port as P
        agg = self.agg_dir(run_dir)
        if not os.path.lexists(os.path.join(agg, "skip_lists.pt")):
            raise FileNotFoundError(f"{agg}/skip_lists.pt is missing: run the baseline's prepare step first")
        return P.infer_hook(adapter, self.settings, agg)

    def _official(self, adapter, run_dir: str | None):
        from . import _calibatt_wan as W
        run_dir = run_dir or self._run_dir
        if run_dir is None:
            raise ValueError("CalibAtt in its official pipeline reads its calibration from the run directory: call "
                             "prepare(adapter, run_dir) first or pass run_dir")
        if self._pipeline is None:
            self._pipeline, self._run_dir = W.Pipeline(adapter, self.settings, self.work_dir(run_dir)), run_dir
        return W, self._pipeline

    def generate(self, adapter, clips: list[str], out_dir: str, seed: int, dense: bool = False, log=print,
                 run_dir: str | None = None) -> None:
        if not self.standalone:
            return super().generate(adapter, clips, out_dir, seed, dense, log)
        W, pl = self._official(adapter, run_dir or self._run_dir or default_run_dir(out_dir))
        W.generate(pl, adapter, clips, out_dir, seed, dense, log)

    def time(self, adapter, clip: str, seed: int, log=print, run_dir: str | None = None,
             reps: int | None = None) -> dict:
        if not self.standalone:
            return super().time(adapter, clip, seed, log)
        W, pl = self._official(adapter, run_dir)
        return W.time_rows(pl, adapter, clip, seed, log, reps)


def default_run_dir(out_dir: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(out_dir)))


BASELINE = CalibAtt
