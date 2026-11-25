# Third-Party Notices

Model and baseline code is not included; the fetch scripts download it, and it keeps its own license. The files below
adapt third-party code or contain lines of it.

| Upstream | License | Files |
| --- | --- | --- |
| FlashInfer 0.2.10 | Apache-2.0 | `src/decoupattn/kernels/block_sparse.py` |
| Diffusers 0.34 / 0.41 | Apache-2.0 | `src/decoupattn/models/_wanapi_wan.py`, `cosmos3_nano.py`, `hunyuan_video.py` |
| FastVideo | Apache-2.0 | `src/decoupattn/models/_hunyuan.py` |
| LightX2V | Apache-2.0 | `src/decoupattn/models/_wanapi_wan.py`, `wan2_2.py`; `third_party/baselines/LightX2V/` |
| scikit-image | BSD-3-Clause | `src/decoupattn/eval/metrics.py` |
| ml-calibatt | Apple sample code license, Copyright (C) 2026 Apple Inc. | `third_party/baselines/ml-calibatt/` |
| piecewise-sparse-attention | MIT, Copyright (c) 2026 Haopeng Li | `third_party/baselines/piecewise-sparse-attention/` |
| MovieGen Video Bench | CC BY-NC 4.0 | `data/prompts/` (line numbers only) |
