#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-python}
model=${1:-}
run=${2:-}
plan=${3:-${run:+$run/plan.npz}}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -n 1 | cut -d, -f1)
  export CUDA_VISIBLE_DEVICES
fi
echo "GPU $CUDA_VISIBLE_DEVICES"
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch does not see a CUDA GPU"
import flashinfer
import triton
print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| triton", triton.__version__, "| flashinfer",
      getattr(flashinfer, "__version__", "?"))
EOF
"$PY" -m pytest tests/test_gpu_kernels.py -v -s -rs

if [ -n "$model" ]; then
  : "${run:?run directory}"
  clip=$("$PY" -c "import json, sys; s = json.load(open(sys.argv[1])); print([c for c in s['test'] if c not in s.get('dropped', [])][0])" "data/splits/$model.json")
  "$PY" -m decoupattn generate --model "$model" --run-dir "$run" --dense --clips "$clip"
  "$PY" -m decoupattn generate --model "$model" --run-dir "$run" --plan "$plan" --clips "$clip"
  "$PY" -m decoupattn evaluate --model "$model" --run-dir "$run" --clips "$clip" --no-lpips
fi
