#!/usr/bin/env bash
set -euo pipefail
model=$1 run=$2 plan=${3:-$2/plan.npz}
python -m decoupattn time --model "$model" --run-dir "$run" --plan "$plan"
