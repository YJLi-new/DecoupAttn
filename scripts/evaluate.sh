#!/usr/bin/env bash
set -euo pipefail
model=$1 run=$2 plan=${3:-$2/plan.npz}
python -m decoupattn generate --model "$model" --run-dir "$run" --dense
python -m decoupattn generate --model "$model" --run-dir "$run" --plan "$plan"
python -m decoupattn evaluate --model "$model" --run-dir "$run" --tag decoupattn --ref dense
