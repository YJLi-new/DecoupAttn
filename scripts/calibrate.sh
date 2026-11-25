#!/usr/bin/env bash
set -euo pipefail
model=$1 run=$2
shift 2
python -m decoupattn calibrate --model "$model" --run-dir "$run" "$@"
