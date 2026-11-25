#!/usr/bin/env bash
set -euo pipefail

target=${1:?model or "all"}
runs=${2:?runs directory}
plans=${3:-}
PY=${PY:-python}
MODELS="cp1 cp25 wan21 wan22 hy nano t25 dr alaya unirelight"

column() {
  local model=$1 run=$runs/$1 py var
  var=PY_$(echo "$model" | tr '[:lower:]' '[:upper:]')
  py=${!var:-$PY}
  if [ -n "$plans" ] && [ -d "$plans/$model" ]; then
    (cd "$plans/$model" && find . -type f) | while read -r f; do
      if [ ! -e "$run/$f" ]; then
        mkdir -p "$(dirname "$run/$f")"
        cp "$plans/$model/$f" "$run/$f"
      fi
    done
  fi
  local variants=()
  if [ -n "${VARIANTS:-}" ]; then variants=(--variants "$VARIANTS"); fi
  "$py" -m decoupattn table2 --model "$model" --run-dir "$run" ${variants[@]+"${variants[@]}"} --check
}

if [ "$target" = "all" ]; then
  for model in $MODELS; do column "$model"; done
  "$PY" -m decoupattn table2 --model all --run-dir "$runs" --check
else
  column "$target"
fi
