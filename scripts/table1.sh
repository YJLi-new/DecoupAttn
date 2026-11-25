#!/usr/bin/env bash
set -euo pipefail

target=${1:?model or "all"}
runs=${2:?runs directory}
plans=${3:-}
PY=${PY:-python}
METHODS=${METHODS:-calibatt,svoo,svg2,svgear,sol,pisa,sparsepr}
MODELS="cp1 cp25 wan21 wan22 hy nano t25 dr alaya unirelight"

py_of() {
  case "$1" in
    calibatt) echo "${PY_CALIBATT:-$2}" ;;
    svoo) echo "${PY_SVOO:-$2}" ;;
    svg2 | svgear) echo "${PY_SVG:-$2}" ;;
    sol | pisa) echo "${PY_SOLPISA:-$2}" ;;
    sparsepr) echo "${PY_SPARSEPR:-$2}" ;;
  esac
}

row() {
  local model=$1 run=$runs/$1 py var
  var=PY_$(echo "$model" | tr '[:lower:]' '[:upper:]')
  py=${!var:-$PY}
  local plan=()
  if [ -n "$plans" ]; then plan=(--plan "$plans/$model/plan.npz"); fi
  "$py" -m decoupattn table1 --model "$model" --run-dir "$run" ${plan[@]+"${plan[@]}"} \
    --stages dense,floor,ours,metrics,timing --methods none
  if [ "$METHODS" != "none" ]; then
    local m
    for m in ${METHODS//,/ }; do
      "$(py_of "$m" "$py")" -m decoupattn table1 --model "$model" --run-dir "$run" \
        --stages baselines,metrics,timing --methods "$m"
    done
  fi
  "$py" -m decoupattn table1 --model "$model" --run-dir "$run" --stages assemble --check
}

if [ "$target" = "all" ]; then
  for model in $MODELS; do row "$model"; done
  "$PY" -m decoupattn table1 --model all --run-dir "$runs" --check
else
  row "$target"
fi
