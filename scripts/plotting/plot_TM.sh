#!/bin/bash
# Convenience wrapper: submits TM datashader plot jobs for all comparison pairs.
# Usage (from repo root): bash scripts/plotting/plot_TM.sh
# Accepts env vars: PARENT_MODEL, VELOCITY_SOURCE, etc. (same as PBS scripts)

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
source scripts/env_defaults.sh

MODEL_SHORT="${MODEL_SHORT:-${PARENT_MODEL}}"
WALLTIME="${WALLTIME_PLOT:-00:30:00}"

common_vars="PARENT_MODEL=${PARENT_MODEL},VELOCITY_SOURCE=${VELOCITY_SOURCE},W_FORMULATION=${W_FORMULATION},ADVECTION_SCHEME=${ADVECTION_SCHEME},TIMESTEPPER=${TIMESTEPPER}"

for pair in const:avg; do
    lx="${pair%%:*}"; ly="${pair#*:}"
    job=$(qsub \
        -N "${MODEL_SHORT}_plotTM_${ly}_vs_${lx}" -l walltime="${WALLTIME}" \
        -v "${common_vars},TM_LABEL_X=${lx},TM_LABEL_Y=${ly}" \
        scripts/plotting/plot_TM_datashader.sh)
    echo "Plot TM ${ly} vs ${lx}: $job"
done
