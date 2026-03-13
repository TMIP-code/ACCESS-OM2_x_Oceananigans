#!/bin/bash
# Convenience wrapper: submits scatter and datashader TM plot jobs in parallel.
# Usage (from repo root): bash scripts/plotting/plot_TM.sh
# Accepts env vars: PARENT_MODEL, VELOCITY_SOURCE, etc. (same as PBS scripts)

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
source scripts/env_defaults.sh

MODEL_SHORT="${MODEL_SHORT:-${PARENT_MODEL}}"
WALLTIME="${WALLTIME_PLOT:-00:30:00}"

common_vars="PARENT_MODEL=${PARENT_MODEL},VELOCITY_SOURCE=${VELOCITY_SOURCE},W_FORMULATION=${W_FORMULATION},ADVECTION_SCHEME=${ADVECTION_SCHEME},TIMESTEPPER=${TIMESTEPPER}"

SCATTER_JOB=$(qsub \
    -N "${MODEL_SHORT}_plotTM_scatter" -l walltime="${WALLTIME}" \
    -v "${common_vars}" \
    scripts/plotting/plot_TM_scatter.sh)
echo "Scatter: $SCATTER_JOB"

DATASHADER_JOB=$(qsub \
    -N "${MODEL_SHORT}_plotTM_datashader" -l walltime="${WALLTIME}" \
    -v "${common_vars}" \
    scripts/plotting/plot_TM_datashader.sh)
echo "Datashader: $DATASHADER_JOB"
