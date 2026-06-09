#!/bin/bash

#PBS -P y99
#PBS -l mem=192GB
#PBS -q normal
#PBS -l ncpus=48
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/TM
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"

# Two ways to build the averaged matrix (TM/{MC}/avg/M.jld2), selected by
# TMAVG_METHOD:
#   monthly  (default) — one Jacobian per preprocessed monthly velocity field,
#                        averaged. Reuses the const build (matrix_setup.jl); needs
#                        NO prior 1-year run. Coarser (assumes zero ∂η/∂t).
#   snapshot           — one Jacobian per velocity snapshot saved by a 1-year run;
#                        captures ∂η/∂t. Requires run1yr output on disk.
TMAVG_METHOD=${TMAVG_METHOD:-monthly}
case "$TMAVG_METHOD" in
    monthly)  avg_script=src/create_monthly_matrices.jl;  tag="monthly_matrices" ;;
    snapshot) avg_script=src/create_snapshot_matrices.jl; tag="snapshot_matrices" ;;
    *) echo "ERROR: TMAVG_METHOD must be monthly or snapshot (got: $TMAVG_METHOD)" >&2; exit 1 ;;
esac

echo "Building averaged matrix (TMAVG_METHOD=$TMAVG_METHOD -> $avg_script) for MODEL_CONFIG=$MODEL_CONFIG"
log_file="$run_log_dir/${tag}_${MODEL_CONFIG}_${job_id}.log"
julia $JULIA_BOUNDS_FLAG --project "$avg_script" &> "$log_file"
echo "Done building averaged matrix"
echo "logged output in $log_file"
