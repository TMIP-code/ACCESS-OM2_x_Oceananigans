#!/bin/bash

#PBS -N OM2-1_NK
#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l walltime=03:00:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

# Export solver-specific env vars if set
[ -n "${JVP_METHOD:-}" ] && export JVP_METHOD && echo "JVP_METHOD=$JVP_METHOD"
[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"
[ -n "${TRACE_SOLVER_HISTORY:-}" ] && export TRACE_SOLVER_HISTORY && echo "TRACE_SOLVER_HISTORY=$TRACE_SOLVER_HISTORY"
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
[ -n "${INITIAL_AGE:-}" ] && export INITIAL_AGE && echo "INITIAL_AGE=$INITIAL_AGE"

echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

lumpspray_tag="prec"
[ "${LUMP_AND_SPRAY:-no}" = "yes" ] && lumpspray_tag="LSprec"

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/periodic/NK
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_${LINEAR_SOLVER:-Pardiso}_${lumpspray_tag}_${job_id}.log"

echo "Running src/solve_periodic_NK.jl for PARENT_MODEL=$PARENT_MODEL"
echo "logging output in $log_file"
julia $JULIA_BOUNDS_FLAG --project src/solve_periodic_NK.jl &> "$log_file"
echo "Done running src/solve_periodic_NK.jl for PARENT_MODEL=$PARENT_MODEL"

# Submit CPU plot job if tracing solver history
if [ "$TRACE_SOLVER_HISTORY" = "yes" ]; then
    echo "Submitting plot_trace_history CPU job"
    qsub -v VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER",NONLINEAR_SOLVER=newton \
        scripts/ACCESS-OM2-1_plot_trace_history_job.sh
fi
