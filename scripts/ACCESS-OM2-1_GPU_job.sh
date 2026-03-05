#!/bin/bash

#PBS -N run_OM2-1_GPU
#PBS -P y99
#PBS -l mem=96GB
#PBS -q gpuvolta
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

NONLINEAR_SOLVER=${NONLINEAR_SOLVER:-1year}  # 1year | 10years | 100years | newton | anderson
echo "NONLINEAR_SOLVER=$NONLINEAR_SOLVER"

# Select Julia script based on NONLINEAR_SOLVER
case "$NONLINEAR_SOLVER" in
    1year)    SCRIPT="src/run_1year.jl" ;;
    10years)  SCRIPT="src/run_10years.jl" ;;
    100years) SCRIPT="src/run_100years.jl" ;;
    long)     SCRIPT="src/run_long.jl" ;;
    newton)   SCRIPT="src/solve_periodic_newton.jl" ;;
    anderson) SCRIPT="src/solve_periodic_anderson.jl" ;;
    *)        echo "Unknown NONLINEAR_SOLVER=$NONLINEAR_SOLVER (must be: 1year, 10years, 100years, long, newton, anderson)"; exit 1 ;;
esac
echo "SCRIPT=$SCRIPT"

# Export solver-specific env vars if set
[ -n "${JVP_METHOD:-}" ] && export JVP_METHOD && echo "JVP_METHOD=$JVP_METHOD"
[ -n "${AA_SOLVER:-}" ] && export AA_SOLVER && echo "AA_SOLVER=$AA_SOLVER"
[ -n "${PRECONDITIONER_MATRIX_TYPE:-}" ] && export PRECONDITIONER_MATRIX_TYPE && echo "PRECONDITIONER_MATRIX_TYPE=$PRECONDITIONER_MATRIX_TYPE"
[ -n "${TIMESTEPPER:-}" ] && export TIMESTEPPER && echo "TIMESTEPPER=$TIMESTEPPER"
[ -n "${TRACE_SOLVER_HISTORY:-}" ] && export TRACE_SOLVER_HISTORY && echo "TRACE_SOLVER_HISTORY=$TRACE_SOLVER_HISTORY"
[ -n "${LINEAR_SOLVER:-}" ] && export LINEAR_SOLVER && echo "LINEAR_SOLVER=$LINEAR_SOLVER"
[ -n "${LUMP_AND_SPRAY:-}" ] && export LUMP_AND_SPRAY && echo "LUMP_AND_SPRAY=$LUMP_AND_SPRAY"
[ -n "${INITIAL_AGE:-}" ] && export INITIAL_AGE && echo "INITIAL_AGE=$INITIAL_AGE"
[ -n "${NYEARS:-}" ] && export NYEARS && echo "NYEARS=$NYEARS"

echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

# Unlimited stack size to avoid segfaults in MKL Pardiso METIS recursive reordering
# ulimit -s unlimited

echo "Running $SCRIPT for PARENT_MODEL=$PARENT_MODEL"
run_log_dir=logs/julia/run_ACCESS-OM2
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
# Build solver suffix for log filename (include AA_SOLVER for anderson runs)
solver_tag="${NONLINEAR_SOLVER}"
[ "$NONLINEAR_SOLVER" = "anderson" ] && [ -n "${AA_SOLVER:-}" ] && solver_tag="${NONLINEAR_SOLVER}_${AA_SOLVER}"
echo "logging output in $run_log_dir"
julia $JULIA_BOUNDS_FLAG --project "$SCRIPT" &> "$run_log_dir/run_ACCESS-OM2_${MODEL_CONFIG}_${solver_tag}_${job_id}.log"
echo "Done running $SCRIPT for PARENT_MODEL=$PARENT_MODEL"

# Submit CPU plot job after simulation
if [ "$NONLINEAR_SOLVER" = "1year" ]; then
    echo "Submitting plot_1year_age CPU job"
    qsub -v VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_plot_1year_age_job.sh
fi
if [ "$NONLINEAR_SOLVER" = "10years" ]; then
    echo "Submitting plot_10years_age CPU job"
    qsub -v VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_plot_10years_age_job.sh
fi
if [ "$NONLINEAR_SOLVER" = "100years" ]; then
    echo "Submitting plot_100years_age CPU job"
    qsub -v VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_plot_100years_age_job.sh
fi
if [ "$TRACE_SOLVER_HISTORY" = "yes" ] && { [ "$NONLINEAR_SOLVER" = "newton" ] || [ "$NONLINEAR_SOLVER" = "anderson" ]; }; then
    echo "Submitting plot_trace_history CPU job"
    qsub -v VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_plot_trace_history_job.sh
fi
