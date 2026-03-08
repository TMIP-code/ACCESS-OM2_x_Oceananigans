#!/bin/bash

#PBS -N OM2-025_GPU
#PBS -P y99
#PBS -l mem=96GB
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
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
[ -n "${TIMESTEPPER:-}" ] && export TIMESTEPPER && echo "TIMESTEPPER=$TIMESTEPPER"
[ -n "${TM_SOURCE:-}" ] && export TM_SOURCE && echo "TM_SOURCE=$TM_SOURCE"
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
job_id="${PBS_JOBID:-interactive}"
# Route log to run-type subdirectory and strip redundant prefixes from filename
case "$NONLINEAR_SOLVER" in
    1year|10years|100years)
        run_log_dir=logs/julia/$PARENT_MODEL/standardrun
        log_file="$run_log_dir/${MODEL_CONFIG}_${NONLINEAR_SOLVER}_${job_id}.log"
        ;;
    long)
        run_log_dir=logs/julia/$PARENT_MODEL/standardrun
        log_file="$run_log_dir/${MODEL_CONFIG}_long_${NYEARS:-3000}years_${job_id}.log"
        ;;
    newton)
        lumpspray_tag="prec"
        [ "${LUMP_AND_SPRAY:-no}" = "yes" ] && lumpspray_tag="LSprec"
        run_log_dir=logs/julia/$PARENT_MODEL/periodic/NK
        log_file="$run_log_dir/${MODEL_CONFIG}_${LINEAR_SOLVER:-Pardiso}_${lumpspray_tag}_${job_id}.log"
        ;;
    anderson)
        run_log_dir=logs/julia/$PARENT_MODEL/periodic/AA
        log_file="$run_log_dir/${MODEL_CONFIG}_${AA_SOLVER:-SpeedMapping}_${job_id}.log"
        ;;
esac
mkdir -p "$run_log_dir"
echo "logging output in $log_file"
julia $JULIA_BOUNDS_FLAG --project "$SCRIPT" &> "$log_file"
echo "Done running $SCRIPT for PARENT_MODEL=$PARENT_MODEL"

# Submit CPU plot job after simulation
if [ "$NONLINEAR_SOLVER" = "1year" ]; then
    echo "Submitting plot_1year_age CPU job"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-025_plot_1year_age_job.sh
fi
if [ "$NONLINEAR_SOLVER" = "10years" ]; then
    echo "Submitting plot_10years_age CPU job"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-025_plot_10years_age_job.sh
fi
if [ "$NONLINEAR_SOLVER" = "100years" ]; then
    echo "Submitting plot_100years_age CPU job"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-025_plot_100years_age_job.sh
fi
if [ "$TRACE_SOLVER_HISTORY" = "yes" ] && { [ "$NONLINEAR_SOLVER" = "newton" ] || [ "$NONLINEAR_SOLVER" = "anderson" ]; }; then
    echo "Submitting plot_trace_history CPU job"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER",NONLINEAR_SOLVER="$NONLINEAR_SOLVER" \
        scripts/ACCESS-OM2-025_plot_trace_history_job.sh
fi
