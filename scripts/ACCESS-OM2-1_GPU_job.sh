#!/bin/bash

#PBS -N run_OM2-1_GPU
#PBS -P y99
#PBS -l mem=47GB
#PBS -q gpuvolta
#PBS -l walltime=01:00:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

# parent model (falls back to existing env or sensible default)
PARENT_MODEL=ACCESS-OM2-1
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
ADVECTION_SCHEME=${ADVECTION_SCHEME:-centered2}
SOLVE_METHOD=${SOLVE_METHOD:-1year}  # 1year | newton | anderson
MODEL_CONFIG="${VELOCITY_SOURCE}_${W_FORMULATION}_${ADVECTION_SCHEME}"
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME
echo "environment variables set:"
echo "PARENT_MODEL=$PARENT_MODEL"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"
echo "ADVECTION_SCHEME=$ADVECTION_SCHEME"
echo "SOLVE_METHOD=$SOLVE_METHOD"

# Select Julia script based on SOLVE_METHOD
case "$SOLVE_METHOD" in
    1year)    SCRIPT="src/run_1year.jl" ;;
    newton)   SCRIPT="src/solve_periodic_newton.jl" ;;
    anderson) SCRIPT="src/solve_periodic_anderson.jl" ;;
    *)        echo "Unknown SOLVE_METHOD=$SOLVE_METHOD (must be: 1year, newton, anderson)"; exit 1 ;;
esac
echo "SCRIPT=$SCRIPT"

# Export solver-specific env vars if set
[ -n "${JVP_METHOD:-}" ] && export JVP_METHOD && echo "JVP_METHOD=$JVP_METHOD"
[ -n "${ACCELERATION_METHOD:-}" ] && export ACCELERATION_METHOD && echo "ACCELERATION_METHOD=$ACCELERATION_METHOD"
[ -n "${PRECONDITIONER_MATRIX_TYPE:-}" ] && export PRECONDITIONER_MATRIX_TYPE && echo "PRECONDITIONER_MATRIX_TYPE=$PRECONDITIONER_MATRIX_TYPE"

# Bounds checking: set CHECK_BOUNDS=yes to run julia with --check-bounds=yes
CHECK_BOUNDS=${CHECK_BOUNDS:-no}
JULIA_BOUNDS_FLAG=""
if [ "$CHECK_BOUNDS" = "yes" ]; then
    JULIA_BOUNDS_FLAG="--check-bounds=yes"
    echo "CHECK_BOUNDS=yes (running julia with --check-bounds=yes)"
fi

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "cd $repo_root"
cd "$repo_root"

echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

# Unlimited stack size to avoid segfaults in MKL Pardiso METIS recursive reordering
ulimit -s unlimited

echo "Running $SCRIPT for PARENT_MODEL=$PARENT_MODEL"
run_log_dir="$repo_root/logs/julia/run_ACCESS-OM2"
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
echo "logging output in $run_log_dir"
julia $JULIA_BOUNDS_FLAG --project "$repo_root/$SCRIPT" &> "$run_log_dir/run_ACCESS-OM2_${MODEL_CONFIG}_${SOLVE_METHOD}_${job_id}.log"
echo "Done running $SCRIPT for PARENT_MODEL=$PARENT_MODEL"

# Submit CPU plot job after 1-year simulation
if [ "$SOLVE_METHOD" = "1year" ]; then
    echo "Submitting plot_1year_age CPU job"
    qsub -v VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME" \
        "$repo_root/scripts/ACCESS-OM2-1_plot_1year_age_job.sh"
fi
