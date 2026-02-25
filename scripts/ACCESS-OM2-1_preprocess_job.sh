#!/bin/bash

#PBS -N preprocess_OM2-1_CPU
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

PARENT_MODEL=ACCESS-OM2-1
export PARENT_MODEL

# Downstream job submission flags – set via: qsub -v SUBMIT_OFFLINE_GPU=true,...
SUBMIT_OFFLINE_CPU=${SUBMIT_OFFLINE_CPU:-false}
SUBMIT_OFFLINE_GPU=${SUBMIT_OFFLINE_GPU:-false}
SUBMIT_MATRIX=${SUBMIT_MATRIX:-false}

# Variables forwarded to downstream jobs
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
ENABLE_AGE_SOLVE=${ENABLE_AGE_SOLVE:-false}
export VELOCITY_SOURCE W_FORMULATION ENABLE_AGE_SOLVE

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENT_MODEL=$PARENT_MODEL, REPO_ROOT=$repo_root"
cd "$repo_root"

job_id="${PBS_JOBID:-interactive}"

echo "Creating grid for PARENT_MODEL=$PARENT_MODEL"
grid_log_dir="$repo_root/logs/julia/create_grid"
mkdir -p "$grid_log_dir"
julia --project "$repo_root/src/create_grid.jl" 1> "$grid_log_dir/create_grid_${PARENT_MODEL}_${job_id}.out" 2> "$grid_log_dir/create_grid_${PARENT_MODEL}_${job_id}.err"
echo "Done creating grid for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $grid_log_dir/create_grid_${PARENT_MODEL}_${job_id}.{out,err}"

echo "Running preprocessing (interpolated + mass-transport velocities) for PARENT_MODEL=$PARENT_MODEL"
vel_log_dir="$repo_root/logs/julia/create_velocities"
mkdir -p "$vel_log_dir"
julia --project "$repo_root/src/create_velocities.jl" 1> "$vel_log_dir/create_velocities_${PARENT_MODEL}_${job_id}.out" 2> "$vel_log_dir/create_velocities_${PARENT_MODEL}_${job_id}.err"
echo "Done preprocessing for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $vel_log_dir/create_velocities_${PARENT_MODEL}_${job_id}.{out,err}"

# Submit downstream jobs if requested (only reached when preprocessing succeeded)
if [[ "$SUBMIT_OFFLINE_CPU" == "true" ]]; then
    echo "Submitting offline CPU job (VELOCITY_SOURCE=$VELOCITY_SOURCE, W_FORMULATION=$W_FORMULATION)"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION" \
        "$repo_root/scripts/ACCESS-OM2-1_CPU_job.sh"
fi

if [[ "$SUBMIT_OFFLINE_GPU" == "true" ]]; then
    echo "Submitting offline GPU job (VELOCITY_SOURCE=$VELOCITY_SOURCE, W_FORMULATION=$W_FORMULATION)"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION" \
        "$repo_root/scripts/ACCESS-OM2-1_GPU_job.sh"
fi

if [[ "$SUBMIT_MATRIX" == "true" ]]; then
    echo "Submitting matrix build job (VELOCITY_SOURCE=$VELOCITY_SOURCE, ENABLE_AGE_SOLVE=$ENABLE_AGE_SOLVE)"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",ENABLE_AGE_SOLVE="$ENABLE_AGE_SOLVE" \
        "$repo_root/scripts/ACCESS-OM2-1_matrix_job.sh"
fi
