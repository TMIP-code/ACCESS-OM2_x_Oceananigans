#!/bin/bash

#PBS -N OM2-1_preproc
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
TIMESTEPPER=${TIMESTEPPER:-AB2}
export VELOCITY_SOURCE W_FORMULATION TIMESTEPPER

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "PARENT_MODEL=$PARENT_MODEL, repo_root=$repo_root"
cd $repo_root

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/julia/preprocess
mkdir -p "$log_dir"

echo "Creating grid for PARENT_MODEL=$PARENT_MODEL"
julia --project src/create_grid.jl &> "$log_dir/create_grid_${PARENT_MODEL}_${job_id}.log"
echo "Done creating grid for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/create_grid_${PARENT_MODEL}_${job_id}.log"

echo "Running preprocessing (interpolated + mass-transport velocities) for PARENT_MODEL=$PARENT_MODEL"
julia --project src/create_velocities.jl &> "$log_dir/create_velocities_${PARENT_MODEL}_${job_id}.log"
echo "Done preprocessing for PARENT_MODEL=$PARENT_MODEL"
echo "logged output in $log_dir/create_velocities_${PARENT_MODEL}_${job_id}.log"

# Submit downstream jobs if requested (only reached when preprocessing succeeded)
if [[ "$SUBMIT_OFFLINE_CPU" == "true" ]]; then
    echo "Submitting offline CPU job (VELOCITY_SOURCE=$VELOCITY_SOURCE, W_FORMULATION=$W_FORMULATION, TIMESTEPPER=$TIMESTEPPER)"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_CPU_job.sh
fi

if [[ "$SUBMIT_OFFLINE_GPU" == "true" ]]; then
    echo "Submitting offline GPU job (VELOCITY_SOURCE=$VELOCITY_SOURCE, W_FORMULATION=$W_FORMULATION, TIMESTEPPER=$TIMESTEPPER)"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_GPU_job.sh
fi

if [[ "$SUBMIT_MATRIX" == "true" ]]; then
    echo "Submitting matrix build job (VELOCITY_SOURCE=$VELOCITY_SOURCE, TIMESTEPPER=$TIMESTEPPER)"
    qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",TIMESTEPPER="$TIMESTEPPER" \
        scripts/ACCESS-OM2-1_matrix_job.sh
fi
