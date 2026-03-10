#!/bin/bash

#PBS -N OM2-025_1yr
#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l walltime=02:00:00
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

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/standardrun
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_1year_${job_id}.log"

echo "Running src/run_1year.jl for PARENT_MODEL=$PARENT_MODEL"
echo "logging output in $log_file"
julia $JULIA_BOUNDS_FLAG --project src/run_1year.jl &> "$log_file"
echo "Done running src/run_1year.jl for PARENT_MODEL=$PARENT_MODEL"

# Submit CPU plot job after simulation
echo "Submitting plot_1year_age CPU job"
qsub -v PARENT_MODEL="$PARENT_MODEL",VELOCITY_SOURCE="$VELOCITY_SOURCE",W_FORMULATION="$W_FORMULATION",ADVECTION_SCHEME="$ADVECTION_SCHEME",TIMESTEPPER="$TIMESTEPPER" \
    scripts/ACCESS-OM2-025_plot_1year_age_job.sh
