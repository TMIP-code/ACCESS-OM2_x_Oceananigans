#!/bin/bash

#PBS -N debug_tendency
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=00:30:00
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

PARENT_MODEL=ACCESS-OM2-1
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
ADVECTION_SCHEME=${ADVECTION_SCHEME:-weno3}
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME
echo "PARENT_MODEL=$PARENT_MODEL"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"
echo "ADVECTION_SCHEME=$ADVECTION_SCHEME"

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

run_log_dir="$repo_root/logs/julia/debug"
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
echo "Running debug_tendency_budget.jl on CPU"
julia --project --check-bounds=yes "$repo_root/src/debug_tendency_budget.jl" &> "$run_log_dir/debug_tendency_${VELOCITY_SOURCE}_${W_FORMULATION}_${ADVECTION_SCHEME}_${job_id}.log"
echo "Done"
