#!/bin/bash

#PBS -N run_OM2-1_CPU
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=01:00:00
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
MODEL_CONFIG="${VELOCITY_SOURCE}_${W_FORMULATION}"
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "Sourced: PARENT_MODEL=$PARENT_MODEL, REPO_ROOT=$repo_root"
cd "$repo_root"

# echo "Create grid on CPU with PARENT_MODEL=$PARENT_MODEL"
# source $repo_root/scripts/create_grid.sh $PARENT_MODEL
# echo "Done creating grid on CPU with PARENT_MODEL=$PARENT_MODEL"

echo "Creating velocities on CPU for PARENT_MODEL=$PARENT_MODEL"
run_log_dir="$repo_root/logs/runs/$MODEL_CONFIG"
mkdir -p "$run_log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project "$repo_root/src/create_velocities.jl" &> "$run_log_dir/create_velocities.$job_id.out"
echo "Done creating velocities on CPU for PARENT_MODEL=$PARENT_MODEL"

# echo "Creating transport-derived velocities for PARENT_MODEL=$PARENT_MODEL"
# julia --project $repo_root/src/create_velocities_from_transports.jl
# echo "Done creating transport-derived velocities for PARENT_MODEL=$PARENT_MODEL"

# echo "Create closures on CPU with PARENT_MODEL=$PARENT_MODEL"
# source $repo_root/scripts/create_closures.sh $PARENT_MODEL
# echo "Done creating closures on CPU with PARENT_MODEL=$PARENT_MODEL"
