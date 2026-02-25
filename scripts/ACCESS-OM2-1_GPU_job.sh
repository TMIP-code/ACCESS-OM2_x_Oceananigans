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
#PBS -o scratch_output/PBS/
#PBS -e scratch_output/PBS/
#PBS -l wd

set -euo pipefail

# parent model (falls back to existing env or sensible default)
PARENT_MODEL=ACCESS-OM2-1
VELOCITY_SOURCE=${VELOCITY_SOURCE:-bgridvelocities}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
MODEL_CONFIG="${VELOCITY_SOURCE}_${W_FORMULATION}"
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION
echo "environment variables set:"
echo "PARENT_MODEL=$PARENT_MODEL"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "cd $repo_root"
cd "$repo_root"

echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

echo "Running offline ACCESS-OM2 for PARENT_MODEL=$PARENT_MODEL"
run_log_dir="$repo_root/scratch_output/runs/$MODEL_CONFIG"
echo "logging output in $run_log_dir"
mkdir -p "$run_log_dir"
julia --project "$repo_root/src/offline_ACCESS-OM2.jl" 1> "$run_log_dir/$PBS_JOBID.out" 2> "$run_log_dir/$PBS_JOBID.err"
echo "Done running offline ACCESS-OM2 for PARENT_MODEL=$PARENT_MODEL"

