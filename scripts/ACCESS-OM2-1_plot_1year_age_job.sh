#!/bin/bash

#PBS -N plot_1year_age_OM2-1
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

# Configuration (inherit from parent job or use defaults)
PARENT_MODEL=ACCESS-OM2-1
VELOCITY_SOURCE=${VELOCITY_SOURCE:-cgridtransports}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
ADVECTION_SCHEME=${ADVECTION_SCHEME:-centered2}
export PARENT_MODEL VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME
echo "environment variables set:"
echo "PARENT_MODEL=$PARENT_MODEL"
echo "VELOCITY_SOURCE=$VELOCITY_SOURCE"
echo "W_FORMULATION=$W_FORMULATION"
echo "ADVECTION_SCHEME=$ADVECTION_SCHEME"

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "cd $repo_root"
cd "$repo_root"

# Unlimited stack size
ulimit -s unlimited

echo "Running plot_1year_age.jl on CPU"
log_dir="$repo_root/logs/julia/plot_1year_age"
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"
julia --project "$repo_root/src/plot_1year_age.jl" &> "$log_dir/plot_1year_age_${VELOCITY_SOURCE}_${W_FORMULATION}_${ADVECTION_SCHEME}_${job_id}.log"
echo "Done running plot_1year_age.jl"
