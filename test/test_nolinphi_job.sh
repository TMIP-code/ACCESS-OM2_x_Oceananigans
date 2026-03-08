#!/bin/bash

#PBS -N test_nolinphi
#PBS -P y99
#PBS -l mem=96GB
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

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

echo "Loading CUDA module"
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false

export TEST_OUTPUT_DIR="${repo_root}/outputs/${PARENT_MODEL}/test_nolinphi"
mkdir -p "$TEST_OUTPUT_DIR"

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/test
mkdir -p "$log_dir"
log_file="$log_dir/test_nolinphi_${job_id}.log"
echo "logging output in $log_file"

julia --project test/test_nolinphi.jl &> "$log_file"
echo "Done — test_nolinphi"
