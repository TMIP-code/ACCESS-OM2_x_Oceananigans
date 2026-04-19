#!/bin/bash

#PBS -N compare_linphi
#PBS -P y99
#PBS -l mem=4GB
#PBS -q normal
#PBS -l walltime=00:05:00
#PBS -l ncpus=1
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

TEST_DIR="${repo_root}/outputs/${PARENT_MODEL}/test_nolinphi"

job_id="${PBS_JOBID:-interactive}"
log_dir=logs/julia/test
mkdir -p "$log_dir"
log_file="$log_dir/compare_${job_id}.log"
echo "logging output in $log_file"

julia --project test/test_remove_linphi.jl "$TEST_DIR" &> "$log_file"
echo "Done — compare"
