#!/bin/bash

#PBS -P y99
#PBS -l mem=190GB
#PBS -q express
#PBS -l ncpus=48
#PBS -l storage=scratch/y99+gdata/y99
#PBS -l walltime=01:00:00
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

mkdir -p logs/PBS
log_file="logs/PBS/count_wet_cells_${PBS_JOBID:-interactive}.log"

echo "Counting wet cells (TM dimension) per parent model..."
julia --project scripts/maintenance/count_wet_cells.jl 2>&1 | tee "$log_file"
echo "Done. Logged to $log_file"
