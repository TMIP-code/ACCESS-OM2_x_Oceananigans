#!/bin/bash
#
# PBS wrapper for src/build_atlantic_mask_rhospace.jl — builds the 2D
# Atlantic basin mask on the (grid_xt_ocean, grid_yu_ocean) axes used by the
# density-space MOC NetCDFs. Reads coordinates from psi_tot_global.nc, writes
# atlantic_mask.nc alongside it.
#
# Usage:
#   qsub -v "PARENT_MODEL=ACCESS-OM2-1"   scripts/prepreprocessing/build_atlantic_mask_rhospace.sh
#   qsub -v "PARENT_MODEL=ACCESS-OM2-025" scripts/prepreprocessing/build_atlantic_mask_rhospace.sh
#
# Writes:  /scratch/y99/TMIP/data/{PARENT_MODEL}/{EXPERIMENT}/rhospace/atlantic_mask.nc

#PBS -P y99
#PBS -q express
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l walltime=00:10:00
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l jobfs=2GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

job_id="${PBS_JOBID:-interactive}"

log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/preprocess
mkdir -p "$log_dir"

echo "Building Atlantic rhospace mask for PARENT_MODEL=$PARENT_MODEL EXPERIMENT=$EXPERIMENT"
julia --project src/build_atlantic_mask_rhospace.jl \
    &> "$log_dir/build_atlantic_mask_rhospace_${job_id}.log"
echo "Done; log: $log_dir/build_atlantic_mask_rhospace_${job_id}.log"
