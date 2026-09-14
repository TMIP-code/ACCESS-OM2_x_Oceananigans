#!/bin/bash

#PBS -N instantiate_GPU
#PBS -P y99
#PBS -l mem=96GB
#PBS -q gpuvolta
#PBS -l walltime=00:15:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

# Package precompilation is model-independent, but env_defaults.sh requires
# PARENT_MODEL (plain `qsub` from pkg_update_project.sh passes none).
export PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
source scripts/env_defaults.sh

echo "Instantiating packages on compute node on GPU"
julia --project -e 'using Pkg; Pkg.instantiate()'
echo "Done instantiating packages on compute node on GPU"
