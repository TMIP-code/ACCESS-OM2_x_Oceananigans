#!/bin/bash

#PBS -N instantiate_CPU
#PBS -P y99
#PBS -l mem=190GB
#PBS -q express
#PBS -l walltime=01:00:00
#PBS -l ncpus=48
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99
#PBS -l jobfs=4GB
#PBS -o scratch_output/PBS/
#PBS -e scratch_output/PBS/
#PBS -l wd

# locate repo root by walking up to the directory named ACCESS-OM2_x_Oceananigans
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
echo "REPO_ROOT=$repo_root"

echo "Instantiating packages on compute node on CPU"
julia --project -e 'using Pkg; Pkg.instantiate()'
echo "Done instantiating packages on compute node on CPU"
