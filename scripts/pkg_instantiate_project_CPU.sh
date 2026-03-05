#!/bin/bash

#PBS -N instantiate_CPU
#PBS -P y99
#PBS -l mem=47GB
#PBS -q express
#PBS -l walltime=01:00:00
#PBS -l ncpus=12
#PBS -l storage=scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

echo "Instantiating packages on compute node on CPU"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo "Done instantiating and precompiling packages on compute node on CPU"
