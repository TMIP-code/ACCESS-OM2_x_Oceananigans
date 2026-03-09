#!/bin/bash

#PBS -N mpi_precompile_CPU
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

module load openmpi/5.0.8
export LD_LIBRARY_PATH=/apps/openmpi/5.0.8/lib

echo "Precompiling packages on CPU with MPI support"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo "Done precompiling packages on CPU"
