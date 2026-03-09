#!/bin/bash

#PBS -N mpi_precompile_GPU
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

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

module load cuda/12.9.0
module load openmpi/5.0.8
export JULIA_CUDA_USE_COMPAT=false
export LD_LIBRARY_PATH=/apps/openmpi/5.0.8/lib

echo "Precompiling packages on GPU with MPI support"
julia --project -e 'using Pkg; Pkg.instantiate()'
echo "Done precompiling packages on GPU"
