#!/bin/bash

#PBS -N test_mpi
#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuvolta
#PBS -l walltime=00:30:00
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99
#PBS -l jobfs=10GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

module load cuda/12.9.0
module load openmpi/5.0.8
export JULIA_CUDA_USE_COMPAT=false
export LD_LIBRARY_PATH=/apps/openmpi/5.0.8/lib
export JULIA_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export UCX_WARN_UNUSED_ENV_VARS=n

echo "Running MPI test on 4 GPUs"
mpirun -n 4 julia --project src/test_mpi.jl
echo "MPI test completed"
