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

source scripts/env_defaults.sh

echo "Running MPI test on 4 GPUs"
mpirun -n 4 julia --project src/test_mpi.jl
echo "MPI test completed"
