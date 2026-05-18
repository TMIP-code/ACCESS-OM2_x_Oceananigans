#!/bin/bash

#PBS -P y99
#PBS -l mem=16GB
#PBS -q express
#PBS -l ngpus=0
#PBS -l ncpus=1
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Wrapper for scripts/analysis/nsys_per_rank_busy.py — runs nsys
# export + SQLite-based per-rank GPU-busy summary on a compute node
# (login nodes get killed by nsys export memory pressure).
#
# Required env: NSYS_GLOB — shell glob matching the rank files.
# Optional env:  OUT_CSV   — CSV output path.
#
# Submit:
#   NSYS_GLOB='logs/julia/ACCESS-OM2-01/.../*168522455*_rank*.nsys-rep' \
#       qsub -v NSYS_GLOB scripts/tests/run_nsys_per_rank_busy.sh
#   # with CSV:
#   NSYS_GLOB='...' OUT_CSV=/scratch/y99/bp3051/per_rank.csv \
#       qsub -v NSYS_GLOB,OUT_CSV scripts/tests/run_nsys_per_rank_busy.sh

set -euo pipefail

: "${NSYS_GLOB:?Must set NSYS_GLOB to a shell glob matching nsys-rep files}"

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

module load cuda/12.9.0

MYSCRATCH=/scratch/y99/bp3051
export TMPDIR="$MYSCRATCH/tmp"
mkdir -p "$TMPDIR"

job_id="${PBS_JOBID:-interactive}"
log_dir="logs/analysis"
mkdir -p "$log_dir"
log_file="$log_dir/nsys_per_rank_busy_${job_id}.log"

echo "Running scripts/analysis/nsys_per_rank_busy.py"
echo "  NSYS_GLOB=$NSYS_GLOB"
echo "  OUT_CSV=${OUT_CSV:-(not set)}"
echo "  log_file=$log_file"
echo

csv_arg=()
[ -n "${OUT_CSV:-}" ] && csv_arg=(--csv "$OUT_CSV")

# `$NSYS_GLOB` must be unquoted so the shell expands the glob.
python3 scripts/analysis/nsys_per_rank_busy.py $NSYS_GLOB "${csv_arg[@]}" 2>&1 | tee "$log_file"
echo "Done"
