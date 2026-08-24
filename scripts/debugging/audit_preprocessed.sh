#!/bin/bash

#PBS -P y99
#PBS -l mem=32GB
#PBS -q normal
#PBS -l ncpus=4
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99

#PBS -l jobfs=4GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

# Read-only audit of preprocessed NetCDF outputs for the missing-dask-chunk
# corruption (docs/periodicaverage_corruption_bug.md). Streams one 2D slice at a
# time, so memory is bounded regardless of the 70 GB per-variable file sizes;
# the cost is I/O, not RAM.
#
# Submit over everything:
#   qsub scripts/debugging/audit_preprocessed.sh
#
# Or scope it to one experiment / check every level / skip archived copies:
#   AUDIT_ROOT=preprocessed_inputs/ACCESS-OM2-01 AUDIT_FULL=yes \
#   AUDIT_SKIP=nc_archive \
#     qsub -v AUDIT_ROOT,AUDIT_FULL,AUDIT_SKIP scripts/debugging/audit_preprocessed.sh

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root

AUDIT_ROOT=${AUDIT_ROOT:-preprocessed_inputs}
AUDIT_FULL=${AUDIT_FULL:-no}
AUDIT_SKIP=${AUDIT_SKIP:-}
extra=""
if [ "${AUDIT_FULL}" = "yes" ]; then extra="--full"; fi
if [ -n "${AUDIT_SKIP}" ]; then extra="$extra --skip ${AUDIT_SKIP}"; fi

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3-26.07

log_dir=logs/python/audit
mkdir -p "$log_dir"
job_id="${PBS_JOBID:-interactive}"

echo "Auditing $AUDIT_ROOT (full=$AUDIT_FULL)"
# `set -e` would abort before we could report; capture the status explicitly.
status=0
python3 src/audit_preprocessed.py "$AUDIT_ROOT" $extra \
    &> "$log_dir/audit_${job_id}.log" || status=$?
echo "Done (exit $status). Log: $log_dir/audit_${job_id}.log"
exit $status
