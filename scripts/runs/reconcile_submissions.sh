#!/usr/bin/env bash
# Fill empty PBS-side columns in scripts/runs/submissions.tsv by querying
# `qstat -fx <jobid>` for each row missing them.
#
# Columns filled (positions 13-20 of the canonical 20-col schema):
#   exit_code, queue, walltime_req, walltime_used, mem_req, mem_used,
#   ncpus, ngpus
#
# Sentinels:
#   ""    pending (queued/held/running) — next reconcile picks it up
#   "DRY" DRY_RUN row (not a real job)
#   "?"   finished but Exit_status missing (e.g. afterok-aborted), or aged out
#   "-"   field unavailable from qstat
#
# Implementation note: this is a thin wrapper around a Python script so we get
# proper TSV parsing (bash `IFS=$'\t' read -a` collapses consecutive tabs and
# loses empty fields).

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

exec python3 scripts/runs/reconcile_submissions.py "$@"
