#!/usr/bin/env bash
# Fill empty PBS-side columns in scripts/runs/submissions.tsv by parsing
# PBS resource logs from logs/PBS/<jobid>.gadi-pbs.OU/ER for each row missing them.
#
# Columns filled (positions 13-21 of the canonical 21-col schema):
#   exit_code, queue, walltime_req, walltime_used, mem_req_GB, mem_used_GB,
#   ncpus, ngpus, service_units
#
# Sentinels:
#   ""    pending (queued/held/running) — PBS log not yet available
#   "DRY" DRY_RUN row (not a real job)
#   "?"   PBS log not found (job completed but log not recorded)
#   "-"   field unavailable from PBS log
#
# Implementation note: this is a thin wrapper around a Python script so we get
# proper TSV parsing (bash `IFS=$'\t' read -a` collapses consecutive tabs and
# loses empty fields).

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

exec python3 scripts/runs/reconcile_submissions.py "$@"
