#!/usr/bin/env bash
# Fill empty exit_code / walltime_used columns in scripts/runs/submissions.tsv
# by querying `qstat -fx <jobid>` for each row missing them.
#
# Usage:
#   bash scripts/runs/reconcile_submissions.sh
#
# Behaviour:
# - DRY_RUN_* jobs       → exit_code="DRY",   walltime_used="-"
# - Jobs still Q/H/R     → leave empty (next reconcile picks them up)
# - Jobs in qstat history (state=F) → fill exit_code + walltime_used
# - Jobs aged out / not found → exit_code="?", walltime_used="?"
#
# Rewrites the TSV in place via tmpfile + mv. Tolerates rows from older
# 12-column format (pre-exit_code) by treating missing trailing fields
# as empty, then filling them.

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

TSV="${SUBMISSIONS_TSV:-scripts/runs/submissions.tsv}"

if [ ! -f "$TSV" ]; then
    echo "TSV not found: $TSV" >&2
    exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

n_filled=0
n_dry=0
n_kept=0
n_pending=0
n_unknown=0

# Header line constant (always emitted, replaces older 12-column header)
HEADER=$'timestamp\tjobid\tstep\tdeps\tmanifest_path\tcase_file\tgit_commit\tJOB_CHAIN\tPARENT_MODEL\tTIME_WINDOW\tMLD_TIME_WINDOW\tscript\texit_code\twalltime_used'

emit_row() {
    # 14 fields, tab-separated, in TSV order.
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$tmp"
}

while IFS=$'\t' read -r ts jobid step deps manifest case_file gc jc pm tw mtw script exit_code walltime_used; do
    # Header line: rewrite as canonical 14-column header.
    if [ "$ts" = "timestamp" ]; then
        echo "$HEADER" >> "$tmp"
        continue
    fi

    # Dry-run jobs: mark and pass through.
    if [[ "$jobid" == DRY_RUN_* ]]; then
        n_dry=$((n_dry + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" "DRY" "-"
        continue
    fi

    # Already filled (non-empty exit_code that isn't a sentinel placeholder).
    if [ -n "$exit_code" ] && [ "$exit_code" != "?" ]; then
        n_kept=$((n_kept + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" "$exit_code" "$walltime_used"
        continue
    fi

    # Query qstat history.
    info=$(qstat -fx "$jobid" 2>/dev/null || true)
    if [ -z "$info" ]; then
        # Aged out of qstat or invalid id.
        n_unknown=$((n_unknown + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" "?" "?"
        continue
    fi

    state=$(echo "$info" | grep -oP 'job_state = \K\S+' | head -1 || true)
    if [ "$state" != "F" ]; then
        # Still queued/held/running — leave empty for next reconcile.
        n_pending=$((n_pending + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" "" ""
        continue
    fi

    ec=$(echo "$info" | grep -oP 'Exit_status = \K-?\d+' | head -1 || true)
    wt=$(echo "$info" | grep -oP 'resources_used\.walltime = \K\S+' | head -1 || true)
    n_filled=$((n_filled + 1))
    emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" "${ec:-?}" "${wt:--}"
done < "$TSV"

mv "$tmp" "$TSV"
trap - EXIT

echo "reconcile: filled=$n_filled  pending=$n_pending  dry=$n_dry  already=$n_kept  unknown=$n_unknown"
echo "TSV: $TSV"
