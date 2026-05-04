#!/usr/bin/env bash
# Fill empty PBS-side columns in scripts/runs/submissions.tsv by querying
# `qstat -fx <jobid>` for each row missing them.
#
# Columns filled (positions 13-20 of the canonical 20-col schema):
#   exit_code, queue, walltime_req, walltime_used, mem_req, mem_used,
#   ncpus, ngpus
#
# Behaviour:
# - DRY_RUN_* jobs       → exit_code="DRY", others "-"
# - Jobs still Q/H/R     → leave empty (next reconcile picks them up)
# - Jobs in qstat (state=F) → fill all PBS-side fields
# - Jobs aged out / not found → exit_code="?", others "?" (unless previous
#                              row already had values, in which case keep them)
#
# Tolerates older row formats (12, 13, or 14 fields) by treating missing
# trailing fields as empty. Always emits canonical 20-col output.

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

HEADER=$'timestamp\tjobid\tstep\tdeps\tmanifest_path\tcase_file\tgit_commit\tJOB_CHAIN\tPARENT_MODEL\tTIME_WINDOW\tMLD_TIME_WINDOW\tscript\texit_code\tqueue\twalltime_req\twalltime_used\tmem_req\tmem_used\tncpus\tngpus'

emit_row() {
    # 20 fields, tab-separated.
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$tmp"
}

while IFS=$'\t' read -r -a F || [ ${#F[@]} -gt 0 ]; do
    # Skip blank trailing lines.
    [ ${#F[@]} -eq 0 ] && continue

    # Header row.
    if [ "${F[0]}" = "timestamp" ]; then
        echo "$HEADER" >> "$tmp"
        continue
    fi

    # First 12 fields are the stable submission metadata. Use :- for safety
    # against malformed rows (set -u would otherwise abort).
    ts="${F[0]:-}" jobid="${F[1]:-}" step="${F[2]:-}" deps="${F[3]:-}"
    manifest="${F[4]:-}" case_file="${F[5]:-}" gc="${F[6]:-}" jc="${F[7]:-}"
    pm="${F[8]:-}" tw="${F[9]:-}" mtw="${F[10]:-}" script="${F[11]:-}"

    # Trailing PBS fields — read whatever is present, default to empty.
    exit_code="${F[12]:-}" queue="${F[13]:-}" walltime_req="${F[14]:-}"
    walltime_used="${F[15]:-}" mem_req="${F[16]:-}" mem_used="${F[17]:-}"
    ncpus="${F[18]:-}" ngpus="${F[19]:-}"

    # Compatibility shim: older rows had only (exit_code, walltime_used) at
    # positions 13-14. If the field at position 14 looks like a walltime
    # (HH:MM:SS), shift it into walltime_used and clear queue.
    if [ ${#F[@]} -le 14 ] && [[ "$queue" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
        walltime_used="$queue"
        queue=""
    fi

    # Dry runs.
    if [[ "$jobid" == DRY_RUN_* ]]; then
        n_dry=$((n_dry + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" \
                 "DRY" "-" "-" "-" "-" "-" "-" "-"
        continue
    fi

    # Already fully filled (real numeric exit_code AND ncpus populated).
    if [ -n "$exit_code" ] && [ "$exit_code" != "?" ] && [ -n "$ncpus" ] && [ "$ncpus" != "?" ]; then
        n_kept=$((n_kept + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" \
                 "$exit_code" "$queue" "$walltime_req" "$walltime_used" "$mem_req" "$mem_used" "$ncpus" "$ngpus"
        continue
    fi

    # Query qstat history.
    info=$(qstat -fx "$jobid" 2>/dev/null || true)
    if [ -z "$info" ]; then
        # Not in qstat. Preserve any existing values; mark missing as '?'.
        n_unknown=$((n_unknown + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" \
                 "${exit_code:-?}" "${queue:-?}" "${walltime_req:-?}" "${walltime_used:-?}" \
                 "${mem_req:-?}" "${mem_used:-?}" "${ncpus:-?}" "${ngpus:-?}"
        continue
    fi

    state=$(echo "$info" | grep -oP 'job_state = \K\S+' | head -1 || true)
    if [ "$state" != "F" ]; then
        # Still queued/held/running — leave PBS fields empty.
        n_pending=$((n_pending + 1))
        emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" \
                 "" "" "" "" "" "" "" ""
        continue
    fi

    # Fully filled fields from qstat -fx.
    ec=$(echo "$info" | grep -oP 'Exit_status = \K-?\d+' | head -1 || true)
    q=$(echo "$info" | grep -oP '^\s*queue = \K\S+' | head -1 || true)
    wreq=$(echo "$info" | grep -oP 'Resource_List\.walltime = \K\S+' | head -1 || true)
    wuse=$(echo "$info" | grep -oP 'resources_used\.walltime = \K\S+' | head -1 || true)
    mreq=$(echo "$info" | grep -oP 'Resource_List\.mem = \K\S+' | head -1 || true)
    muse=$(echo "$info" | grep -oP 'resources_used\.mem = \K\S+' | head -1 || true)
    ncreq=$(echo "$info" | grep -oP 'Resource_List\.ncpus = \K\S+' | head -1 || true)
    ngreq=$(echo "$info" | grep -oP 'Resource_List\.ngpus = \K\S+' | head -1 || true)

    n_filled=$((n_filled + 1))
    emit_row "$ts" "$jobid" "$step" "$deps" "$manifest" "$case_file" "$gc" "$jc" "$pm" "$tw" "$mtw" "$script" \
             "${ec:-?}" "${q:--}" "${wreq:--}" "${wuse:--}" \
             "${mreq:--}" "${muse:--}" "${ncreq:--}" "${ngreq:-0}"
done < "$TSV"

mv "$tmp" "$TSV"
trap - EXIT

echo "reconcile: filled=$n_filled  pending=$n_pending  dry=$n_dry  already=$n_kept  unknown=$n_unknown"
echo "TSV: $TSV"
