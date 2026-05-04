# Shared job submission helper for driver.sh and test_driver.sh.
# Source this file after setting MODEL_SHORT, COMMON_VARS, and resource vars.
#
# Usage:
#   JOB_ID=$(submit_job NAME WALLTIME SCRIPT [options...])
#
# Options:
#   --gpu              Full GPU partition (NGPUS GPUs, GPU_NCPUS CPUs, GPU_MEM)
#   --gpu-single       Single GPU (1 GPU, 12 CPUs, GPU_MEM_SINGLE)
#   --cpu              Explicit CPU queue (CPU_NCPUS CPUs, CPU_MEM)
#   --deps DEPS        Colon-separated afterok dependency job IDs (empty = no deps)
#   --vars VARS        Extra -v vars (comma-separated, appended to COMMON_VARS)
#   --queue Q          Override queue
#   --ngpus N          Override ngpus
#   --ncpus N          Override ncpus
#   --mem M            Override mem (e.g., 47GB)
#
# When none of --gpu/--gpu-single/--cpu is specified, no queue/resource flags
# are added; the PBS script's #PBS directives serve as defaults.
#
# DRY_RUN=yes prints the qsub command without executing.
#
# Side effects:
#   Appends to a temp file for tracking (step counter + summary)
#   Echoes "[STEP] NAME: JOB_ID (afterok deps)" to stderr

# Temp file for tracking submitted jobs (persists across subshells).
# Lines are written as: step|jobid|deps|script
# - step:    short step name (e.g., TMbuild, NK_c)
# - jobid:   PBS job id returned by qsub
# - deps:    colon-separated afterok dependency job IDs (may be empty)
# - script:  path to the PBS script that was submitted
_SUBMIT_LOG=$(mktemp)
trap "rm -f $_SUBMIT_LOG" EXIT

# Append-only central index of every submission across all driver invocations.
# One row per PBS job. Header is written on first creation.
SUBMISSIONS_TSV="${SUBMISSIONS_TSV:-scripts/runs/submissions.tsv}"
_ensure_submissions_tsv_header() {
    [ -f "$SUBMISSIONS_TSV" ] && return 0
    mkdir -p "$(dirname "$SUBMISSIONS_TSV")"
    printf 'timestamp\tjobid\tstep\tdeps\tmanifest_path\tcase_file\tgit_commit\tJOB_CHAIN\tPARENT_MODEL\tTIME_WINDOW\tMLD_TIME_WINDOW\tscript\n' > "$SUBMISSIONS_TSV"
}

submit_job() {
    local name="$1" walltime="$2" script="$3"
    shift 3

    # Parse options
    local mode="" deps="" extra_vars=""
    local ovr_queue="" ovr_ngpus="" ovr_ncpus="" ovr_mem=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu)        mode=gpu ;;
            --gpu-single) mode=gpu-single ;;
            --cpu)        mode=cpu ;;
            --deps)       shift; deps="$1" ;;
            --vars)       shift; extra_vars="$1" ;;
            --queue)      shift; ovr_queue="$1" ;;
            --ngpus)      shift; ovr_ngpus="$1" ;;
            --ncpus)      shift; ovr_ncpus="$1" ;;
            --mem)        shift; ovr_mem="$1" ;;
            *) echo "submit_job: unknown option: $1" >&2; return 1 ;;
        esac
        shift
    done

    # Build qsub arguments
    local -a qsub_args=()

    # Job name
    qsub_args+=(-N "${MODEL_SHORT}_${name}")

    # Walltime
    qsub_args+=(-l "walltime=${walltime}")

    # Resource mode (sets defaults; overrides applied below)
    local queue="" ngpus="" ncpus="" mem=""
    case "$mode" in
        gpu)
            queue="$GPU_QUEUE"; ngpus="$NGPUS"; ncpus="$GPU_NCPUS"; mem="$GPU_MEM"
            ;;
        gpu-single)
            queue="$GPU_QUEUE"; ngpus=1; ncpus=12; mem="$GPU_MEM_SINGLE"
            ;;
        cpu)
            queue="$CPU_QUEUE"; ngpus=0; ncpus="$CPU_NCPUS"; mem="$CPU_MEM"
            ;;
    esac

    # Apply overrides
    [ -n "$ovr_queue" ] && queue="$ovr_queue"
    [ -n "$ovr_ngpus" ] && ngpus="$ovr_ngpus"
    [ -n "$ovr_ncpus" ] && ncpus="$ovr_ncpus"
    [ -n "$ovr_mem" ]   && mem="$ovr_mem"

    # Add resource flags (only if set)
    [ -n "$queue" ] && qsub_args+=(-q "$queue")
    [ -n "$ngpus" ] && qsub_args+=(-l "ngpus=${ngpus}")
    [ -n "$ncpus" ] && qsub_args+=(-l "ncpus=${ncpus}")
    [ -n "$mem" ]   && qsub_args+=(-l "mem=${mem}")

    # Dependencies
    if [ -n "$deps" ]; then
        qsub_args+=(-W "depend=afterok:${deps}")
    fi

    # Environment variables
    local vars="$COMMON_VARS"
    [ -n "$extra_vars" ] && vars="${vars},${extra_vars}"
    qsub_args+=(-v "$vars")

    # Script
    qsub_args+=("$script")

    # Submit or dry-run
    local job_id
    if [ "${DRY_RUN:-no}" = "yes" ]; then
        echo "qsub ${qsub_args[*]}" >&2
        # Use line count as unique counter
        local n=$(wc -l < "$_SUBMIT_LOG")
        job_id="DRY_RUN_$((n + 1))"
    else
        job_id=$(qsub "${qsub_args[@]}")
    fi

    # Register in temp file (persists across subshells)
    echo "${name}|${job_id}|${deps}|${script}" >> "$_SUBMIT_LOG"
    local step_num=$(wc -l < "$_SUBMIT_LOG")

    # Log to stderr
    local dep_msg=""
    [ -n "$deps" ] && dep_msg=" (afterok $deps)"
    echo "[$step_num] ${name}: ${job_id}${dep_msg}" >&2

    # Append a row to the central submissions TSV. MANIFEST_PATH is set by
    # driver.sh once known; left empty here is fine — driver.sh fills it in
    # the manifest itself, and the rows can be joined back via timestamp+jobid.
    _ensure_submissions_tsv_header
    local ts="${SUBMIT_TS:-$(date -Iseconds)}"
    local manifest="${MANIFEST_PATH:-}"
    local case_file="${CASE_FILE:-}"
    local git_commit="${GIT_COMMIT:-unknown}"
    local job_chain="${JOB_CHAIN:-}"
    local pm="${PARENT_MODEL:-}"
    local tw="${TIME_WINDOW:-}"
    local mtw=""
    [ "${MLD_EXPLICIT:-no}" = "yes" ] && mtw="${MLD_TIME_WINDOW:-}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ts" "$job_id" "$name" "$deps" "$manifest" "$case_file" \
        "$git_commit" "$job_chain" "$pm" "$tw" "$mtw" "$script" \
        >> "$SUBMISSIONS_TSV"

    # Return job ID on stdout
    echo "$job_id"
}

# Print summary of all submitted jobs
print_summary() {
    local label="${1:-Pipeline}"
    local count=$(wc -l < "$_SUBMIT_LOG")
    echo "" >&2
    echo "=== $count jobs submitted for ${label} ===" >&2
    echo "" >&2
    if [ "$count" -eq 0 ]; then
        echo "  (no jobs submitted)" >&2
    else
        while IFS='|' read -r name job_id deps script; do
            printf "  %-25s %s\n" "$name" "$job_id" >&2
        done < "$_SUBMIT_LOG"
    fi
}
