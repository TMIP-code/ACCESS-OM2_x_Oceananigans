#!/usr/bin/env bash
set -euo pipefail

# Submit all LINEAR_SOLVER × LUMP_AND_SPRAY × MATRIX_PROCESSING combinations
# for solve_matrix_age.
#
# MATRIX_PROCESSING variants are only submitted for Pardiso (the only solver
# that uses matrix_type); ParU and UMFPACK use raw only.
#
# Total: Pardiso (1 × 2 × 4 = 8) + ParU (1 × 2 × 1 = 2) + UMFPACK (1 × 2 × 1 = 2) + CUDSS (1 × 2 × 1 = 2) = 14 jobs
#
# 1s delay between submissions to avoid concurrent precompilation OOM.
#
# Optional env vars forwarded to all jobs:
#   VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, CHECK_BOUNDS
#
# GPU queue selection (default: gpuhopper):
#   GPU_QUEUE=gpuvolta bash scripts/submit_all_solve_matrix_age.sh

DELAY=1  # seconds between submissions
GPU_QUEUE=${GPU_QUEUE:-gpuhopper}

# Auto-set GPU memory based on queue
case "$GPU_QUEUE" in
    gpuvolta)  GPU_MEM=96GB ;;
    gpuhopper) GPU_MEM=256GB ;;
    *)         echo "Unknown GPU_QUEUE=$GPU_QUEUE"; exit 1 ;;
esac

echo "GPU_QUEUE=$GPU_QUEUE (mem=$GPU_MEM)"

# Forward optional env vars to all jobs
EXTRA_VARS=""
for var in VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER CHECK_BOUNDS TM_SOURCE; do
    val="${!var:-}"
    [ -n "$val" ] && EXTRA_VARS="${EXTRA_VARS},${var}=${val}"
done
[ -n "$EXTRA_VARS" ] && echo "Forwarding:$EXTRA_VARS"

TOTAL_JOBS=14
count=0

submit() {
    local ls="$1" las="$2" mp="$3"
    # Abbreviate for PBS job name: LS=Pa/PU/UM/CU, coarse=c/f, MP=r/sf/dz/sd
    local ls_abbr mp_abbr c_abbr
    case "$ls" in Pardiso) ls_abbr=Pa;; ParU) ls_abbr=PU;; UMFPACK) ls_abbr=UM;; CUDSS) ls_abbr=CU;; esac
    case "$mp" in raw) mp_abbr=r;; symfill) mp_abbr=sf;; dropzeros) mp_abbr=dz;; symdrop) mp_abbr=sd;; esac
    [ "$las" = "yes" ] && c_abbr=c || c_abbr=f
    # CUDSS uses the GPU job script; all others use the CPU job script
    if [ "$ls" = "CUDSS" ]; then
        local job_script=scripts/ACCESS-OM2-1_solve_matrix_age_GPU_job.sh
        qsub -N "OM2-1_TM_${c_abbr}_${ls_abbr}_${mp_abbr}" \
            -q $GPU_QUEUE -l mem=$GPU_MEM \
            -v LINEAR_SOLVER=${ls},LUMP_AND_SPRAY=${las},MATRIX_PROCESSING=${mp}${EXTRA_VARS} \
            "$job_script"
    else
        local job_script=scripts/ACCESS-OM2-1_solve_matrix_age_job.sh
        qsub -N "OM2-1_TM_${c_abbr}_${ls_abbr}_${mp_abbr}" \
            -v LINEAR_SOLVER=${ls},LUMP_AND_SPRAY=${las},MATRIX_PROCESSING=${mp}${EXTRA_VARS} \
            "$job_script"
    fi
    count=$((count + 1))
    local tag="${ls}/${las}/${mp}"
    echo "[$count] Submitted $tag."
    if [ "$count" -lt "$TOTAL_JOBS" ]; then
        echo "Waiting ${DELAY}s..."
        sleep $DELAY
    fi
}

# Pardiso — all MATRIX_PROCESSING variants
for las in no yes; do
    for mp in raw symfill dropzeros symdrop; do
        submit Pardiso "$las" "$mp"
    done
done

# ParU — raw only
for las in no yes; do
    submit ParU "$las" raw
done

# UMFPACK — raw only
for las in no yes; do
    submit UMFPACK "$las" raw
done

# CUDSS (GPU) — raw only
for las in no yes; do
    submit CUDSS "$las" raw
done

echo "Submitted ${count} jobs."
