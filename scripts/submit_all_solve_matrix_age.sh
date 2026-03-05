#!/usr/bin/env bash
set -euo pipefail

# Submit all LINEAR_SOLVER × LUMP_AND_SPRAY × MATRIX_PROCESSING combinations
# for solve_matrix_age.
#
# MATRIX_PROCESSING variants are only submitted for Pardiso (the only solver
# that uses matrix_type); ParU and UMFPACK use raw only.
#
# Total: Pardiso (1 × 2 × 4 = 8) + ParU (1 × 2 × 1 = 2) + UMFPACK (1 × 2 × 1 = 2) = 12 jobs
#
# 60s delay between submissions to avoid concurrent precompilation OOM.
#
# Optional env vars forwarded to all jobs:
#   VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, CHECK_BOUNDS

DELAY=60  # seconds between submissions

# Forward optional env vars to all jobs
EXTRA_VARS=""
for var in VELOCITY_SOURCE W_FORMULATION ADVECTION_SCHEME TIMESTEPPER CHECK_BOUNDS; do
    val="${!var:-}"
    [ -n "$val" ] && EXTRA_VARS="${EXTRA_VARS},${var}=${val}"
done
[ -n "$EXTRA_VARS" ] && echo "Forwarding:$EXTRA_VARS"

count=0

submit() {
    local ls="$1" las="$2" mp="$3"
    qsub -v LINEAR_SOLVER=${ls},LUMP_AND_SPRAY=${las},MATRIX_PROCESSING=${mp}${EXTRA_VARS} \
        scripts/ACCESS-OM2-1_solve_matrix_age_job.sh
    count=$((count + 1))
    local tag="${ls}/${las}/${mp}"
    echo "[$count] Submitted $tag."
    if [ "$count" -lt 12 ]; then
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

echo "Submitted ${count} jobs."
