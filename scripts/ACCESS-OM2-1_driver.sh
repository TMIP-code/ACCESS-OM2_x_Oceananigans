#!/usr/bin/env bash
set -euo pipefail

# Driver script: submits the ACCESS-OM2-1 pipeline as chained PBS jobs.
# Run from the login node (not as a PBS job).
#
# Usage:
#   bash scripts/ACCESS-OM2-1_driver.sh                          # full pipeline (11 jobs)
#   JOB_CHAIN=TM-TMage-NK bash scripts/ACCESS-OM2-1_driver.sh   # restart from TM
#   PREPROCESS_ARCH=GPU bash scripts/ACCESS-OM2-1_driver.sh      # velocities on GPU
#
# Dependency DAG:
#   grid → vel → 1year → snapM+avgM (TM_SOURCE=avg24) → TMage(avg24) + NK(avg24)
#            └→ constM (TM_SOURCE=const)               → TMage(const) + NK(const)

PARENT_MODEL=ACCESS-OM2-1
PREFIX=ACCESS-OM2-1

# Pipeline configuration
JOB_CHAIN=${JOB_CHAIN:-grid-vel-1year-TM-TMage-NK}
PREPROCESS_ARCH=${PREPROCESS_ARCH:-CPU}

# Solver configuration (shared by TMage and NK)
JVP_METHOD=exact
LINEAR_SOLVER=Pardiso
LUMP_AND_SPRAY=yes
INITIAL_AGE=0

echo "=== ${PARENT_MODEL} pipeline driver ==="
echo "JOB_CHAIN=$JOB_CHAIN"
echo "PREPROCESS_ARCH=$PREPROCESS_ARCH"
echo "JVP_METHOD=$JVP_METHOD, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, INITIAL_AGE=$INITIAL_AGE"

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

STEP=0
LAST_DEP=""
GRID_JOB="" VEL_JOB="" TEST_JOB="" CONSTM_JOB="" AVGM_JOB=""
TMAGE_CONST_CPU="" TMAGE_CONST_GPU="" TMAGE_AVG_CPU="" TMAGE_AVG_GPU=""
NK_CONST="" NK_AVG=""

# --- grid ---
if has_step grid; then
    STEP=$((STEP + 1))
    GRID_JOB=$(qsub scripts/${PREFIX}_grid_job.sh)
    echo "[$STEP] Grid: $GRID_JOB"
    LAST_DEP=$GRID_JOB
fi

# --- vel ---
if has_step vel; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$LAST_DEP" ] && dep_flag=(-W "depend=afterok:${LAST_DEP}")
    gpu_flags=()
    if [ "$PREPROCESS_ARCH" = "GPU" ]; then
        gpu_flags=(-q gpuvolta -l ngpus=1 -l ncpus=12 -l mem=96GB)
    fi
    VEL_JOB=$(qsub "${dep_flag[@]}" "${gpu_flags[@]}" scripts/${PREFIX}_vel_job.sh)
    echo "[$STEP] Velocities: $VEL_JOB${LAST_DEP:+ (afterok $LAST_DEP)}${PREPROCESS_ARCH:+ [$PREPROCESS_ARCH]}"
    LAST_DEP=$VEL_JOB
fi

VEL_DEP=$LAST_DEP  # save for branches

# --- 1year test run ---
if has_step 1year; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    TEST_JOB=$(qsub "${dep_flag[@]}" \
        -v NONLINEAR_SOLVER=1year \
        scripts/${PREFIX}_GPU_job.sh)
    echo "[$STEP] 1-year test: $TEST_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# --- TM (submits 2 jobs: constM after vel, snapM+avgM after 1year) ---
if has_step TM; then
    # Constant-field matrix (depends on vel)
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    CONSTM_JOB=$(qsub "${dep_flag[@]}" \
        scripts/${PREFIX}_matrix_job.sh)
    echo "[$STEP] Constant TM: $CONSTM_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"

    # Snapshot + average matrices (depends on 1year)
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${TEST_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TEST_JOB}")
    AVGM_JOB=$(qsub "${dep_flag[@]}" \
        scripts/${PREFIX}_TM_job.sh)
    echo "[$STEP] Snapshot+avg TM: $AVGM_JOB${TEST_JOB:+ (afterok $TEST_JOB)}"
fi

# --- TMage (4 jobs: Pardiso+CUDSS for each TM_SOURCE) ---
if has_step TMage; then
    # const branch
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${CONSTM_JOB:-}" ] && dep_flag=(-W "depend=afterok:${CONSTM_JOB}")
    TMAGE_CONST_CPU=$(qsub "${dep_flag[@]}" \
        -v TM_SOURCE=const,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/${PREFIX}_solve_matrix_age_job.sh)
    echo "[$STEP] TMage const/Pardiso: $TMAGE_CONST_CPU${CONSTM_JOB:+ (afterok $CONSTM_JOB)}"

    STEP=$((STEP + 1))
    TMAGE_CONST_GPU=$(qsub "${dep_flag[@]}" \
        -v TM_SOURCE=const,LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/${PREFIX}_solve_matrix_age_GPU_job.sh)
    echo "[$STEP] TMage const/CUDSS: $TMAGE_CONST_GPU${CONSTM_JOB:+ (afterok $CONSTM_JOB)}"

    # avg24 branch
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${AVGM_JOB:-}" ] && dep_flag=(-W "depend=afterok:${AVGM_JOB}")
    TMAGE_AVG_CPU=$(qsub "${dep_flag[@]}" \
        -v TM_SOURCE=avg24,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/${PREFIX}_solve_matrix_age_job.sh)
    echo "[$STEP] TMage avg24/Pardiso: $TMAGE_AVG_CPU${AVGM_JOB:+ (afterok $AVGM_JOB)}"

    STEP=$((STEP + 1))
    TMAGE_AVG_GPU=$(qsub "${dep_flag[@]}" \
        -v TM_SOURCE=avg24,LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/${PREFIX}_solve_matrix_age_GPU_job.sh)
    echo "[$STEP] TMage avg24/CUDSS: $TMAGE_AVG_GPU${AVGM_JOB:+ (afterok $AVGM_JOB)}"
fi

# --- NK (2 jobs: one per TM_SOURCE) ---
if has_step NK; then
    # const branch
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${CONSTM_JOB:-}" ] && dep_flag=(-W "depend=afterok:${CONSTM_JOB}")
    NK_CONST=$(qsub "${dep_flag[@]}" \
        -v TM_SOURCE=const,NONLINEAR_SOLVER=newton,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE} \
        scripts/${PREFIX}_GPU_job.sh)
    echo "[$STEP] NK const: $NK_CONST${CONSTM_JOB:+ (afterok $CONSTM_JOB)}"

    # avg24 branch
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${AVGM_JOB:-}" ] && dep_flag=(-W "depend=afterok:${AVGM_JOB}")
    NK_AVG=$(qsub "${dep_flag[@]}" \
        -v TM_SOURCE=avg24,NONLINEAR_SOLVER=newton,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE} \
        scripts/${PREFIX}_GPU_job.sh)
    echo "[$STEP] NK avg24: $NK_AVG${AVGM_JOB:+ (afterok $AVGM_JOB)}"
fi

echo ""
echo "=== $STEP jobs submitted ==="
echo ""
echo "Dependency chain:"
[ -n "$GRID_JOB" ]          && echo "  grid ($GRID_JOB)"
[ -n "$VEL_JOB" ]           && echo "   └── vel ($VEL_JOB)"
[ -n "$TEST_JOB" ]          && echo "        ├── 1year ($TEST_JOB)"
[ -n "$AVGM_JOB" ]          && echo "        │    └── snapM+avgM ($AVGM_JOB)"
[ -n "$TMAGE_AVG_CPU" ]     && echo "        │         ├── TMage avg24/Pardiso ($TMAGE_AVG_CPU)"
[ -n "$TMAGE_AVG_GPU" ]     && echo "        │         ├── TMage avg24/CUDSS ($TMAGE_AVG_GPU)"
[ -n "$NK_AVG" ]            && echo "        │         └── NK avg24 ($NK_AVG)"
[ -n "$CONSTM_JOB" ]        && echo "        └── constM ($CONSTM_JOB)"
[ -n "$TMAGE_CONST_CPU" ]   && echo "             ├── TMage const/Pardiso ($TMAGE_CONST_CPU)"
[ -n "$TMAGE_CONST_GPU" ]   && echo "             ├── TMage const/CUDSS ($TMAGE_CONST_GPU)"
[ -n "$NK_CONST" ]          && echo "             └── NK const ($NK_CONST)"
