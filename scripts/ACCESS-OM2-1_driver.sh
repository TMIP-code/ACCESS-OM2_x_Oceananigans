#!/usr/bin/env bash
set -euo pipefail

# Driver script: submits the ACCESS-OM2-1 pipeline as chained PBS jobs.
# Run from the login node (not as a PBS job).
#
# Usage:
#   bash scripts/ACCESS-OM2-1_driver.sh                                      # full pipeline (default)
#   JOB_CHAIN=grid-vel-1year-10years-100years-TM-TMage-NK bash scripts/...   # with 10yr+100yr
#   GPU_RESOURCES=gpuvolta bash scripts/ACCESS-OM2-1_driver.sh                # on Volta GPUs
#   GPU_RESOURCES=gpuvolta2 bash scripts/ACCESS-OM2-1_driver.sh               # 2×Volta GPUs
#   JOB_CHAIN=TM-TMage-NK bash scripts/ACCESS-OM2-1_driver.sh               # restart from TM
#   PREPROCESS_ARCH=GPU bash scripts/ACCESS-OM2-1_driver.sh                  # velocities on GPU
#
# Dependency DAG:
#   grid → vel → 1year    (afterok vel)
#            │ → 10years  (afterok vel, opt-in, parallel with 1year)
#            │ → 100years (afterok vel, opt-in, parallel with 1year)
#            │ → long     (afterok vel, opt-in, parallel with 1year)
#            │ → constM (TM_SOURCE=const)               → TMage(const) + NK(const)
#            └→ 1year → snapM+avgM (TM_SOURCE=avg24)    → TMage(avg24) + NK(avg24)

PARENT_MODEL=ACCESS-OM2-1
PREFIX=ACCESS-OM2-1

# Pipeline configuration
JOB_CHAIN=${JOB_CHAIN:-grid-vel-1year-TM-TMage-NK}
PREPROCESS_ARCH=${PREPROCESS_ARCH:-CPU}

# GPU queue configuration
GPU_RESOURCES=${GPU_RESOURCES:-gpuhopper}
case "$GPU_RESOURCES" in
    gpuvolta)  GPU_MEM=96GB;  GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuvolta ;;
    gpuvolta2) GPU_MEM=192GB; GPU_NGPUS=2; GPU_NCPUS=24; GPU_QUEUE=gpuvolta ;;
    gpuhopper) GPU_MEM=256GB; GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuhopper ;;
    *)         echo "Unknown GPU_RESOURCES=$GPU_RESOURCES (must be: gpuvolta, gpuvolta2, gpuhopper)"; exit 1 ;;
esac

# Solver configuration (shared by TMage and NK)
JVP_METHOD=exact
LINEAR_SOLVER=Pardiso
LUMP_AND_SPRAY=yes
INITIAL_AGE=0

echo "=== ${PARENT_MODEL} pipeline driver ==="
echo "JOB_CHAIN=$JOB_CHAIN"
echo "PREPROCESS_ARCH=$PREPROCESS_ARCH"
echo "GPU_RESOURCES=$GPU_RESOURCES (queue=$GPU_QUEUE, ngpus=$GPU_NGPUS, ncpus=$GPU_NCPUS, mem=$GPU_MEM)"
echo "JVP_METHOD=$JVP_METHOD, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, INITIAL_AGE=$INITIAL_AGE"

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

STEP=0
LAST_DEP=""
GRID_JOB="" VEL_JOB="" TEST_JOB="" CONSTM_JOB="" AVGM_JOB=""
TENYEAR_JOB="" HUNDREDYEAR_JOB="" LONG_JOB=""
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
        gpu_flags=(-q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM)
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
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        scripts/${PREFIX}_run_1year.sh)
    echo "[$STEP] 1-year test: $TEST_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# --- 10years (opt-in, parallel with 1year) ---
if has_step 10years; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    TENYEAR_JOB=$(qsub "${dep_flag[@]}" \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        scripts/${PREFIX}_run_10years.sh)
    echo "[$STEP] 10-year run: $TENYEAR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# --- 100years (opt-in, parallel with 1year) ---
if has_step 100years; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    HUNDREDYEAR_JOB=$(qsub "${dep_flag[@]}" \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        scripts/${PREFIX}_run_100years.sh)
    echo "[$STEP] 100-year run: $HUNDREDYEAR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# --- long (opt-in, parallel with 1year) ---
if has_step long; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    LONG_JOB=$(qsub "${dep_flag[@]}" \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v NYEARS=${NYEARS:-3000} \
        scripts/${PREFIX}_run_long.sh)
    echo "[$STEP] Long run: $LONG_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
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
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
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
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
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
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v TM_SOURCE=const,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE} \
        scripts/${PREFIX}_solve_periodic_NK.sh)
    echo "[$STEP] NK const: $NK_CONST${CONSTM_JOB:+ (afterok $CONSTM_JOB)}"

    # avg24 branch
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${AVGM_JOB:-}" ] && dep_flag=(-W "depend=afterok:${AVGM_JOB}")
    NK_AVG=$(qsub "${dep_flag[@]}" \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v TM_SOURCE=avg24,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE} \
        scripts/${PREFIX}_solve_periodic_NK.sh)
    echo "[$STEP] NK avg24: $NK_AVG${AVGM_JOB:+ (afterok $AVGM_JOB)}"
fi

# --- AA (commented out — re-enable when needed) ---
# if has_step AA; then
#     STEP=$((STEP + 1))
#     dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
#     AA_JOB=$(qsub "${dep_flag[@]}" \
#         -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
#         -v INITIAL_AGE=TMage \
#         scripts/${PREFIX}_solve_periodic_AA.sh)
#     echo "[$STEP] AA: $AA_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
# fi

echo ""
echo "=== $STEP jobs submitted ==="
echo ""
echo "Dependency chain:"
[ -n "$GRID_JOB" ]          && echo "  grid ($GRID_JOB)"
[ -n "$VEL_JOB" ]           && echo "   └── vel ($VEL_JOB)"
[ -n "$TEST_JOB" ]          && echo "        ├── 1year ($TEST_JOB)"
[ -n "$TENYEAR_JOB" ]       && echo "        ├── 10years ($TENYEAR_JOB)"
[ -n "$HUNDREDYEAR_JOB" ]   && echo "        ├── 100years ($HUNDREDYEAR_JOB)"
[ -n "$LONG_JOB" ]          && echo "        ├── long ($LONG_JOB)"
[ -n "$AVGM_JOB" ]          && echo "        │    └── snapM+avgM ($AVGM_JOB)"
[ -n "$TMAGE_AVG_CPU" ]     && echo "        │         ├── TMage avg24/Pardiso ($TMAGE_AVG_CPU)"
[ -n "$TMAGE_AVG_GPU" ]     && echo "        │         ├── TMage avg24/CUDSS ($TMAGE_AVG_GPU)"
[ -n "$NK_AVG" ]            && echo "        │         └── NK avg24 ($NK_AVG)"
[ -n "$CONSTM_JOB" ]        && echo "        └── constM ($CONSTM_JOB)"
[ -n "$TMAGE_CONST_CPU" ]   && echo "             ├── TMage const/Pardiso ($TMAGE_CONST_CPU)"
[ -n "$TMAGE_CONST_GPU" ]   && echo "             ├── TMage const/CUDSS ($TMAGE_CONST_GPU)"
[ -n "$NK_CONST" ]          && echo "             └── NK const ($NK_CONST)"
