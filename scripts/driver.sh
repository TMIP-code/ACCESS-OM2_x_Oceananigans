#!/usr/bin/env bash
set -euo pipefail

# Unified driver script: submits pipeline jobs for any PARENT_MODEL.
# Run from the login node (not as a PBS job).
#
# Usage:
#   PARENT_MODEL=ACCESS-OM2-1   JOB_CHAIN=preprocessing-run1yr bash scripts/driver.sh
#   PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=full bash scripts/driver.sh
#   JOB_CHAIN=vel..NK bash scripts/driver.sh                      # range notation
#   GPU_RESOURCES=gpuvolta JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
#
# Steps:
#   grid vel run1yr run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK
#   plot1yr plot10yr plot100yr plotNK plotNKtrace
#
# Shortcuts:
#   preprocessing  = grid-vel
#   standardruns   = run1yr-run10yr-run100yr-runlong
#   TMall          = TMbuild-TMsnapshot-TMsolve
#   plotall        = plot1yr-plot10yr-plot100yr-plotNK
#   full           = preprocessing-run1yr-TMall-NK-plot1yr-plotNK
#
# Range notation:
#   A..B expands to all steps from A to B in canonical order.
#   Example: vel..NK = vel-run1yr-run10yr-run100yr-runlong-TMbuild-TMsnapshot-TMsolve-NK
#
# Dependency DAG:
#   grid → vel → run1yr    (afterok vel)
#            │ → run10yr   (afterok vel, parallel with run1yr)
#            │ → run100yr  (afterok vel, parallel with run1yr)
#            │ → runlong   (afterok vel, parallel with run1yr)
#            │ → TMbuild   (afterok vel)       → TMsolve(const) + NK(const)
#            └→ run1yr → TMsnapshot            → TMsolve(avg24) + NK(avg24)
#
#   plot1yr     (afterok run1yr)
#   plot10yr    (afterok run10yr)
#   plot100yr   (afterok run100yr)
#   plotNK      (afterok NK)

PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
export PARENT_MODEL

# Source model config for MODEL_SHORT and walltimes
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"
MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2
    exit 1
fi
source "$MODEL_CONF"

# --- JOB_CHAIN: required ---
if [ -z "${JOB_CHAIN:-}" ]; then
    echo "Usage: PARENT_MODEL=... JOB_CHAIN=... bash scripts/driver.sh"
    echo ""
    echo "  PARENT_MODEL  Model to run (default: ACCESS-OM2-1)"
    echo ""
    echo "  Steps:"
    echo "    grid vel run1yr run10yr run100yr runlong"
    echo "    TMbuild TMsnapshot TMsolve NK"
    echo "    plot1yr plot10yr plot100yr plotNK plotNKtrace"
    echo ""
    echo "  Shortcuts:"
    echo "    preprocessing  = grid-vel"
    echo "    standardruns   = run1yr-run10yr-run100yr-runlong"
    echo "    TMall          = TMbuild-TMsnapshot-TMsolve"
    echo "    plotall        = plot1yr-plot10yr-plot100yr-plotNK"
    echo "    full           = preprocessing-run1yr-TMall-NK-plot1yr-plotNK"
    echo ""
    echo "  Range: A..B (e.g., vel..NK expands to all steps from vel to NK)"
    echo ""
    echo "  Examples:"
    echo "    JOB_CHAIN=preprocessing-run1yr-plot1yr bash scripts/driver.sh"
    echo "    PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=full bash scripts/driver.sh"
    echo "    JOB_CHAIN=vel..NK bash scripts/driver.sh"
    exit 1
fi

# --- Canonical step order (for range expansion) ---
ALL_STEPS=(grid vel run1yr run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK plot1yr plot10yr plot100yr plotNK plotNKtrace)

# --- Expand range notation (A..B) ---
expand_range() {
    local token="$1"
    if [[ "$token" != *".."* ]]; then
        echo "$token"
        return
    fi
    local start="${token%..*}"
    local end="${token#*..}"
    local in_range=false result=""
    for step in "${ALL_STEPS[@]}"; do
        [[ "$step" == "$start" ]] && in_range=true
        if $in_range; then
            result="${result:+$result-}$step"
        fi
        [[ "$step" == "$end" ]] && break
    done
    if [ -z "$result" ]; then
        echo "ERROR: Invalid range '$token' — check step names." >&2
        exit 1
    fi
    echo "$result"
}

# Process each hyphen-separated token: expand ranges first, then shortcuts
expanded=""
IFS='-' read -ra TOKENS <<< "$JOB_CHAIN"
for token in "${TOKENS[@]}"; do
    piece=$(expand_range "$token")
    expanded="${expanded:+$expanded-}$piece"
done
JOB_CHAIN="$expanded"

# Expand shortcuts (order matters: expand 'full' before its sub-shortcuts)
JOB_CHAIN="${JOB_CHAIN//full/preprocessing-run1yr-TMall-NK-plot1yr-plotNK}"
JOB_CHAIN="${JOB_CHAIN//preprocessing/grid-vel}"
JOB_CHAIN="${JOB_CHAIN//standardruns/run1yr-run10yr-run100yr-runlong}"
JOB_CHAIN="${JOB_CHAIN//TMall/TMbuild-TMsnapshot-TMsolve}"
JOB_CHAIN="${JOB_CHAIN//plotall/plot1yr-plot10yr-plot100yr-plotNK}"

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

# --- GPU queue configuration ---
GPU_RESOURCES=${GPU_RESOURCES:-gpuhopper}
case "$GPU_RESOURCES" in
    gpuvolta)  GPU_MEM=96GB;  GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuvolta ;;
    gpuvolta2) GPU_MEM=192GB; GPU_NGPUS=2; GPU_NCPUS=24; GPU_QUEUE=gpuvolta ;;
    gpuhopper) GPU_MEM=256GB; GPU_NGPUS=1; GPU_NCPUS=12; GPU_QUEUE=gpuhopper ;;
    *)         echo "Unknown GPU_RESOURCES=$GPU_RESOURCES (must be: gpuvolta, gpuvolta2, gpuhopper)"; exit 1 ;;
esac

# --- Solver configuration (shared by TMsolve and NK) ---
JVP_METHOD=${JVP_METHOD:-exact}
LINEAR_SOLVER=${LINEAR_SOLVER:-Pardiso}
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-yes}
INITIAL_AGE=${INITIAL_AGE:-0}

# --- Common -v vars passed to all jobs ---
COMMON_VARS="PARENT_MODEL=${PARENT_MODEL}"

echo "=== ${PARENT_MODEL} pipeline driver ==="
echo "MODEL_SHORT=$MODEL_SHORT"
echo "JOB_CHAIN=$JOB_CHAIN"
echo "GPU_RESOURCES=$GPU_RESOURCES (queue=$GPU_QUEUE, ngpus=$GPU_NGPUS, ncpus=$GPU_NCPUS, mem=$GPU_MEM)"
echo "JVP_METHOD=$JVP_METHOD, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, INITIAL_AGE=$INITIAL_AGE"
echo ""

STEP=0
GRID_JOB="" VEL_JOB="" RUN1YR_JOB="" RUN10YR_JOB="" RUN100YR_JOB="" RUNLONG_JOB=""
TMBUILD_JOB="" TMSNAP_JOB=""
TMSOLVE_CONST_CPU="" TMSOLVE_CONST_GPU="" TMSOLVE_AVG_CPU="" TMSOLVE_AVG_GPU=""
NK_CONST="" NK_AVG=""
PLOT1YR_JOB="" PLOT10YR_JOB="" PLOT100YR_JOB="" PLOTNK_JOB="" PLOTNKTRACE_JOB=""

# ============================================================
# 1. Preprocessing
# ============================================================

# 1a. grid
if has_step grid; then
    STEP=$((STEP + 1))
    GRID_JOB=$(qsub \
        -N "${MODEL_SHORT}_grid" -l walltime=${WALLTIME_GRID} \
        -v ${COMMON_VARS} \
        scripts/preprocessing/build_grid.sh)
    echo "[$STEP] Grid: $GRID_JOB"
fi

# 1b. vel (depends on: grid)
if has_step vel; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$GRID_JOB" ] && dep_flag=(-W "depend=afterok:${GRID_JOB}")
    gpu_flags=()
    PREPROCESS_ARCH=${PREPROCESS_ARCH:-CPU}
    if [ "$PREPROCESS_ARCH" = "GPU" ]; then
        gpu_flags=(-q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM)
    fi
    VEL_JOB=$(qsub "${dep_flag[@]}" "${gpu_flags[@]}" \
        -N "${MODEL_SHORT}_vel" -l walltime=${WALLTIME_VEL} \
        -v ${COMMON_VARS} \
        scripts/preprocessing/build_velocities.sh)
    echo "[$STEP] Velocities: $VEL_JOB${GRID_JOB:+ (afterok $GRID_JOB)}${PREPROCESS_ARCH:+ [$PREPROCESS_ARCH]}"
fi

VEL_DEP="${VEL_JOB:-${GRID_JOB:-}}"

# ============================================================
# 2. Standard runs (depend on: vel)
# ============================================================

# 2a. run1yr
if has_step run1yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN1YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run1yr" -l walltime=${WALLTIME_RUN_1YEAR} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS} \
        scripts/standard_runs/run_1year.sh)
    echo "[$STEP] 1-year run: $RUN1YR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 2b. run10yr (parallel with run1yr)
if has_step run10yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN10YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run10yr" -l walltime=${WALLTIME_RUN_10YEARS} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS} \
        scripts/standard_runs/run_10years.sh)
    echo "[$STEP] 10-year run: $RUN10YR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 2c. run100yr (parallel with run1yr)
if has_step run100yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN100YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run100yr" -l walltime=${WALLTIME_RUN_100YEARS} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS} \
        scripts/standard_runs/run_100years.sh)
    echo "[$STEP] 100-year run: $RUN100YR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 2d. runlong (parallel with run1yr)
if has_step runlong; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUNLONG_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_runlong" -l walltime=${WALLTIME_RUN_LONG} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},NYEARS=${NYEARS:-3000} \
        scripts/standard_runs/run_long.sh)
    echo "[$STEP] Long run: $RUNLONG_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# ============================================================
# 3. Transport matrix building
# ============================================================

# 3a. TMbuild — Jacobian from constant fields (depends on: vel)
if has_step TMbuild; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    TMBUILD_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_TMbuild" -l walltime=${WALLTIME_TM_BUILD} \
        -v ${COMMON_VARS} \
        scripts/preprocessing/build_TMconst.sh)
    echo "[$STEP] TM build (const): $TMBUILD_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 3b. TMsnapshot — snapshot + averaged matrices (depends on: run1yr)
if has_step TMsnapshot; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN1YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN1YR_JOB}")
    TMSNAP_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_TMsnap" -l walltime=${WALLTIME_TM_SNAPSHOT} \
        -v ${COMMON_VARS} \
        scripts/preprocessing/build_TMavg.sh)
    echo "[$STEP] TM snapshot+avg: $TMSNAP_JOB${RUN1YR_JOB:+ (afterok $RUN1YR_JOB)}"
fi

# ============================================================
# 4. Transport matrix age solving
# ============================================================

if has_step TMsolve; then
    # 4a. const branch — CPU (Pardiso)
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${TMBUILD_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMBUILD_JOB}")
    TMSOLVE_CONST_CPU=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_TMslv_c" -l walltime=${WALLTIME_TM_SOLVE} \
        -v ${COMMON_VARS},TM_SOURCE=const,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/solvers/solve_TM_age_CPU.sh)
    echo "[$STEP] TMsolve const/Pardiso: $TMSOLVE_CONST_CPU${TMBUILD_JOB:+ (afterok $TMBUILD_JOB)}"

    # 4b. const branch — GPU (CUDSS)
    STEP=$((STEP + 1))
    TMSOLVE_CONST_GPU=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_TMslv_cG" -l walltime=${WALLTIME_TM_SOLVE} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},TM_SOURCE=const,LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/solvers/solve_TM_age_GPU.sh)
    echo "[$STEP] TMsolve const/CUDSS: $TMSOLVE_CONST_GPU${TMBUILD_JOB:+ (afterok $TMBUILD_JOB)}"

    # 4c. avg24 branch — CPU (Pardiso)
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${TMSNAP_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMSNAP_JOB}")
    TMSOLVE_AVG_CPU=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_TMslv_a" -l walltime=${WALLTIME_TM_SOLVE} \
        -v ${COMMON_VARS},TM_SOURCE=avg24,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/solvers/solve_TM_age_CPU.sh)
    echo "[$STEP] TMsolve avg24/Pardiso: $TMSOLVE_AVG_CPU${TMSNAP_JOB:+ (afterok $TMSNAP_JOB)}"

    # 4d. avg24 branch — GPU (CUDSS)
    STEP=$((STEP + 1))
    TMSOLVE_AVG_GPU=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_TMslv_aG" -l walltime=${WALLTIME_TM_SOLVE} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},TM_SOURCE=avg24,LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/solvers/solve_TM_age_GPU.sh)
    echo "[$STEP] TMsolve avg24/CUDSS: $TMSOLVE_AVG_GPU${TMSNAP_JOB:+ (afterok $TMSNAP_JOB)}"
fi

# ============================================================
# 5. Newton-Krylov solver
# ============================================================

if has_step NK; then
    # 5a. NK from const TM (depends on: TMbuild)
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${TMBUILD_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMBUILD_JOB}")
    NK_CONST=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_NK_c" -l walltime=${WALLTIME_NK} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},TM_SOURCE=const,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE} \
        scripts/solvers/solve_periodic_NK.sh)
    echo "[$STEP] NK const: $NK_CONST${TMBUILD_JOB:+ (afterok $TMBUILD_JOB)}"

    # 5b. NK from avg24 TM (depends on: TMsnapshot)
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${TMSNAP_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMSNAP_JOB}")
    NK_AVG=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_NK_a" -l walltime=${WALLTIME_NK} \
        -q $GPU_QUEUE -l ngpus=$GPU_NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},TM_SOURCE=avg24,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE} \
        scripts/solvers/solve_periodic_NK.sh)
    echo "[$STEP] NK avg24: $NK_AVG${TMSNAP_JOB:+ (afterok $TMSNAP_JOB)}"
fi

# ============================================================
# 6. Plotting
# ============================================================

# 6a. plot1yr (depends on: run1yr)
if has_step plot1yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN1YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN1YR_JOB}")
    PLOT1YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plot1yr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS} \
        scripts/plotting/plot_1year_age.sh)
    echo "[$STEP] Plot 1yr: $PLOT1YR_JOB${RUN1YR_JOB:+ (afterok $RUN1YR_JOB)}"
fi

# 6b. plot10yr (depends on: run10yr)
if has_step plot10yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN10YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN10YR_JOB}")
    PLOT10YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plot10yr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS} \
        scripts/plotting/plot_10years_age.sh)
    echo "[$STEP] Plot 10yr: $PLOT10YR_JOB${RUN10YR_JOB:+ (afterok $RUN10YR_JOB)}"
fi

# 6c. plot100yr (depends on: run100yr)
if has_step plot100yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN100YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN100YR_JOB}")
    PLOT100YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plot100yr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS} \
        scripts/plotting/plot_100years_age.sh)
    echo "[$STEP] Plot 100yr: $PLOT100YR_JOB${RUN100YR_JOB:+ (afterok $RUN100YR_JOB)}"
fi

# 6d. plotNK (depends on: NK — uses const NK job by default)
if has_step plotNK; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${NK_CONST:-}" ] && dep_flag=(-W "depend=afterok:${NK_CONST}")
    PLOTNK_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plotNK" -l walltime=${WALLTIME_PLOT_NK} \
        -v ${COMMON_VARS},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/plotting/plot_1year_from_periodic_sol.sh)
    echo "[$STEP] Plot NK: $PLOTNK_JOB${NK_CONST:+ (afterok $NK_CONST)}"
fi

# 6e. plotNKtrace (depends on: NK — trace history plotting)
if has_step plotNKtrace; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${NK_CONST:-}" ] && dep_flag=(-W "depend=afterok:${NK_CONST}")
    PLOTNKTRACE_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plotNKtr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS} \
        scripts/plotting/plot_trace_history_job.sh)
    echo "[$STEP] Plot NK trace: $PLOTNKTRACE_JOB${NK_CONST:+ (afterok $NK_CONST)}"
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=== $STEP jobs submitted for ${PARENT_MODEL} ==="
echo ""
echo "Dependency chain:"
[ -n "$GRID_JOB" ]           && echo "  grid ($GRID_JOB)"
[ -n "$VEL_JOB" ]            && echo "   └── vel ($VEL_JOB)"
[ -n "$RUN1YR_JOB" ]         && echo "        ├── run1yr ($RUN1YR_JOB)"
[ -n "$PLOT1YR_JOB" ]        && echo "        │    └── plot1yr ($PLOT1YR_JOB)"
[ -n "$RUN10YR_JOB" ]        && echo "        ├── run10yr ($RUN10YR_JOB)"
[ -n "$PLOT10YR_JOB" ]       && echo "        │    └── plot10yr ($PLOT10YR_JOB)"
[ -n "$RUN100YR_JOB" ]       && echo "        ├── run100yr ($RUN100YR_JOB)"
[ -n "$PLOT100YR_JOB" ]      && echo "        │    └── plot100yr ($PLOT100YR_JOB)"
[ -n "$RUNLONG_JOB" ]        && echo "        ├── runlong ($RUNLONG_JOB)"
[ -n "$TMSNAP_JOB" ]         && echo "        │    └── TMsnapshot ($TMSNAP_JOB)"
[ -n "$TMSOLVE_AVG_CPU" ]    && echo "        │         ├── TMsolve avg24/Pardiso ($TMSOLVE_AVG_CPU)"
[ -n "$TMSOLVE_AVG_GPU" ]    && echo "        │         ├── TMsolve avg24/CUDSS ($TMSOLVE_AVG_GPU)"
[ -n "$NK_AVG" ]             && echo "        │         └── NK avg24 ($NK_AVG)"
[ -n "$TMBUILD_JOB" ]        && echo "        └── TMbuild ($TMBUILD_JOB)"
[ -n "$TMSOLVE_CONST_CPU" ]  && echo "             ├── TMsolve const/Pardiso ($TMSOLVE_CONST_CPU)"
[ -n "$TMSOLVE_CONST_GPU" ]  && echo "             ├── TMsolve const/CUDSS ($TMSOLVE_CONST_GPU)"
[ -n "$NK_CONST" ]           && echo "             ├── NK const ($NK_CONST)"
[ -n "$PLOTNK_JOB" ]         && echo "             │    └── plotNK ($PLOTNK_JOB)"
[ -n "$PLOTNKTRACE_JOB" ]    && echo "             │    └── plotNKtrace ($PLOTNKTRACE_JOB)"
