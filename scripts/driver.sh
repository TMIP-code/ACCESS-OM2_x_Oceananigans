#!/usr/bin/env bash
set -euo pipefail

# Unified driver script: submits pipeline jobs for any PARENT_MODEL.
# Run from the login node (not as a PBS job).
#
# Usage:
#   PARENT_MODEL=ACCESS-OM2-1   JOB_CHAIN=preprocessing-run1yr bash scripts/driver.sh
#   PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=full bash scripts/driver.sh
#   EXPERIMENT=1deg_jra55_ryf9091_gadi TIME_WINDOW=1958-1987 JOB_CHAIN=full bash scripts/driver.sh
#   JOB_CHAIN=vel..NK bash scripts/driver.sh                      # range notation
#   GPU_QUEUE=gpuvolta PARTITION=2x2 JOB_CHAIN=run1yr-plot1yr bash scripts/driver.sh
#   TM_SOURCE=both JOB_CHAIN=NK bash scripts/driver.sh            # run both const+avg
#
# Steps:
#   prep grid vel run1yr run10yr run100yr runlong
#   TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM
#   plot1yr plot10yr plot100yr
#
# Shortcuts:
#   preprocessing  = prep-grid-vel
#   standardruns   = run1yr-run10yr-run100yr-runlong
#   TMall          = TMbuild-TMsnapshot-TMsolve
#   plotall         = plot1yr-plot10yr-plot100yr-plotNK-plotTM
#   full           = preprocessing-run1yr-TMall-NK-run1yrNK-plotNK-plot1yr
#
# Range notation:
#   A..B expands to all steps on any path from A to B in the dependency DAG.
#   Example: run1yrNK..plotNK = run1yrNK-plotNK (not plot1yr/10yr/100yr)
#   Example: vel..NK = vel-run1yr-TMbuild-TMsnapshot-NK (not run10yr/runlong/TMsolve)
#
# TM_SOURCE filtering:
#   TM_SOURCE=const  (default) — only const branch for TMsolve/NK/run1yrNK
#   TM_SOURCE=avg  — only avg branch
#   TM_SOURCE=both   — both branches
#
# Dependency DAG:
#   prep ─┐
#   grid ─┤→ vel → run1yr    (afterok vel)
#          │     → run10yr   (afterok vel, parallel)
#          │     → run100yr  (afterok vel, parallel)
#          │     → runlong   (afterok vel, parallel)
#          │     → TMbuild   (afterok vel) → TMsolve(const) + NK(const) → run1yrNK → plotNK
#          │                             └→ plotTM (afterok TMbuild + TMsnapshot)
#          └─────→ run1yr → TMsnapshot     → TMsolve(avg) + NK(avg) → run1yrNK → plotNK
#
#   plot1yr     (afterok run1yr)
#   plot10yr    (afterok run10yr)
#   plot100yr   (afterok run100yr)
#   plotNKtrace (afterok NK)

PARENT_MODEL=${PARENT_MODEL:-ACCESS-OM2-1}
export PARENT_MODEL

# Experiment and time window
if [ -z "${EXPERIMENT:-}" ]; then
    case "$PARENT_MODEL" in
        ACCESS-OM2-1)   EXPERIMENT="1deg_jra55_iaf_omip2_cycle6" ;;
        ACCESS-OM2-025) EXPERIMENT="025deg_jra55_iaf_omip2_cycle6" ;;
        *)              echo "ERROR: No default EXPERIMENT for $PARENT_MODEL" >&2; exit 1 ;;
    esac
fi
TIME_WINDOW=${TIME_WINDOW:-1960-1979}
export EXPERIMENT TIME_WINDOW

# Source model config for MODEL_SHORT and walltimes
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

# Require clean git status before submitting jobs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: Commit before you submit a job. Working tree is not clean:" >&2
    git status --short >&2
    exit 1
fi
GIT_COMMIT=$(git rev-parse HEAD)

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
    echo "  EXPERIMENT    Intake catalog key (default: based on PARENT_MODEL)"
    echo "  TIME_WINDOW   Year range YYYY-YYYY or single year (default: 1960-1979)"
    echo "  TM_SOURCE     const (default), avg, or both"
    echo ""
    echo "  Steps:"
    echo "    prep grid vel run1yr run1yrfast run10yr run100yr runlong"
    echo "    TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM"
    echo "    plot1yr plot10yr plot100yr"
    echo ""
    echo "  Shortcuts:"
    echo "    preprocessing  = prep-grid-vel"
    echo "    standardruns   = run1yr-run10yr-run100yr-runlong"
    echo "    TMall          = TMbuild-TMsnapshot-TMsolve"
    echo "    plotall         = plot1yr-plot10yr-plot100yr-plotNK"
    echo "    full           = preprocessing-run1yr-TMall-NK-run1yrNK-plotNK-plot1yr"
    echo ""
    echo "  Range: A..B follows the dependency DAG (e.g., run1yrNK..plotNK = run1yrNK-plotNK)"
    echo ""
    echo "  Examples:"
    echo "    JOB_CHAIN=preprocessing-run1yr-plot1yr bash scripts/driver.sh"
    echo "    PARENT_MODEL=ACCESS-OM2-025 JOB_CHAIN=full bash scripts/driver.sh"
    echo "    JOB_CHAIN=vel..NK bash scripts/driver.sh"
    echo "    TM_SOURCE=both JOB_CHAIN=NK-run1yrNK-plotNK bash scripts/driver.sh"
    exit 1
fi

# --- Topological step order (for deterministic output in range expansion) ---
ALL_STEPS=(prep grid vel run1yr run1yrfast run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM plot1yr plot10yr plot100yr)

# --- Dependency DAG (step → space-separated children) ---
declare -A DAG
DAG[prep]="vel"
DAG[grid]="vel"
DAG[vel]="run1yr run1yrfast run10yr run100yr runlong TMbuild"
DAG[run1yr]="TMsnapshot plot1yr"
DAG[run10yr]="plot10yr"
DAG[run100yr]="plot100yr"
DAG[TMbuild]="TMsolve NK plotTM"
DAG[TMsnapshot]="TMsolve NK plotTM"
DAG[NK]="run1yrNK plotNKtrace"
DAG[run1yrNK]="plotNK"

# --- DAG-based range expansion (A..B → steps on any path from A to B) ---
expand_range() {
    local token="$1"
    [[ "$token" != *".."* ]] && { echo "$token"; return; }
    local start="${token%..*}" end="${token#*..}"

    # Forward reachability from start (BFS)
    local -A fwd
    local queue=("$start")
    while [ ${#queue[@]} -gt 0 ]; do
        local node="${queue[0]}"; queue=("${queue[@]:1}")
        [ -n "${fwd[$node]:-}" ] && continue; fwd[$node]=1
        for child in ${DAG[$node]:-}; do queue+=("$child"); done
    done

    # Backward reachability to end (BFS on reverse DAG)
    local -A bwd rdag
    for node in "${!DAG[@]}"; do
        for child in ${DAG[$node]}; do
            rdag[$child]="${rdag[$child]:-} $node"
        done
    done
    queue=("$end")
    while [ ${#queue[@]} -gt 0 ]; do
        local node="${queue[0]}"; queue=("${queue[@]:1}")
        [ -n "${bwd[$node]:-}" ] && continue; bwd[$node]=1
        for parent in ${rdag[$node]:-}; do queue+=("$parent"); done
    done

    # Intersection in topological order
    local result=""
    for step in "${ALL_STEPS[@]}"; do
        if [ -n "${fwd[$step]:-}" ] && [ -n "${bwd[$step]:-}" ]; then
            result="${result:+$result-}$step"
        fi
    done
    if [ -z "$result" ]; then
        echo "ERROR: No path from '$start' to '$end' in the dependency DAG." >&2
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
JOB_CHAIN="${JOB_CHAIN//full/preprocessing-run1yr-TMall-NK-run1yrNK-plotNK-plot1yr}"
JOB_CHAIN="${JOB_CHAIN//preprocessing/prep-grid-vel-partition}"
JOB_CHAIN="${JOB_CHAIN//standardruns/run1yr-run10yr-run100yr-runlong}"
JOB_CHAIN="${JOB_CHAIN//TMall/TMbuild-TMsnapshot-TMsolve}"
JOB_CHAIN="${JOB_CHAIN//plotall/plot1yr-plot10yr-plot100yr-plotNK-plotTM}"

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

# --- Partition + queue configuration ---
# PARTITION: MPI partition layout (default 1x1 = serial)
# GPU_QUEUE: set by model config (gpuvolta or gpuhopper)
# CPU_QUEUE: queue for CPU-only jobs (default express)
PARTITION=${PARTITION:-1x1}
PARTITION_X="${PARTITION%%x*}"
PARTITION_Y="${PARTITION#*x}"
CPU_QUEUE=${CPU_QUEUE:-express}
source "$(dirname "${BASH_SOURCE[0]}")/compute_resources.sh"

export PARTITION PARTITION_X PARTITION_Y RANKS

# --- TM_SOURCE filtering (const, avg, or both) ---
TM_SOURCE=${TM_SOURCE:-const}
run_const() { [[ "$TM_SOURCE" == "const" || "$TM_SOURCE" == "both" ]]; }
run_avg() { [[ "$TM_SOURCE" == "avg" || "$TM_SOURCE" == "both" ]]; }

# --- Solver configuration (shared by TMsolve and NK) ---
JVP_METHOD=${JVP_METHOD:-exact}
LINEAR_SOLVER=${LINEAR_SOLVER:-Pardiso}
LUMP_AND_SPRAY=${LUMP_AND_SPRAY:-yes}
INITIAL_AGE=${INITIAL_AGE:-0}

# --- Common -v vars passed to all jobs ---
COMMON_VARS="PARENT_MODEL=${PARENT_MODEL},EXPERIMENT=${EXPERIMENT},TIME_WINDOW=${TIME_WINDOW},GIT_COMMIT=${GIT_COMMIT}"

echo "=== ${PARENT_MODEL} pipeline driver ==="
echo "MODEL_SHORT=$MODEL_SHORT"
echo "EXPERIMENT=$EXPERIMENT"
echo "TIME_WINDOW=$TIME_WINDOW"
echo "JOB_CHAIN=$JOB_CHAIN"
echo "GIT_COMMIT=$GIT_COMMIT"
echo "TM_SOURCE=$TM_SOURCE"
echo "GPU_QUEUE=$GPU_QUEUE, PARTITION=$PARTITION (${PARTITION_X}x${PARTITION_Y}), RANKS=$RANKS, NGPUS=$NGPUS, GPU_NCPUS=$GPU_NCPUS, GPU_MEM=$GPU_MEM"
echo "CPU_QUEUE=$CPU_QUEUE, CPU_NCPUS=$CPU_NCPUS, CPU_MEM=$CPU_MEM"
echo "JVP_METHOD=$JVP_METHOD, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, INITIAL_AGE=$INITIAL_AGE"
echo ""

STEP=0
PREP_JOB="" GRID_JOB="" VEL_JOB="" RUN1YR_JOB="" RUN1YRFAST_JOB="" RUN10YR_JOB="" RUN100YR_JOB="" RUNLONG_JOB=""
TMBUILD_JOB="" TMSNAP_JOB=""
TMSOLVE_CONST_CPU="" TMSOLVE_CONST_GPU="" TMSOLVE_AVG_CPU="" TMSOLVE_AVG_GPU=""
NK_CONST="" NK_AVG="" RUNNK_CONST="" RUNNK_AVG=""
PLOTTM_JOBS=() PLOTTM_LABELS=()
PLOT1YR_JOB="" PLOT10YR_JOB="" PLOT100YR_JOB="" PLOTNK_JOB="" PLOTNKTRACE_JOB=""

# ============================================================
# 1. Preprocessing
# ============================================================

# 1a. prep — Python preprocessing (monthly climatologies + yearly averages)
if has_step prep; then
    STEP=$((STEP + 1))
    PREP_JOB=$(qsub \
        -N "${MODEL_SHORT}_prep" -l walltime=${WALLTIME_PREP} \
        -v ${COMMON_VARS} \
        scripts/prepreprocessing/periodicaverage.sh)
    echo "[$STEP] Prep: $PREP_JOB"
fi

# 1b. grid
if has_step grid; then
    STEP=$((STEP + 1))
    GRID_JOB=$(qsub \
        -N "${MODEL_SHORT}_grid" -l walltime=${WALLTIME_GRID} \
        -v ${COMMON_VARS} \
        scripts/preprocessing/build_grid.sh)
    echo "[$STEP] Grid: $GRID_JOB"
fi

# 1c. vel (depends on: grid + prep)
if has_step vel; then
    STEP=$((STEP + 1))
    deps=""
    [ -n "$GRID_JOB" ] && deps="${deps:+$deps:}$GRID_JOB"
    [ -n "$PREP_JOB" ] && deps="${deps:+$deps:}$PREP_JOB"
    dep_flag=(); [ -n "$deps" ] && dep_flag=(-W "depend=afterok:${deps}")
    gpu_flags=()
    PREPROCESS_ARCH=${PREPROCESS_ARCH:-CPU}
    if [ "$PREPROCESS_ARCH" = "GPU" ]; then
        gpu_flags=(-q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM)
    fi
    VEL_JOB=$(qsub "${dep_flag[@]}" "${gpu_flags[@]}" \
        -N "${MODEL_SHORT}_vel" -l walltime=${WALLTIME_VEL} \
        -v ${COMMON_VARS} \
        scripts/preprocessing/build_velocities.sh)
    echo "[$STEP] Velocities: $VEL_JOB${deps:+ (afterok $deps)}${PREPROCESS_ARCH:+ [$PREPROCESS_ARCH]}"
fi

VEL_DEP="${VEL_JOB:-${GRID_JOB:-}}"

# 1d. partition (depends on: vel + grid, only if multi-rank)
PARTITION_JOB=""
if has_step partition && [[ "$PARTITION" != "1x1" ]]; then
    STEP=$((STEP + 1))
    deps=""
    [ -n "$VEL_JOB" ] && deps="${deps:+$deps:}$VEL_JOB"
    [ -n "$GRID_JOB" ] && deps="${deps:+$deps:}$GRID_JOB"
    dep_flag=(); [ -n "$deps" ] && dep_flag=(-W "depend=afterok:${deps}")
    PARTITION_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_partition" -l walltime=00:30:00 \
        -q $CPU_QUEUE -l ngpus=0 -l ncpus=$CPU_NCPUS -l mem=$CPU_MEM \
        -v ${COMMON_VARS},PARTITION=${PARTITION} \
        scripts/preprocessing/partition_data.sh)
    echo "[$STEP] Partition (${PARTITION_X}x${PARTITION_Y}, CPU): $PARTITION_JOB${deps:+ (afterok $deps)}"
fi

# Update dependency: standard runs depend on partition (if it exists) or vel
if [ -n "$PARTITION_JOB" ]; then
    VEL_DEP="$PARTITION_JOB"
fi

# ============================================================
# 2. Standard runs (depend on: vel or partition)
# ============================================================

# 2a. run1yr
if has_step run1yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN1YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run1yr" -l walltime=${WALLTIME_RUN_1YEAR} \
        -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},PARTITION=${PARTITION} \
        scripts/standard_runs/run_1year.sh)
    echo "[$STEP] 1-year run: $RUN1YR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 2b. run1yrfast — benchmark 1-year walltime (no output writers)
if has_step run1yrfast; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN1YRFAST_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run1yrfast" -l walltime=${WALLTIME_RUN_1YEAR} \
        -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},PARTITION=${PARTITION},PROFILE=${PROFILE:-no} \
        scripts/standard_runs/run_1year_benchmark.sh)
    echo "[$STEP] 1-year benchmark: $RUN1YRFAST_JOB${VEL_DEP:+ (afterok $VEL_DEP)}${PROFILE:+ [PROFILE=$PROFILE]}"
fi

# 2c. run10yr (parallel with run1yr)
if has_step run10yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN10YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run10yr" -l walltime=${WALLTIME_RUN_10YEARS} \
        -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},PARTITION=${PARTITION} \
        scripts/standard_runs/run_10years.sh)
    echo "[$STEP] 10-year run: $RUN10YR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 2c. run100yr (parallel with run1yr)
if has_step run100yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUN100YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_run100yr" -l walltime=${WALLTIME_RUN_100YEARS} \
        -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},PARTITION=${PARTITION} \
        scripts/standard_runs/run_100years.sh)
    echo "[$STEP] 100-year run: $RUN100YR_JOB${VEL_DEP:+ (afterok $VEL_DEP)}"
fi

# 2d. runlong (parallel with run1yr)
if has_step runlong; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "$VEL_DEP" ] && dep_flag=(-W "depend=afterok:${VEL_DEP}")
    RUNLONG_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_runlong" -l walltime=${WALLTIME_RUN_LONG} \
        -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
        -v ${COMMON_VARS},NYEARS=${NYEARS:-3000},PARTITION=${PARTITION} \
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
# 4. Transport matrix age solving (filtered by TM_SOURCE)
# ============================================================

if has_step TMsolve; then
    if run_const; then
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
            -q $GPU_QUEUE -l ngpus=1 -l ncpus=12 -l mem=$GPU_MEM_SINGLE \
            -v ${COMMON_VARS},TM_SOURCE=const,LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
            scripts/solvers/solve_TM_age_GPU.sh)
        echo "[$STEP] TMsolve const/CUDSS: $TMSOLVE_CONST_GPU${TMBUILD_JOB:+ (afterok $TMBUILD_JOB)}"
    fi

    if run_avg; then
        # 4c. avg branch — CPU (Pardiso)
        STEP=$((STEP + 1))
        dep_flag=(); [ -n "${TMSNAP_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMSNAP_JOB}")
        TMSOLVE_AVG_CPU=$(qsub "${dep_flag[@]}" \
            -N "${MODEL_SHORT}_TMslv_a" -l walltime=${WALLTIME_TM_SOLVE} \
            -v ${COMMON_VARS},TM_SOURCE=avg,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
            scripts/solvers/solve_TM_age_CPU.sh)
        echo "[$STEP] TMsolve avg/Pardiso: $TMSOLVE_AVG_CPU${TMSNAP_JOB:+ (afterok $TMSNAP_JOB)}"

        # 4d. avg branch — GPU (CUDSS)
        STEP=$((STEP + 1))
        TMSOLVE_AVG_GPU=$(qsub "${dep_flag[@]}" \
            -N "${MODEL_SHORT}_TMslv_aG" -l walltime=${WALLTIME_TM_SOLVE} \
            -q $GPU_QUEUE -l ngpus=1 -l ncpus=12 -l mem=$GPU_MEM_SINGLE \
            -v ${COMMON_VARS},TM_SOURCE=avg,LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
            scripts/solvers/solve_TM_age_GPU.sh)
        echo "[$STEP] TMsolve avg/CUDSS: $TMSOLVE_AVG_GPU${TMSNAP_JOB:+ (afterok $TMSNAP_JOB)}"
    fi
fi

# ============================================================
# 5. Newton-Krylov solver (filtered by TM_SOURCE)
# ============================================================

if has_step NK; then
    if run_const; then
        # 5a. NK from const TM (depends on: TMbuild)
        STEP=$((STEP + 1))
        dep_flag=(); [ -n "${TMBUILD_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMBUILD_JOB}")
        NK_CONST=$(qsub "${dep_flag[@]}" \
            -N "${MODEL_SHORT}_NK_c" -l walltime=${WALLTIME_NK} \
            -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
            -v ${COMMON_VARS},TM_SOURCE=const,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE},PARTITION=${PARTITION} \
            scripts/solvers/solve_periodic_NK.sh)
        echo "[$STEP] NK const: $NK_CONST${TMBUILD_JOB:+ (afterok $TMBUILD_JOB)}"
    fi

    if run_avg; then
        # 5b. NK from avg TM (depends on: TMsnapshot)
        STEP=$((STEP + 1))
        dep_flag=(); [ -n "${TMSNAP_JOB:-}" ] && dep_flag=(-W "depend=afterok:${TMSNAP_JOB}")
        NK_AVG=$(qsub "${dep_flag[@]}" \
            -N "${MODEL_SHORT}_NK_a" -l walltime=${WALLTIME_NK} \
            -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
            -v ${COMMON_VARS},TM_SOURCE=avg,JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE},PARTITION=${PARTITION} \
            scripts/solvers/solve_periodic_NK.sh)
        echo "[$STEP] NK avg: $NK_AVG${TMSNAP_JOB:+ (afterok $TMSNAP_JOB)}"
    fi
fi

# ============================================================
# 5b. Re-run 1yr from periodic solution (GPU, depends on NK)
#     Filtered by TM_SOURCE
# ============================================================

if has_step run1yrNK; then
    if run_const; then
        # const branch (depends on NK const)
        STEP=$((STEP + 1))
        dep_flag=(); [ -n "${NK_CONST:-}" ] && dep_flag=(-W "depend=afterok:${NK_CONST}")
        RUNNK_CONST=$(qsub "${dep_flag[@]}" \
            -N "${MODEL_SHORT}_run1yrNK_c" -l walltime=${WALLTIME_RUN_1YEAR} \
            -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
            -v ${COMMON_VARS},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},PARTITION=${PARTITION} \
            scripts/standard_runs/run_1year_from_periodic_sol.sh)
        echo "[$STEP] Run NK 1yr (const): $RUNNK_CONST${NK_CONST:+ (afterok $NK_CONST)}"
    fi

    if run_avg; then
        # avg branch (depends on NK avg)
        STEP=$((STEP + 1))
        dep_flag=(); [ -n "${NK_AVG:-}" ] && dep_flag=(-W "depend=afterok:${NK_AVG}")
        RUNNK_AVG=$(qsub "${dep_flag[@]}" \
            -N "${MODEL_SHORT}_run1yrNK_a" -l walltime=${WALLTIME_RUN_1YEAR} \
            -q $GPU_QUEUE -l ngpus=$NGPUS -l ncpus=$GPU_NCPUS -l mem=$GPU_MEM \
            -v ${COMMON_VARS},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},PARTITION=${PARTITION} \
            scripts/standard_runs/run_1year_from_periodic_sol.sh)
        echo "[$STEP] Run NK 1yr (avg): $RUNNK_AVG${NK_AVG:+ (afterok $NK_AVG)}"
    fi
fi

# ============================================================
# 6. Plotting
# ============================================================

# 6a. plotTM (depends on: TMbuild + TMsnapshot — needs all matrix variants)
if has_step plotTM; then
    deps=()
    [ -n "${TMBUILD_JOB:-}" ] && deps+=("${TMBUILD_JOB}")
    [ -n "${TMSNAP_JOB:-}" ] && deps+=("${TMSNAP_JOB}")
    dep_flag=()
    if [ ${#deps[@]} -gt 0 ]; then
        dep_str=$(IFS=:; echo "${deps[*]}")
        dep_flag=(-W "depend=afterok:${dep_str}")
    fi
    plot_tm_res=()
    [ -n "${PLOT_TM_NCPUS:-}" ] && plot_tm_res+=(-l "ncpus=${PLOT_TM_NCPUS}")
    [ -n "${PLOT_TM_MEM:-}" ] && plot_tm_res+=(-l "mem=${PLOT_TM_MEM}")
    for pair in const:avg; do
        lx="${pair%%:*}"; ly="${pair#*:}"
        STEP=$((STEP + 1))
        job=$(qsub "${dep_flag[@]}" "${plot_tm_res[@]}" \
            -N "${MODEL_SHORT}_plotTM_${ly}_vs_${lx}" -l walltime=${WALLTIME_PLOT} \
            -v "${COMMON_VARS},TM_LABEL_X=${lx},TM_LABEL_Y=${ly}" \
            scripts/plotting/plot_TM_datashader.sh)
        PLOTTM_JOBS+=("$job"); PLOTTM_LABELS+=("${ly} vs ${lx}")
        echo "[$STEP] Plot TM ${ly} vs ${lx}: $job${TMBUILD_JOB:+ (afterok $TMBUILD_JOB)}${TMSNAP_JOB:+, $TMSNAP_JOB}"
    done
fi

# 6b. plotNK (depends on: run1yrNK — needs the re-run snapshots)
if has_step plotNK; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUNNK_CONST:-}" ] && dep_flag=(-W "depend=afterok:${RUNNK_CONST}")
    PLOTNK_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plotNK" -l walltime=${WALLTIME_PLOT_NK} \
        -v ${COMMON_VARS},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY} \
        scripts/plotting/plot_1year_from_periodic_sol.sh)
    echo "[$STEP] Plot NK: $PLOTNK_JOB${RUNNK_CONST:+ (afterok $RUNNK_CONST)}"
fi

# 6c. plotNKtrace (depends on: NK — trace history plotting)
if has_step plotNKtrace; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${NK_CONST:-}" ] && dep_flag=(-W "depend=afterok:${NK_CONST}")
    PLOTNKTRACE_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plotNKtr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS} \
        scripts/plotting/plot_trace_history_job.sh)
    echo "[$STEP] Plot NK trace: $PLOTNKTRACE_JOB${NK_CONST:+ (afterok $NK_CONST)}"
fi

# 6d. plot1yr (depends on: run1yr)
if has_step plot1yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN1YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN1YR_JOB}")
    PLOT1YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plot1yr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS},DURATION=1year \
        scripts/plotting/plot_standardrun_age.sh)
    echo "[$STEP] Plot 1yr: $PLOT1YR_JOB${RUN1YR_JOB:+ (afterok $RUN1YR_JOB)}"
fi

# 6e. plot10yr (depends on: run10yr)
if has_step plot10yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN10YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN10YR_JOB}")
    PLOT10YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plot10yr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS},DURATION=10years \
        scripts/plotting/plot_standardrun_age.sh)
    echo "[$STEP] Plot 10yr: $PLOT10YR_JOB${RUN10YR_JOB:+ (afterok $RUN10YR_JOB)}"
fi

# 6f. plot100yr (depends on: run100yr)
if has_step plot100yr; then
    STEP=$((STEP + 1))
    dep_flag=(); [ -n "${RUN100YR_JOB:-}" ] && dep_flag=(-W "depend=afterok:${RUN100YR_JOB}")
    PLOT100YR_JOB=$(qsub "${dep_flag[@]}" \
        -N "${MODEL_SHORT}_plot100yr" -l walltime=${WALLTIME_PLOT} \
        -v ${COMMON_VARS},DURATION=100years \
        scripts/plotting/plot_standardrun_age.sh)
    echo "[$STEP] Plot 100yr: $PLOT100YR_JOB${RUN100YR_JOB:+ (afterok $RUN100YR_JOB)}"
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=== $STEP jobs submitted for ${PARENT_MODEL} (TM_SOURCE=$TM_SOURCE) ==="
echo ""

# Flat summary — only shows submitted jobs, grouped by section
has_any=false
for label_job in \
    "prep:$PREP_JOB" \
    "grid:$GRID_JOB" \
    "vel:$VEL_JOB" \
    "run1yr:$RUN1YR_JOB" \
    "run1yrfast:$RUN1YRFAST_JOB" \
    "run10yr:$RUN10YR_JOB" \
    "run100yr:$RUN100YR_JOB" \
    "runlong:$RUNLONG_JOB" \
    "TMbuild:$TMBUILD_JOB" \
    "TMsnapshot:$TMSNAP_JOB" \
    "TMsolve const/Pardiso:$TMSOLVE_CONST_CPU" \
    "TMsolve const/CUDSS:$TMSOLVE_CONST_GPU" \
    "TMsolve avg/Pardiso:$TMSOLVE_AVG_CPU" \
    "TMsolve avg/CUDSS:$TMSOLVE_AVG_GPU" \
    "NK const:$NK_CONST" \
    "NK avg:$NK_AVG" \
    "run1yrNK const:$RUNNK_CONST" \
    "run1yrNK avg:$RUNNK_AVG" \
; do
    label="${label_job%%:*}"
    job="${label_job#*:}"
    if [ -n "$job" ]; then
        printf "  %-25s %s\n" "$label" "$job"
        has_any=true
    fi
done
for i in "${!PLOTTM_JOBS[@]}"; do
    printf "  %-25s %s\n" "plotTM ${PLOTTM_LABELS[$i]}" "${PLOTTM_JOBS[$i]}"
    has_any=true
done
for label_job in \
    "plotNK:$PLOTNK_JOB" \
    "plotNKtrace:$PLOTNKTRACE_JOB" \
    "plot1yr:$PLOT1YR_JOB" \
    "plot10yr:$PLOT10YR_JOB" \
    "plot100yr:$PLOT100YR_JOB" \
; do
    label="${label_job%%:*}"
    job="${label_job#*:}"
    if [ -n "$job" ]; then
        printf "  %-25s %s\n" "$label" "$job"
        has_any=true
    fi
done
$has_any || echo "  (no jobs submitted)"
