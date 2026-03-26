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
#   prep grid vel clo run1yr run10yr run100yr runlong
#   TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM
#   plot1yr plot10yr plot100yr
#
# Shortcuts:
#   preprocessing  = prep-grid-vel-clo
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
# Dependency DAG (source of truth: scripts/pipeline.mmd):
#   prep ─┐
#   grid ─┤→ vel ─┐
#          └→ clo ─┤→ run1yr    (afterok vel+clo)
#                   │→ run10yr   (afterok vel+clo, parallel)
#                   │→ run100yr  (afterok vel+clo, parallel)
#                   │→ runlong   (afterok vel+clo, parallel)
#                   │→ TMbuild   (afterok vel+clo) → TMsolve(const) + NK(const) → run1yrNK → plotNK
#                   │                             └→ plotTM (afterok TMbuild + TMsnapshot)
#                   └→ run1yr → TMsnapshot     → TMsolve(avg) + NK(avg) → run1yrNK → plotNK
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

# Require clean git status before submitting jobs (skip in dry-run mode)
if [ "${DRY_RUN:-no}" != "yes" ]; then
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "ERROR: Commit before you submit a job. Working tree is not clean:" >&2
        git status --short >&2
        exit 1
    fi
fi
GIT_COMMIT=$(git rev-parse HEAD)

MODEL_CONF="model_configs/${PARENT_MODEL}.sh"
if [ ! -f "$MODEL_CONF" ]; then
    echo "ERROR: Model config not found: $MODEL_CONF" >&2
    exit 1
fi
source "$MODEL_CONF"
source scripts/submit_job.sh

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
    echo "    prep grid vel clo diagnose_w run1yr run1yrfast run10yr run100yr runlong"
    echo "    TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM"
    echo "    plot1yr plot10yr plot100yr"
    echo ""
    echo "  Shortcuts:"
    echo "    preprocessing  = prep-grid-vel-clo-diagnose_w-partition"
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
ALL_STEPS=(prep grid vel clo diagnose_w partition run1yr run1yrfast run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK run1yrNK plotNK plotNKtrace plotTM plot1yr plot10yr plot100yr)

# --- Dependency DAG (parsed from scripts/pipeline.mmd) ---
declare -A DAG
source scripts/parse_dag.sh

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
JOB_CHAIN="${JOB_CHAIN//preprocessing/prep-grid-vel-clo-diagnose_w-partition}"
JOB_CHAIN="${JOB_CHAIN//standardruns/run1yr-run10yr-run100yr-runlong}"
JOB_CHAIN="${JOB_CHAIN//TMall/TMbuild-TMsnapshot-TMsolve}"
JOB_CHAIN="${JOB_CHAIN//plotall/plot1yr-plot10yr-plot100yr-plotNK-plotTM}"

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

# --- Partition + queue configuration ---
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
GM_REDI=${GM_REDI:-no}
MONTHLY_KAPPAV=${MONTHLY_KAPPAV:-no}
W_FORMULATION=${W_FORMULATION:-wdiagnosed}
PRESCRIBED_W_SOURCE=${PRESCRIBED_W_SOURCE:-parent}
COMMON_VARS="PARENT_MODEL=${PARENT_MODEL}"
COMMON_VARS+=",EXPERIMENT=${EXPERIMENT}"
COMMON_VARS+=",TIME_WINDOW=${TIME_WINDOW}"
COMMON_VARS+=",GIT_COMMIT=${GIT_COMMIT}"
COMMON_VARS+=",GM_REDI=${GM_REDI}"
COMMON_VARS+=",MONTHLY_KAPPAV=${MONTHLY_KAPPAV}"
COMMON_VARS+=",W_FORMULATION=${W_FORMULATION}"
COMMON_VARS+=",PRESCRIBED_W_SOURCE=${PRESCRIBED_W_SOURCE}"

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

# Job ID variables (empty = not submitted)
PREP_JOB="" GRID_JOB="" VEL_JOB="" CLO_JOB="" DIAGW_JOB="" PARTITION_JOB=""
RUN1YR_JOB="" RUN1YRFAST_JOB="" RUN10YR_JOB="" RUN100YR_JOB="" RUNLONG_JOB=""
TMBUILD_JOB="" TMSNAP_JOB=""
TMSOLVE_CONST_CPU="" TMSOLVE_CONST_GPU="" TMSOLVE_AVG_CPU="" TMSOLVE_AVG_GPU=""
NK_CONST="" NK_AVG="" RUNNK_CONST="" RUNNK_AVG=""

# ============================================================
# 1. Preprocessing
# ============================================================

has_step prep && \
    PREP_JOB=$(submit_job prep "$WALLTIME_PREP" \
        scripts/prepreprocessing/periodicaverage.sh)

has_step grid && \
    GRID_JOB=$(submit_job grid "$WALLTIME_GRID" \
        scripts/preprocessing/build_grid.sh)

# vel (depends on: grid + prep)
if has_step vel; then
    deps=""
    [ -n "$GRID_JOB" ] && deps="${deps:+$deps:}$GRID_JOB"
    [ -n "$PREP_JOB" ] && deps="${deps:+$deps:}$PREP_JOB"
    vel_flags=(--deps "$deps")
    PREPROCESS_ARCH=${PREPROCESS_ARCH:-CPU}
    [ "$PREPROCESS_ARCH" = "GPU" ] && vel_flags+=(--gpu)
    VEL_JOB=$(submit_job vel "$WALLTIME_VEL" \
        scripts/preprocessing/build_velocities.sh "${vel_flags[@]}")
fi

# clo (depends on: prep + grid)
if has_step clo; then
    deps=""
    [ -n "$PREP_JOB" ] && deps="${deps:+$deps:}$PREP_JOB"
    [ -n "$GRID_JOB" ] && deps="${deps:+$deps:}$GRID_JOB"
    CLO_JOB=$(submit_job clo "$WALLTIME_VEL" \
        scripts/preprocessing/build_closures.sh --deps "$deps")
fi

VEL_DEP="${VEL_JOB:-${GRID_JOB:-}}"
[ -n "$CLO_JOB" ] && VEL_DEP="${VEL_DEP:+${VEL_DEP}:}$CLO_JOB"

# diagnose_w (depends on: vel, single GPU)
if has_step diagnose_w; then
    deps=""
    [ -n "$VEL_JOB" ] && deps="${deps:+$deps:}$VEL_JOB"
    DIAGW_JOB=$(submit_job diagw "$WALLTIME_RUN_1YEAR" \
        scripts/preprocessing/diagnose_w.sh \
        --gpu-single --deps "$deps")
fi

# Update VEL_DEP to include diagnose_w if it ran
if [ -n "$DIAGW_JOB" ]; then
    VEL_DEP="$DIAGW_JOB"
    [ -n "$CLO_JOB" ] && VEL_DEP="${VEL_DEP}:${CLO_JOB}"
fi

# partition (depends on: vel/diagnose_w + clo + grid, only if multi-rank)
if has_step partition && [[ "$PARTITION" != "1x1" ]]; then
    deps=""
    [ -n "$VEL_JOB" ] && deps="${deps:+$deps:}$VEL_JOB"
    [ -n "$DIAGW_JOB" ] && deps="${deps:+$deps:}$DIAGW_JOB"
    [ -n "$CLO_JOB" ] && deps="${deps:+$deps:}$CLO_JOB"
    [ -n "$GRID_JOB" ] && deps="${deps:+$deps:}$GRID_JOB"
    PARTITION_JOB=$(submit_job partition 00:30:00 \
        scripts/preprocessing/partition_data.sh \
        --cpu --deps "$deps" --vars "PARTITION=${PARTITION}")
fi

# Update dependency: standard runs depend on partition (if it exists) or vel
[ -n "$PARTITION_JOB" ] && VEL_DEP="$PARTITION_JOB"

# ============================================================
# 2. Standard runs (depend on: vel or partition)
# ============================================================

has_step run1yr && \
    RUN1YR_JOB=$(submit_job run1yr "$WALLTIME_RUN_1YEAR" \
        scripts/standard_runs/run_1year.sh \
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION}")

has_step run1yrfast && \
    RUN1YRFAST_JOB=$(submit_job run1yrfast "$WALLTIME_RUN_1YEAR" \
        scripts/standard_runs/run_1year_benchmark.sh \
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION},PROFILE=${PROFILE:-no}")

has_step run10yr && \
    RUN10YR_JOB=$(submit_job run10yr "$WALLTIME_RUN_10YEARS" \
        scripts/standard_runs/run_10years.sh \
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION}")

has_step run100yr && \
    RUN100YR_JOB=$(submit_job run100yr "$WALLTIME_RUN_100YEARS" \
        scripts/standard_runs/run_100years.sh \
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION}")

has_step runlong && \
    RUNLONG_JOB=$(submit_job runlong "$WALLTIME_RUN_LONG" \
        scripts/standard_runs/run_long.sh \
        --gpu --deps "$VEL_DEP" --vars "NYEARS=${NYEARS:-3000},PARTITION=${PARTITION}")

# ============================================================
# 3. Transport matrix building
# ============================================================

has_step TMbuild && \
    TMBUILD_JOB=$(submit_job TMbuild "$WALLTIME_TM_BUILD" \
        scripts/preprocessing/build_TMconst.sh --deps "$VEL_DEP")

has_step TMsnapshot && \
    TMSNAP_JOB=$(submit_job TMsnapshot "$WALLTIME_TM_SNAPSHOT" \
        scripts/preprocessing/build_TMavg.sh --deps "${RUN1YR_JOB:-}")

# ============================================================
# 4. Transport matrix age solving (filtered by TM_SOURCE)
# ============================================================

if has_step TMsolve; then
    if run_const; then
        TMSOLVE_CONST_CPU=$(submit_job TMslv_c "$WALLTIME_TM_SOLVE" \
            scripts/solvers/solve_TM_age_CPU.sh \
            --deps "${TMBUILD_JOB:-}" \
            --vars "TM_SOURCE=const,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY}")

        TMSOLVE_CONST_GPU=$(submit_job TMslv_cG "$WALLTIME_TM_SOLVE" \
            scripts/solvers/solve_TM_age_GPU.sh \
            --gpu-single --deps "${TMBUILD_JOB:-}" \
            --vars "TM_SOURCE=const,LUMP_AND_SPRAY=${LUMP_AND_SPRAY}")
    fi

    if run_avg; then
        TMSOLVE_AVG_CPU=$(submit_job TMslv_a "$WALLTIME_TM_SOLVE" \
            scripts/solvers/solve_TM_age_CPU.sh \
            --deps "${TMSNAP_JOB:-}" \
            --vars "TM_SOURCE=avg,LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY}")

        TMSOLVE_AVG_GPU=$(submit_job TMslv_aG "$WALLTIME_TM_SOLVE" \
            scripts/solvers/solve_TM_age_GPU.sh \
            --gpu-single --deps "${TMSNAP_JOB:-}" \
            --vars "TM_SOURCE=avg,LUMP_AND_SPRAY=${LUMP_AND_SPRAY}")
    fi
fi

# ============================================================
# 5. Newton-Krylov solver (filtered by TM_SOURCE)
# ============================================================

NK_VARS="JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE},PARTITION=${PARTITION}"

if has_step NK; then
    run_const && \
        NK_CONST=$(submit_job NK_c "$WALLTIME_NK" \
            scripts/solvers/solve_periodic_NK.sh \
            --gpu --deps "${TMBUILD_JOB:-}" --vars "TM_SOURCE=const,${NK_VARS}")

    run_avg && \
        NK_AVG=$(submit_job NK_a "$WALLTIME_NK" \
            scripts/solvers/solve_periodic_NK.sh \
            --gpu --deps "${TMSNAP_JOB:-}" --vars "TM_SOURCE=avg,${NK_VARS}")
fi

# ============================================================
# 5b. Re-run 1yr from periodic solution (GPU, depends on NK)
# ============================================================

RUNNK_VARS="LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},PARTITION=${PARTITION}"

if has_step run1yrNK; then
    run_const && \
        RUNNK_CONST=$(submit_job run1yrNK_c "$WALLTIME_RUN_1YEAR" \
            scripts/standard_runs/run_1year_from_periodic_sol.sh \
            --gpu --deps "${NK_CONST:-}" --vars "$RUNNK_VARS")

    run_avg && \
        RUNNK_AVG=$(submit_job run1yrNK_a "$WALLTIME_RUN_1YEAR" \
            scripts/standard_runs/run_1year_from_periodic_sol.sh \
            --gpu --deps "${NK_AVG:-}" --vars "$RUNNK_VARS")
fi

# ============================================================
# 6. Plotting
# ============================================================

# plotTM (depends on: TMbuild + TMsnapshot)
if has_step plotTM; then
    deps=()
    [ -n "${TMBUILD_JOB:-}" ] && deps+=("${TMBUILD_JOB}")
    [ -n "${TMSNAP_JOB:-}" ] && deps+=("${TMSNAP_JOB}")
    plotTM_deps=""
    if [ ${#deps[@]} -gt 0 ]; then
        plotTM_deps=$(IFS=:; echo "${deps[*]}")
    fi
    plotTM_overrides=()
    [ -n "${PLOT_TM_NCPUS:-}" ] && plotTM_overrides+=(--ncpus "${PLOT_TM_NCPUS}")
    [ -n "${PLOT_TM_MEM:-}" ] && plotTM_overrides+=(--mem "${PLOT_TM_MEM}")
    for pair in const:avg; do
        lx="${pair%%:*}"; ly="${pair#*:}"
        submit_job "plotTM_${ly}_vs_${lx}" "$WALLTIME_PLOT" \
            scripts/plotting/plot_TM_datashader.sh \
            --deps "$plotTM_deps" "${plotTM_overrides[@]}" \
            --vars "TM_LABEL_X=${lx},TM_LABEL_Y=${ly}" > /dev/null
    done
fi

has_step plotNK && \
    submit_job plotNK "$WALLTIME_PLOT_NK" \
        scripts/plotting/plot_1year_from_periodic_sol.sh \
        --deps "${RUNNK_CONST:-}" \
        --vars "LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY}" > /dev/null

has_step plotNKtrace && \
    submit_job plotNKtrace "$WALLTIME_PLOT" \
        scripts/plotting/plot_trace_history_job.sh \
        --deps "${NK_CONST:-}" > /dev/null

has_step plot1yr && \
    submit_job plot1yr "$WALLTIME_PLOT" \
        scripts/plotting/plot_standardrun_age.sh \
        --deps "${RUN1YR_JOB:-}" --vars "DURATION=1year" > /dev/null

has_step plot10yr && \
    submit_job plot10yr "$WALLTIME_PLOT" \
        scripts/plotting/plot_standardrun_age.sh \
        --deps "${RUN10YR_JOB:-}" --vars "DURATION=10years" > /dev/null

has_step plot100yr && \
    submit_job plot100yr "$WALLTIME_PLOT" \
        scripts/plotting/plot_standardrun_age.sh \
        --deps "${RUN100YR_JOB:-}" --vars "DURATION=100years" > /dev/null

# ============================================================
# Summary
# ============================================================

print_summary "${PARENT_MODEL} (TM_SOURCE=$TM_SOURCE)"
