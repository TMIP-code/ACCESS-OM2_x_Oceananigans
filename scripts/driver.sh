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
#   TMbuild TMsnapshot TMsolve NK run1yrNK ventilation plotNK plotNKtrace plotventilation plotTM
#   plot1yr plot10yr plot100yr plotMOC compareNK
#
# compareNK: cross-resolution NK age comparison — see docs/IAF_NK_age_comparison_plan.md.
#   Standalone (does not consume PARENT_MODEL/EXPERIMENT/TIME_WINDOW).
#   Honors RUN_PHASE1/2/3/3B and REGRID_DIRECTION env vars.
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
#   grid ─┤→ vel → diagnose_w ─┐
#          └→ clo ──────────────┤→ run1yr    (afterok diagw+clo)
#                                │→ run10yr   (afterok diagw+clo, parallel)
#                                │→ run100yr  (afterok diagw+clo, parallel)
#                                │→ runlong   (afterok diagw+clo, parallel)
#                                │→ TMbuild   (afterok diagw+clo) → TMsolve(const) + NK(const) → run1yrNK → plotNK
#                                │                               └→ plotTM (afterok TMbuild + TMsnapshot)
#                                └→ run1yr → TMsnapshot          → TMsolve(avg) + NK(avg) → run1yrNK → plotNK
#
#   plot1yr     (afterok run1yr)
#   plot10yr    (afterok run10yr)
#   plot100yr   (afterok run100yr)
#   plotNKtrace (afterok NK)
#   plotMOC     (afterok prep+grid)

# Pin repo root for any cd-relative reads (model_configs, submit_job, etc.)
repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

# Require clean git status before submitting jobs (skip in dry-run mode).
# Driver-only check — PBS scripts that source env_defaults.sh don't need this.
if [ "${DRY_RUN:-no}" != "yes" ]; then
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "ERROR: Commit before you submit a job. Working tree is not clean:" >&2
        git status --short >&2
        exit 1
    fi
fi
GIT_COMMIT=$(git rev-parse HEAD)
export GIT_COMMIT

# Single source of truth for all defaults: env_defaults.sh handles
# PARENT_MODEL / EXPERIMENT / TIME_WINDOW, sources the per-model config
# (which sets MODEL_SHORT, GPU_QUEUE, PARTITION, TIMESTEP_MULT, LUMP_AND_SPRAY,
# walltimes, etc.), applies cross-model physics defaults, derives MODEL_CONFIG,
# and sources compute_resources.sh for PARTITION_X/Y/RANKS/NGPUS/etc.
# SKIP_MODULES=yes — login node doesn't need cuda/openmpi modules loaded.
SKIP_MODULES=yes source scripts/env_defaults.sh

source scripts/submit_job.sh

# --- JOB_CHAIN: required ---
if [ -z "${JOB_CHAIN:-}" ]; then
    echo "Usage: PARENT_MODEL=... JOB_CHAIN=... bash scripts/driver.sh"
    echo ""
    echo "  PARENT_MODEL  Model to run (default: ACCESS-OM2-1)"
    echo "  EXPERIMENT    Intake catalog key (default: based on PARENT_MODEL)"
    echo "  TIME_WINDOW   Year range YYYY-YYYY or single year (default: 1968-1977)"
    echo "  TM_SOURCE     const (default), avg, or both"
    echo ""
    echo "  Steps:"
    echo "    prep grid vel clo diagnose_w run1yr run1yrfast run1yrncu allocbench allocprofile run10yr run100yr runlong"
    echo "    TMbuild TMsnapshot TMsolve NK run1yrNK ventilation plotNK plotNKtrace plotventilation plotTM"
    echo "    plotgrid plot1yr plot10yr plot100yr plotMOC"
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
ALL_STEPS=(prep grid vel clo diagnose_w partition run1yr run1yrfast run1yrncu allocbench allocprofile run10yr run100yr runlong TMbuild TMsnapshot TMsolve NK run1yrNK ventilation plotgrid plotNK plotNKtrace plotventilation plotTM plot1yr plot10yr plot100yr plotMOC compareNK)

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
JOB_CHAIN="${JOB_CHAIN//plotall/plotgrid-plot1yr-plot10yr-plot100yr-plotNK-plotTM-plotMOC}"

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

# All physics/solver/grid defaults and PARTITION/RANKS resolution are
# now handled by env_defaults.sh (sourced near the top of this script).

# --- TM_SOURCE filtering helpers (use TM_SOURCE set by env_defaults.sh) ---
run_const() { [[ "$TM_SOURCE" == "const" || "$TM_SOURCE" == "both" ]]; }
run_avg() { [[ "$TM_SOURCE" == "avg" || "$TM_SOURCE" == "both" ]]; }
COMMON_VARS="PARENT_MODEL=${PARENT_MODEL}"
COMMON_VARS+=",EXPERIMENT=${EXPERIMENT}"
COMMON_VARS+=",TIME_WINDOW=${TIME_WINDOW}"
[ "$MLD_EXPLICIT" = "yes" ] && COMMON_VARS+=",MLD_TIME_WINDOW=${MLD_TIME_WINDOW}"
COMMON_VARS+=",GIT_COMMIT=${GIT_COMMIT}"
COMMON_VARS+=",VELOCITY_SOURCE=${VELOCITY_SOURCE}"
COMMON_VARS+=",ADVECTION_SCHEME=${ADVECTION_SCHEME}"
COMMON_VARS+=",TIMESTEPPER=${TIMESTEPPER}"
COMMON_VARS+=",TIMESTEP_MULT=${TIMESTEP_MULT}"
COMMON_VARS+=",PLOT_TS=${PLOT_TS:-no}"
COMMON_VARS+=",GM_REDI=${GM_REDI}"
COMMON_VARS+=",MONTHLY_KAPPAV=${MONTHLY_KAPPAV}"
COMMON_VARS+=",IMPLICIT_KAPPAV=${IMPLICIT_KAPPAV}"
COMMON_VARS+=",W_FORMULATION=${W_FORMULATION}"
COMMON_VARS+=",PRESCRIBED_W_SOURCE=${PRESCRIBED_W_SOURCE}"
COMMON_VARS+=",TBLOCKING=${TBLOCKING}"
COMMON_VARS+=",GRID_HX=${GRID_HX}"
COMMON_VARS+=",GRID_HY=${GRID_HY}"
COMMON_VARS+=",GRID_HZ=${GRID_HZ}"
COMMON_VARS+=",LOAD_BALANCE=${LOAD_BALANCE}"
COMMON_VARS+=",ACTIVE_CELLS_MAP=${ACTIVE_CELLS_MAP}"
COMMON_VARS+=",TRAF=${TRAF}"
COMMON_VARS+=",TRAF_TM_SOURCE=${TRAF_TM_SOURCE}"
COMMON_VARS+=",OMEGA=${OMEGA}"
COMMON_VARS+=",MATRIX_PROCESSING=${MATRIX_PROCESSING}"
COMMON_VARS+=",MPI_BINDING=${MPI_BINDING}"

# --- Submission manifest path (TOML, written at exit) ---
# Mirrors outputdir construction in src/shared_utils/config.jl: when no profile
# is configured (current state), outputs land relative to the repo root.
MANIFEST_DIR="${repo_root}/outputs/${PARENT_MODEL}/${EXPERIMENT}/${OUTPUT_TAG}/manifests"
SUBMIT_TS=$(date -Iseconds)
SUBMIT_ID="$(date +%Y%m%dT%H%M%S)_$$"
MANIFEST_PATH="${MANIFEST_DIR}/${SUBMIT_ID}.toml"
export SUBMIT_TS MANIFEST_PATH
mkdir -p "$MANIFEST_DIR"

echo "=== ${PARENT_MODEL} pipeline driver ==="
echo "MODEL_SHORT=$MODEL_SHORT"
echo "EXPERIMENT=$EXPERIMENT"
echo "TIME_WINDOW=$TIME_WINDOW"
echo "MLD_TIME_WINDOW=$MLD_TIME_WINDOW (explicit=$MLD_EXPLICIT)"
echo "OUTPUT_TAG=$OUTPUT_TAG"
echo "MANIFEST_PATH=$MANIFEST_PATH"
echo "JOB_CHAIN=$JOB_CHAIN"
echo "GIT_COMMIT=$GIT_COMMIT"
echo "TM_SOURCE=$TM_SOURCE"
echo "GPU_QUEUE=$GPU_QUEUE, PARTITION=$PARTITION (${PARTITION_X}x${PARTITION_Y}), RANKS=$RANKS, NGPUS=$NGPUS, GPU_NCPUS=$GPU_NCPUS, GPU_MEM=$GPU_MEM"
echo "CPU_QUEUE=$CPU_QUEUE, CPU_NCPUS=$CPU_NCPUS, CPU_MEM=$CPU_MEM"
echo "JVP_METHOD=$JVP_METHOD, LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY, INITIAL_AGE=$INITIAL_AGE"
echo "TBLOCKING=$TBLOCKING, GRID halos=(${GRID_HX},${GRID_HY},${GRID_HZ}), OMEGA=$OMEGA"
echo ""

# Guard: OM2-01 cannot fit a single monthly FTS on one H200 (each ~50 GB, GPU has 140 GB).
if [ "$PARENT_MODEL" = "ACCESS-OM2-01" ] && [ "$NGPUS" -lt 2 ]; then
    for s in diagnose_w run1yr run1yrfast run1yrncu allocbench allocprofile run10yr run100yr runlong NK run1yrNK; do
        has_step "$s" && { echo "ERROR: ACCESS-OM2-01 requires NGPUS>=2 (each monthly FTS ~50GB on 140GB H200). Use GPU_RESOURCES=gpuhopper-1x4 or larger. Blocking step: $s" >&2; exit 1; }
    done
fi

# Job ID variables (empty = not submitted; can be pre-set via env vars to chain
# downstream steps onto jobs submitted by an earlier driver invocation, e.g.
# `GRID_JOB=12345.gadi-pbs VEL_JOB=12346.gadi-pbs CLO_JOB=12347.gadi-pbs \
#  PARTITION=1x4 JOB_CHAIN=partition-run1yr bash scripts/driver.sh`)
PREP_JOB="${PREP_JOB:-}" GRID_JOB="${GRID_JOB:-}" VEL_JOB="${VEL_JOB:-}" CLO_JOB="${CLO_JOB:-}" DIAGW_JOB="${DIAGW_JOB:-}" PARTITION_JOB="${PARTITION_JOB:-}"
RUN1YR_JOB="" RUN1YRFAST_JOB="" RUN10YR_JOB="" RUN100YR_JOB="" RUNLONG_JOB=""
TMBUILD_JOB="" TMSNAP_JOB=""
TMSOLVE_CONST_CPU="" TMSOLVE_CONST_GPU="" TMSOLVE_AVG_CPU="" TMSOLVE_AVG_GPU=""
NK_CONST="${NK_CONST:-}" NK_AVG="${NK_AVG:-}" RUNNK_CONST="${RUNNK_CONST:-}" RUNNK_AVG="${RUNNK_AVG:-}" VENT_CONST="${VENT_CONST:-}" VENT_AVG="${VENT_AVG:-}"

# ============================================================
# 1. Preprocessing
# ============================================================

if has_step prep; then
    prep_flags=(--ncpus "${PREP_NCPUS:-48}" --mem "${PREP_MEM:-192GB}")
    [ -n "${PREP_QUEUE:-}" ] && prep_flags+=(--queue "${PREP_QUEUE}")
    PREP_JOB=$(submit_job prep "$WALLTIME_PREP" \
        scripts/prepreprocessing/periodicaverage.sh \
        "${prep_flags[@]}")
fi

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
    [ -n "${VEL_NCPUS:-}" ] && vel_flags+=(--ncpus "${VEL_NCPUS}")
    [ -n "${VEL_QUEUE:-}" ] && vel_flags+=(--queue "${VEL_QUEUE}")
    [ -n "${VEL_MEM:-}" ]   && vel_flags+=(--mem "${VEL_MEM}")
    VEL_JOB=$(submit_job vel "$WALLTIME_VEL" \
        scripts/preprocessing/build_velocities.sh "${vel_flags[@]}")
fi

# clo (depends on: prep + grid)
if has_step clo; then
    deps=""
    [ -n "$PREP_JOB" ] && deps="${deps:+$deps:}$PREP_JOB"
    [ -n "$GRID_JOB" ] && deps="${deps:+$deps:}$GRID_JOB"
    clo_flags=(--deps "$deps")
    [ -n "${CLO_NCPUS:-}" ] && clo_flags+=(--ncpus "${CLO_NCPUS}")
    [ -n "${CLO_QUEUE:-}" ] && clo_flags+=(--queue "${CLO_QUEUE}")
    [ -n "${CLO_MEM:-}" ]   && clo_flags+=(--mem "${CLO_MEM}")
    CLO_JOB=$(submit_job clo "${WALLTIME_CLO:-$WALLTIME_VEL}" \
        scripts/preprocessing/build_closures.sh "${clo_flags[@]}")
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
    # If model config sets PARTITION_MEM_PER_RANK and PARTITION_MEM is unset,
    # compute mem = RANKS × per-rank. Bump ncpus to satisfy queue's mem/cpu ratio.
    # PARTITION_QUEUE may differ from CPU_QUEUE (e.g., OM2-01 uses megamem just
    # for partition).
    if [ -z "${PARTITION_MEM:-}" ] && [ -n "${PARTITION_MEM_PER_RANK:-}" ]; then
        # MEM_PER_CPU and queue minimums depend on PARTITION_QUEUE (falls back to CPU_QUEUE)
        case "${PARTITION_QUEUE:-$CPU_QUEUE}" in
            express|normal) part_mem_per_cpu=4;  part_mem_min=0 ;;
            hugemem)        part_mem_per_cpu=32; part_mem_min=192 ;;
            megamem)        part_mem_per_cpu=64; part_mem_min=1000 ;;
            *)              part_mem_per_cpu=$MEM_PER_CPU; part_mem_min=0 ;;
        esac
        part_mem_gb=$(( RANKS * PARTITION_MEM_PER_RANK ))
        [ "$part_mem_gb" -lt "$part_mem_min" ] && part_mem_gb=$part_mem_min
        PARTITION_MEM="${part_mem_gb}GB"
        min_ncpus_for_mem=$(( part_mem_gb / part_mem_per_cpu ))
        PARTITION_NCPUS=${PARTITION_NCPUS:-$(( min_ncpus_for_mem > RANKS ? min_ncpus_for_mem : RANKS ))}
    fi
    partition_flags=(--cpu --deps "$deps" --vars "PARTITION=${PARTITION}")
    [ -n "${PARTITION_QUEUE:-}" ] && partition_flags+=(--queue "${PARTITION_QUEUE}")
    [ -n "${PARTITION_NCPUS:-}" ] && partition_flags+=(--ncpus "${PARTITION_NCPUS}")
    [ -n "${PARTITION_MEM:-}" ] && partition_flags+=(--mem "${PARTITION_MEM}")
    PARTITION_JOB=$(submit_job partition "${PARTITION_WALLTIME:-00:30:00}" \
        scripts/preprocessing/partition_data.sh \
        "${partition_flags[@]}")
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
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION},PROFILE=${PROFILE:-no},SYNC_GC_NSTEPS=${SYNC_GC_NSTEPS:-},BENCHMARK_STEPS=${BENCHMARK_STEPS:-}")

has_step run1yrncu && \
    RUN1YRNCU_JOB=$(submit_job run1yrncu "${WALLTIME_RUN_NCU:-01:00:00}" \
        scripts/standard_runs/run_1year_ncu.sh \
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION},BENCHMARK_STEPS=${BENCHMARK_STEPS:-},SYNC_GC_NSTEPS=${SYNC_GC_NSTEPS:-},NCU_SET=${NCU_SET:-},NCU_KERNEL=${NCU_KERNEL:-},NCU_SKIP=${NCU_SKIP:-},NCU_COUNT=${NCU_COUNT:-},NCU_RANKS=${NCU_RANKS:-}")

has_step allocbench && \
    submit_job allocbench "$WALLTIME_RUN_1YEAR" \
        scripts/standard_runs/run_alloc_benchmark.sh \
        --gpu --deps "$VEL_DEP" --vars "PARTITION=${PARTITION},ALLOC_BATCH_STEPS=${ALLOC_BATCH_STEPS:-20}" > /dev/null

has_step allocprofile && \
    submit_job allocprofile 01:00:00 \
        scripts/standard_runs/run_1year_alloc_profile.sh \
        --gpu --deps "$VEL_DEP" \
        --vars "PARTITION=${PARTITION},ALLOC_SAMPLE_RATE=${ALLOC_SAMPLE_RATE:-0.01},ALLOC_PROFILE_STEPS=${ALLOC_PROFILE_STEPS:-3}" > /dev/null

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

if has_step TMbuild; then
    # Base deps for TMbuild come from the preprocessing chain. Additionally, an
    # external job ID can be passed via UPSTREAM_TMBUILD_JOB to make this
    # TMbuild afterok-depend on it — this is how TRAF chains chain onto the
    # corresponding IAF TMbuild from a different driver invocation (TRAF needs
    # the forward M.jld2 to exist before its algebraic invVMtV synthesis can
    # run; see docs/TRAF_simulations.md). Skipped silently when unset, so IAF
    # invocations behave identically.
    TMBUILD_DEPS="$VEL_DEP"
    [ -n "${UPSTREAM_TMBUILD_JOB:-}" ] && \
        TMBUILD_DEPS="${UPSTREAM_TMBUILD_JOB}${TMBUILD_DEPS:+:}${TMBUILD_DEPS}"
    tmbuild_flags=(--deps "$TMBUILD_DEPS")
    [ -n "${TMBUILD_QUEUE:-}" ] && tmbuild_flags+=(--queue "${TMBUILD_QUEUE}")
    [ -n "${TMBUILD_NCPUS:-}" ] && tmbuild_flags+=(--ncpus "${TMBUILD_NCPUS}")
    [ -n "${TMBUILD_MEM:-}" ]   && tmbuild_flags+=(--mem "${TMBUILD_MEM}")
    TMBUILD_JOB=$(submit_job TMbuild "$WALLTIME_TM_BUILD" \
        scripts/preprocessing/build_TMconst.sh "${tmbuild_flags[@]}")
fi

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

NK_VARS="JVP_METHOD=${JVP_METHOD},LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},INITIAL_AGE=${INITIAL_AGE},PARTITION=${PARTITION},TM_MODEL_CONFIG=${TM_MODEL_CONFIG:-}"

if has_step NK; then
    # NK depends on TMbuild only (the preconditioner matrix). The previous
    # TMsolve dep was required by the old INITIAL_AGE=TMage default — now NK
    # defaults to INITIAL_AGE=0 (or `latest` / explicit path), so the TM age
    # file is not needed at submit time.
    NK_CONST_DEPS="${TMBUILD_JOB:-}"
    NK_AVG_DEPS="${TMSNAP_JOB:-}"

    run_const && \
        NK_CONST=$(submit_job NK_c "$WALLTIME_NK" \
            scripts/solvers/solve_periodic_NK.sh \
            --gpu --deps "$NK_CONST_DEPS" --vars "TM_SOURCE=const,${NK_VARS}")

    run_avg && \
        NK_AVG=$(submit_job NK_a "$WALLTIME_NK" \
            scripts/solvers/solve_periodic_NK.sh \
            --gpu --deps "$NK_AVG_DEPS" --vars "TM_SOURCE=avg,${NK_VARS}")
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
# 5c. Surface ventilation diagnostic (CPU, depends on NK)
# ============================================================

VENT_VARS="LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},PARTITION=${PARTITION}"

if has_step ventilation; then
    run_const && \
        VENT_CONST=$(submit_job ventilation_c "${WALLTIME_VENTILATION:-00:30:00}" \
            scripts/solvers/compute_ventilation.sh \
            --deps "${RUNNK_CONST:-${NK_CONST:-}}" --vars "TM_SOURCE=const,${VENT_VARS}")

    run_avg && \
        VENT_AVG=$(submit_job ventilation_a "${WALLTIME_VENTILATION:-00:30:00}" \
            scripts/solvers/compute_ventilation.sh \
            --deps "${RUNNK_AVG:-${NK_AVG:-}}" --vars "TM_SOURCE=avg,${VENT_VARS}")
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

if has_step plotNK; then
    plotNK_overrides=()
    [ -n "${PLOT_NK_NCPUS:-}" ] && plotNK_overrides+=(--ncpus "$PLOT_NK_NCPUS")
    [ -n "${PLOT_NK_MEM:-}" ] && plotNK_overrides+=(--mem "$PLOT_NK_MEM")
    submit_job plotNK "$WALLTIME_PLOT_NK" \
        scripts/plotting/plot_1year_from_periodic_sol.sh \
        --deps "${RUNNK_CONST:-${NK_CONST:-}}" "${plotNK_overrides[@]}" \
        --vars "LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY}" > /dev/null
fi

has_step plotNKtrace && \
    submit_job plotNKtrace "$WALLTIME_PLOT" \
        scripts/plotting/plot_trace_history_job.sh \
        --deps "${NK_CONST:-}" > /dev/null

if has_step plotventilation; then
    plotvent_overrides=()
    [ -n "${PLOT_VENT_NCPUS:-}" ] && plotvent_overrides+=(--ncpus "$PLOT_VENT_NCPUS")
    [ -n "${PLOT_VENT_MEM:-}" ] && plotvent_overrides+=(--mem "$PLOT_VENT_MEM")
    submit_job plotventilation "${WALLTIME_PLOT_VENTILATION:-00:30:00}" \
        scripts/plotting/plot_ventilation.sh \
        --deps "${VENT_CONST:-${NK_CONST:-}}" "${plotvent_overrides[@]}" \
        --vars "LINEAR_SOLVER=${LINEAR_SOLVER},LUMP_AND_SPRAY=${LUMP_AND_SPRAY},PARTITION=${PARTITION}" > /dev/null
fi

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

# plotgrid (depends on: grid)
has_step plotgrid && \
    submit_job plotgrid "$WALLTIME_PLOT" \
        scripts/plotting/plot_grid_metrics.sh \
        --deps "${GRID_JOB:-}" > /dev/null

# compareNK: cross-resolution NK age comparison
# Standalone step — reads all 4 (PM × TW) periodic-NK FTS files that must
# already exist. Independent of the current PARENT_MODEL/EXPERIMENT/TIME_WINDOW.
if has_step compareNK; then
    compare_vars="RUN_PHASE1=${RUN_PHASE1:-yes},RUN_PHASE2=${RUN_PHASE2:-yes}"
    compare_vars+=",RUN_PHASE3=${RUN_PHASE3:-yes},RUN_PHASE3B=${RUN_PHASE3B:-no}"
    compare_vars+=",REGRID_DIRECTION=${REGRID_DIRECTION:-fine2coarse}"
    [ -n "${MC_OM2_1:-}"   ] && compare_vars+=",MC_OM2_1=${MC_OM2_1}"
    [ -n "${MC_OM2_025:-}" ] && compare_vars+=",MC_OM2_025=${MC_OM2_025}"
    submit_job compareNK "${WALLTIME_COMPARE_NK:-03:00:00}" \
        scripts/plotting/compare_NK_ages.sh \
        --vars "$compare_vars" > /dev/null
fi

# plotMOC (depends on: prep + grid — needs ty_trans_monthly.nc and grid.jld2)
if has_step plotMOC; then
    plotMOC_deps=()
    [ -n "${PREP_JOB:-}" ] && plotMOC_deps+=("${PREP_JOB}")
    [ -n "${GRID_JOB:-}" ] && plotMOC_deps+=("${GRID_JOB}")
    plotMOC_dep_str=""
    if [ ${#plotMOC_deps[@]} -gt 0 ]; then
        plotMOC_dep_str=$(IFS=:; echo "${plotMOC_deps[*]}")
    fi
    submit_job plotMOC "$WALLTIME_PLOT" \
        scripts/plotting/plot_MOC.sh \
        --deps "$plotMOC_dep_str" > /dev/null
fi

# ============================================================
# Summary
# ============================================================

print_summary "${PARENT_MODEL} (TM_SOURCE=$TM_SOURCE)"

# ============================================================
# Manifest (TOML, dropped next to outputs)
# ============================================================

write_manifest() {
    local n_submitted
    n_submitted=$(wc -l < "$_SUBMIT_LOG" 2>/dev/null || echo 0)
    [ "$n_submitted" -eq 0 ] && { echo "(no jobs submitted; skipping manifest)" >&2; return 0; }

    # Git info — driver requires clean tree before submission, but DRY_RUN may bypass.
    local git_branch git_dirty
    git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
        git_dirty="true"
    else
        git_dirty="false"
    fi

    local host user case_file_q
    host=$(hostname)
    user=${USER:-unknown}
    case_file_q="${CASE_FILE:-}"

    {
        echo "# auto-written by scripts/driver.sh @ submit time"
        echo "[meta]"
        echo "timestamp = \"${SUBMIT_TS}\""
        echo "submit_id = \"${SUBMIT_ID}\""
        echo "user      = \"${user}\""
        echo "host      = \"${host}\""
        echo "case_file = \"${case_file_q}\""
        echo ""
        echo "[git]"
        echo "commit = \"${GIT_COMMIT}\""
        echo "branch = \"${git_branch}\""
        echo "dirty  = ${git_dirty}"
        echo ""
        echo "[env]"
        # Reflect COMMON_VARS as TOML key/value pairs.
        local IFS_save="$IFS"
        IFS=','
        for kv in $COMMON_VARS; do
            local k="${kv%%=*}"
            local v="${kv#*=}"
            # Skip empty values to keep manifest tidy.
            [ -z "$v" ] && continue
            echo "${k} = \"${v}\""
        done
        IFS="$IFS_save"
        # JOB_CHAIN isn't in COMMON_VARS, but it's the most useful piece of intent.
        echo "JOB_CHAIN = \"${JOB_CHAIN}\""
        echo "TM_SOURCE = \"${TM_SOURCE}\""
        echo "JVP_METHOD = \"${JVP_METHOD}\""
        echo "LINEAR_SOLVER = \"${LINEAR_SOLVER}\""
        echo "LUMP_AND_SPRAY = \"${LUMP_AND_SPRAY}\""
        echo "INITIAL_AGE = \"${INITIAL_AGE}\""
        echo "PARTITION = \"${PARTITION}\""
        echo "GPU_QUEUE = \"${GPU_QUEUE}\""
        echo ""
        # One [[jobs]] block per submitted PBS job.
        while IFS='|' read -r step jobid deps script; do
            echo "[[jobs]]"
            echo "step   = \"${step}\""
            echo "jobid  = \"${jobid}\""
            echo "script = \"${script}\""
            if [ -z "$deps" ]; then
                echo "deps   = []"
            else
                # Convert "a:b:c" → ["a","b","c"]
                local deps_quoted
                deps_quoted=$(echo "$deps" | awk -F: '{out=""; for(i=1;i<=NF;i++){out=out"\""$i"\""; if(i<NF) out=out","} print out}')
                echo "deps   = [${deps_quoted}]"
            fi
            echo ""
        done < "$_SUBMIT_LOG"
    } > "$MANIFEST_PATH"

    echo "" >&2
    echo "Manifest written: $MANIFEST_PATH" >&2
    echo "Index updated:    $SUBMISSIONS_TSV" >&2
}

write_manifest
