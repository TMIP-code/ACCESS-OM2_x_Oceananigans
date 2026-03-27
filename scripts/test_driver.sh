#!/bin/bash
# Unified test driver for ACCESS-OM2_x_Oceananigans.
#
# Usage (from login node):
#   GPU_QUEUE=gpuvolta PARTITION=2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill bash scripts/test_driver.sh
#   PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diag bash scripts/test_driver.sh
#   GPU_QUEUE=gpuvolta PARTITION=2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill-diag-mpi bash scripts/test_driver.sh
#
# Available test steps (dash-separated in JOB_CHAIN):
#   halofill  — fill_halo_regions! MWE at all staggered locations (distributed GPU)
#   halofillcpu — same MWE on 4 CPU ranks (no GPUs, express queue)
#   jld2      — JLD2Writer deadlock MWE on 2 CPU ranks (CliMA/Oceananigans.jl#5410)
#   diag      — 10-step diagnostic run saving every step (serial or distributed GPU)
#   diagcpu   — 10-step diagnostic on CPU (distributed MPI, no GPUs, express queue)
#   diagcpuserial — 10-step diagnostic on CPU (serial, no GPUs, express queue)
#   compare   — compare serial vs distributed outputs (CPU, express queue)
#               set DURATION_TAG=diag or DURATION_TAG=1year (default: diag)
#   mpi       — MPI smoke test (rank/device info, 10-iteration simulation)
#   prediagw    — compare 1-year age from wdiagnosed vs wprescribed (parent & prediag)
#   prediagwNK  — compare NK periodic age from wdiagnosed vs wprescribed (parent & prediag)
#   mkappaVNK   — compare NK periodic age: yearly vs monthly kappaV

set -euo pipefail

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

source scripts/env_defaults.sh
source scripts/submit_job.sh

# --- Validate JOB_CHAIN ---
JOB_CHAIN=${JOB_CHAIN:-}
if [[ -z "$JOB_CHAIN" ]]; then
    echo "Usage: JOB_CHAIN=<step[-step...]> [GPU_QUEUE=...] [PARTITION=...] [PARENT_MODEL=...] bash scripts/test_driver.sh"
    echo ""
    echo "Available test steps: halofill halofillcpu jld2 diag diagcpu diagcpuserial compare gridtest mpi prediagw prediagwNK mkappaVNK"
    echo ""
    echo "Examples:"
    echo "  GPU_QUEUE=gpuvolta PARTITION=2x2 PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=halofill bash scripts/test_driver.sh"
    echo "  PARENT_MODEL=ACCESS-OM2-1 JOB_CHAIN=diag bash scripts/test_driver.sh"
    echo "  GPU_QUEUE=gpuvolta PARTITION=2x2 JOB_CHAIN=halofill-diag-mpi bash scripts/test_driver.sh"
    exit 1
fi

has_step() { [[ "-${JOB_CHAIN}-" == *"-$1-"* ]]; }

COMMON_VARS="PARENT_MODEL=${PARENT_MODEL},GIT_COMMIT=${GIT_COMMIT}"
WALLTIME=00:30:00

echo "=== ${PARENT_MODEL} test driver ==="
echo "MODEL_SHORT=$MODEL_SHORT"
echo "JOB_CHAIN=$JOB_CHAIN"
echo "GIT_COMMIT=$GIT_COMMIT"
echo "GPU_QUEUE=$GPU_QUEUE, PARTITION=$PARTITION (${PARTITION_X}x${PARTITION_Y}), RANKS=$RANKS, NGPUS=$NGPUS, GPU_MEM=$GPU_MEM"
echo ""

# --- Job submissions ---

has_step halofill && \
    submit_job halofill "$WALLTIME" scripts/tests/run_halofill_test.sh \
        --gpu --vars "GPU_QUEUE=${GPU_QUEUE},PARTITION=${PARTITION}" > /dev/null

has_step halofillcpu && \
    submit_job halofillcpu "$WALLTIME" scripts/tests/run_halofill_test.sh \
        --queue express --ngpus 0 --ncpus 4 --mem 47GB > /dev/null

has_step jld2 && \
    submit_job jld2 "$WALLTIME" scripts/tests/run_jld2writer_test.sh \
        --queue express --ngpus 0 --ncpus 2 --mem 16GB > /dev/null

has_step diag && \
    submit_job diag "$WALLTIME" scripts/tests/run_diagnostic_steps.sh \
        --gpu --vars "GPU_QUEUE=${GPU_QUEUE},PARTITION=${PARTITION}" > /dev/null

has_step diagcpu && \
    submit_job diagcpu 00:30:00 scripts/tests/run_diagnostic_steps.sh \
        --queue express --ngpus 0 --ncpus 4 --mem 47GB \
        --vars "PARTITION=${PARTITION}" > /dev/null

has_step diagcpuserial && \
    submit_job diagcpuserial 00:30:00 scripts/tests/run_diagnostic_steps.sh \
        --queue express --ngpus 0 --ncpus 1 --mem 47GB > /dev/null

if has_step compare; then
    DURATION_TAG=${DURATION_TAG:-diag}
    GPU_TAG="${PARTITION_X}x${PARTITION_Y}"
    submit_job compare 01:00:00 scripts/plotting/compare_runs_across_architectures.sh \
        --queue express --ngpus 0 --ncpus 12 --mem 47GB \
        --vars "GPU_TAG=${GPU_TAG},DURATION_TAG=${DURATION_TAG}" > /dev/null
fi

has_step gridtest && \
    submit_job gridtest 00:30:00 scripts/tests/run_grid_identity_test.sh \
        --queue express --ngpus 0 --ncpus 4 --mem 47GB > /dev/null

has_step mpi && \
    submit_job mpi "$WALLTIME" scripts/tests/run_mpi_test.sh \
        --gpu --vars "GPU_QUEUE=${GPU_QUEUE},PARTITION=${PARTITION}" > /dev/null

# Helper: build MODEL_CONFIG for a given w-formulation tag
_wmc() {
    local wf=$1 suffix=""
    [ "$GM_REDI" = "yes" ] && suffix="${suffix}_GMREDI"
    [ "$MONTHLY_KAPPAV" = "yes" ] && suffix="${suffix}_mkappaV"
    echo "${VELOCITY_SOURCE}_${wf}_${ADVECTION_SCHEME}_${TIMESTEPPER}${suffix}"
}

# prediagw: serial 1-year w-formulation comparison (2 pairs)
if has_step prediagw; then
    DIAG=$(_wmc wdiagnosed); PARENT=$(_wmc wparent); PREDIAG=$(_wmc wprediag)
    DUR=${DURATION_TAG:-1year}
    submit_job prediagw_dp 01:00:00 scripts/plotting/compare_runs.sh \
        --queue express --ngpus 0 --ncpus 12 --mem 47GB \
        --vars "SOURCE_A=serial:${DIAG}:${DUR},SOURCE_B=serial:${PARENT}:${DUR},COMPARE_LABEL=w_serial_diag_vs_parent" > /dev/null
    submit_job prediagw_dd 01:00:00 scripts/plotting/compare_runs.sh \
        --queue express --ngpus 0 --ncpus 12 --mem 47GB \
        --vars "SOURCE_A=serial:${DIAG}:${DUR},SOURCE_B=serial:${PREDIAG}:${DUR},COMPARE_LABEL=w_serial_diag_vs_prediag" > /dev/null
fi

# prediagwNK: NK periodic w-formulation comparison (2 pairs)
if has_step prediagwNK; then
    DIAG=$(_wmc wdiagnosed); PARENT=$(_wmc wparent); PREDIAG=$(_wmc wprediag)
    STAG="${LINEAR_SOLVER}_$([ "$LUMP_AND_SPRAY" = "yes" ] && echo LSprec || echo prec)"
    submit_job prediagwNK_dp 01:00:00 scripts/plotting/compare_runs.sh \
        --queue express --ngpus 0 --ncpus 12 --mem 47GB \
        --vars "SOURCE_A=NK:${DIAG}:${STAG},SOURCE_B=NK:${PARENT}:${STAG},COMPARE_LABEL=w_NK_diag_vs_parent" > /dev/null
    submit_job prediagwNK_dd 01:00:00 scripts/plotting/compare_runs.sh \
        --queue express --ngpus 0 --ncpus 12 --mem 47GB \
        --vars "SOURCE_A=NK:${DIAG}:${STAG},SOURCE_B=NK:${PREDIAG}:${STAG},COMPARE_LABEL=w_NK_diag_vs_prediag" > /dev/null
fi

# mkappaVNK: compare NK periodic age with yearly vs monthly kappaV
if has_step mkappaVNK; then
    BASE="${VELOCITY_SOURCE}_${W_FORMULATION}_${ADVECTION_SCHEME}_${TIMESTEPPER}"
    MKAPPAV="${BASE}_mkappaV"
    STAG="${LINEAR_SOLVER}_$([ "$LUMP_AND_SPRAY" = "yes" ] && echo LSprec || echo prec)"
    submit_job mkappaVNK 01:00:00 scripts/plotting/compare_runs.sh \
        --queue express --ngpus 0 --ncpus 12 --mem 47GB \
        --vars "SOURCE_A=NK:${BASE}:${STAG},SOURCE_B=NK:${MKAPPAV}:${STAG},COMPARE_LABEL=NK_yearly_vs_monthly_kappaV" > /dev/null
fi

# --- Summary ---
print_summary "${PARENT_MODEL}"
