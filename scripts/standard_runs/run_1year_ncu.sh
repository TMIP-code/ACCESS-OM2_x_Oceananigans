#!/bin/bash
#
# Nsight Compute (ncu) wrapper around src/run_1year_benchmark.jl.
#
# Wraps each MPI rank's Julia process with `ncu`, filtering to a chosen
# kernel-name regex and capturing a small range of launches so the per-kernel
# replay overhead stays bounded. Outputs one .ncu-rep per rank under the same
# log directory as nsys profiles.
#
# Knobs (all overridable via -v on submission):
#   NCU_SET     basic | roofline | full      (default: roofline)
#   NCU_KERNEL  --kernel-name regex          (default: regex:.*Gc.*)
#   NCU_SKIP    --launch-skip-before-match   (default: 12)
#   NCU_COUNT   --launch-count               (default: 12)
#   BENCHMARK_STEPS  macro-step cap          (default: 24)
#   NCU_RANKS   "0" | "all"                  (default: 0 — only rank 0 wrapped)
#
# Tip: Gc fires once per substep, so for TBLOCKING=K use BENCHMARK_STEPS≥(NCU_SKIP+NCU_COUNT)/K
#      to ensure enough Gc launches exist to skip past warmup and capture COUNT.

#PBS -P y99
#PBS -l mem=256GB
#PBS -q gpuhopper
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99
#PBS -l jobfs=80GB
#PBS -o logs/PBS/
#PBS -e logs/PBS/
#PBS -l wd

set -euo pipefail

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd $repo_root
source scripts/env_defaults.sh

export SYNC_GC_NSTEPS="${SYNC_GC_NSTEPS:-}"
export LOAD_BALANCE="${LOAD_BALANCE:-no}"

job_id="${PBS_JOBID:-interactive}"
run_log_dir=logs/julia/$PARENT_MODEL/$EXPERIMENT/$LOG_TW_TAG/standardrun
mkdir -p "$run_log_dir"
log_file="$run_log_dir/${MODEL_CONFIG}_1yearncu_${job_id}.log"

NGPUS="${PBS_NGPUS:-1}"
JULIA_CMD="julia $JULIA_BOUNDS_FLAG --project"

MPI_BINDING="${MPI_BINDING:-numa}"
case "$MPI_BINDING" in
    numa)   MPI_BIND_FLAGS="--bind-to numa --map-by numa" ;;
    socket) MPI_BIND_FLAGS="--bind-to socket --map-by socket" ;;
    *) echo "ERROR: MPI_BINDING must be numa or socket (got: $MPI_BINDING)" >&2; exit 1 ;;
esac

# ncu knobs
NCU=/apps/cuda/12.9.0/bin/ncu
NCU_SET="${NCU_SET:-roofline}"
NCU_KERNEL="${NCU_KERNEL:-regex:.*Gc.*}"
NCU_SKIP="${NCU_SKIP:-12}"
NCU_COUNT="${NCU_COUNT:-12}"
NCU_RANKS="${NCU_RANKS:-0}"

# Keep the simulation short so warmup + (skip+count) Gc launches fit cheaply.
export BENCHMARK_STEPS="${BENCHMARK_STEPS:-24}"
# Sync GC honors the same default policy as PROFILE=yes runs.
export SYNC_GC_NSTEPS="${SYNC_GC_NSTEPS:-5}"
export JULIA_NVTX_CALLBACKS=gc

profile_base="$run_log_dir/${MODEL_CONFIG}_1yearncu_${job_id}_${NCU_SET}"

NCU_FLAGS=(
    --kernel-name "$NCU_KERNEL"
    --launch-skip-before-match "$NCU_SKIP"
    --launch-count "$NCU_COUNT"
    --set "$NCU_SET"
    --target-processes all
    --replay-mode kernel
    --import-source on
    --force-overwrite
)

echo "ncu wrapper config:"
echo "  NCU=$NCU"
echo "  NCU_SET=$NCU_SET"
echo "  NCU_KERNEL=$NCU_KERNEL"
echo "  NCU_SKIP=$NCU_SKIP  NCU_COUNT=$NCU_COUNT"
echo "  NCU_RANKS=$NCU_RANKS"
echo "  BENCHMARK_STEPS=$BENCHMARK_STEPS"
echo "  output base: $profile_base"
echo "  log: $log_file"

if [ "$NGPUS" -gt 1 ]; then
    # Per-rank dispatch: only wrap selected rank(s) in ncu, others run plain.
    # NCU_RANKS=0 → only rank 0; NCU_RANKS=all → every rank.
    case "$NCU_RANKS" in
        all) cond='true' ;;
        *)   cond='[ "$OMPI_COMM_WORLD_RANK" = "'"$NCU_RANKS"'" ]' ;;
    esac
    mpiexec $MPI_BIND_FLAGS -n "$NGPUS" --report-bindings --display map-devel,bind bash -c "
        if $cond; then
            $NCU ${NCU_FLAGS[*]} \
                --export ${profile_base}_rank\${OMPI_COMM_WORLD_RANK} \
                $JULIA_CMD src/run_1year_benchmark.jl
        else
            $JULIA_CMD src/run_1year_benchmark.jl
        fi
    " &> "$log_file"
else
    $NCU "${NCU_FLAGS[@]}" \
        --export "${profile_base}_rank0" \
        $JULIA_CMD src/run_1year_benchmark.jl &> "$log_file"
fi

echo "Done running run_1year_ncu.sh for PARENT_MODEL=$PARENT_MODEL"
