"""
Benchmark the NK preconditioner's coarsened linear solve as a function of the
lump-and-spray coarsening factor (LUMP_AND_SPRAY).

This reproduces the *exact* preconditioner matrix Q built by `solve_periodic_NK.jl`
(`src/solve_periodic_NK.jl:173-223`) — coarsen the **raw** transport matrix M, scale
by `stop_time`, then `process_sparse_matrix` — and measures how long the factorization
and solve take and how much memory they use. It does NOT run a Newton solve; it only
factorizes Q and solves against representative right-hand sides.

Run one job per LUMP_AND_SPRAY value (5x5, 4x4, 3x3) so PBS `resources_used.mem`
cleanly attributes the peak memory to each coarsening factor.

CPU only — no GPU required.

Usage — interactive:
```
qsub -I -P y99 -q hugemem -l ncpus=48 -l mem=1470GB -l walltime=06:00:00 \\
     -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/benchmark_precond_solve.jl")
```

Environment variables:
  PARENT_MODEL      – model resolution tag  (e.g. ACCESS-OM2-01)
  MODEL_CONFIG      – config tag locating the TM dir (set by env_defaults.sh)
  LINEAR_SOLVER     – Pardiso | ParU | UMFPACK  (default: Pardiso)
  LUMP_AND_SPRAY    – AxB  (the swept value, e.g. 5x5, 4x4, 3x3)
  MATRIX_PROCESSING – raw | symfill | dropzeros | symdrop  (default: symdrop;
                      Pardiso REAL_SYM requires structural symmetry → symdrop/symfill)
  TM_SOURCE         – const | avg  (default: const)
  PARDISO_NPROCS    – Pardiso threads (default: PBS_NCPUS, else 12)
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Grids: on_architecture, znodes
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using KernelAbstractions: @kernel, @index
using LinearAlgebra
using SparseArrays
using Statistics
using JLD2
using Printf
using OceanTransportMatrixBuilder
import Pardiso
import ParU_jll
using LinearSolve
using TOML

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

(; parentmodel, experiment_dir, outputdir) = load_project_config()

year = years = 365.25 * 86400  # seconds

const nprocs = parse(Int, get(ENV, "PARDISO_NPROCS", get(ENV, "PBS_NCPUS", "12")))

model_config = require_env("MODEL_CONFIG")
git_commit = get(ENV, "GIT_COMMIT", "unknown")

LINEAR_SOLVER = require_env("LINEAR_SOLVER")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) || error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

MATRIX_PROCESSING = require_env("MATRIX_PROCESSING")
(MATRIX_PROCESSING ∈ ("raw", "symfill", "dropzeros", "symdrop")) || error("MATRIX_PROCESSING must be one of: raw, symfill, dropzeros, symdrop (got: $MATRIX_PROCESSING)")

ls = parse_lump_and_spray()
LUMP_AND_SPRAY = ls.on
coarse_tag = ls.on ? ls.tag : "full"

TM_SOURCE = require_env("TM_SOURCE")
(TM_SOURCE ∈ ("const", "avg")) || error("TM_SOURCE must be one of: const, avg (got: $TM_SOURCE)")

matrices_dir = joinpath(outputdir, "TM", model_config)
M_dir = joinpath(matrices_dir, TM_SOURCE)
bench_dir = joinpath(M_dir, "benchmarks")
mkpath(bench_dir)
bench_tag = "precond_solve_$(coarse_tag)_$(LINEAR_SOLVER)_$(MATRIX_PROCESSING)"

@info "Preconditioner-solve benchmark configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- LINEAR_SOLVER     = $LINEAR_SOLVER"
@info "- MATRIX_PROCESSING = $MATRIX_PROCESSING"
@info "- LUMP_AND_SPRAY    = $LUMP_AND_SPRAY (di=$(ls.di), dj=$(ls.dj), dk=$(ls.dk), tag: $coarse_tag)"
@info "- TM_SOURCE         = $TM_SOURCE"
@info "- nprocs (Pardiso)  = $nprocs"
@info "- model_config      = $model_config"
@info "- M_dir             = $M_dir"
@info "- bench_dir         = $bench_dir"
flush(stdout); flush(stderr)

################################################################################
# Load grid, wet mask, cell volumes
################################################################################

@info "Loading grid"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
@info "Grid loaded: $(size(grid))"

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
v1D = interior(compute_volume(grid))[idx]
@info "Number of wet cells (fine): $Nidx"
flush(stdout); flush(stderr)

################################################################################
# Load transport matrix M (raw — matches NK preconditioner build order)
################################################################################

M_file = joinpath(M_dir, "M.jld2")
@info "Loading transport matrix from $M_file"
flush(stdout); flush(stderr)
M = load(M_file, "M")
@info "Loaded M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"
@assert Nidx == size(M, 1) "Mismatch: wet cells ($Nidx) ≠ matrix rows ($(size(M, 1)))"
flush(stdout); flush(stderr)

################################################################################
# Build Q exactly as solve_periodic_NK.jl:173-197
#   coarsen RAW M → Mc = LUMP*M*SPRAY → Q = stop_time*Mc → process_sparse_matrix
# (stop_time is a uniform scalar on the nonzeros: cosmetic for factorization cost
#  and memory, applied only for numerical parity with the NK preconditioner.)
################################################################################

stop_time = year

if LUMP_AND_SPRAY
    @info "Computing LUMP and SPRAY matrices (di=$(ls.di), dj=$(ls.dj), dk=$(ls.dk))"
    flush(stdout); flush(stderr)
    LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(
        wet3D, v1D, M; di = ls.di, dj = ls.dj, dk = ls.dk,
    )
    Mc = LUMP * M * SPRAY
    @info "Coarsened Jacobian Mc: $(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc))"
    Q = copy(Mc)
    Q.nzval .*= stop_time
else
    @info "LUMP_AND_SPRAY=no — using full Q = stop_time * M (no coarsening)"
    LUMP = I
    SPRAY = I
    Mc = M
    Q = copy(M)
    Q.nzval .*= stop_time
end
flush(stdout); flush(stderr)

@info "Processing Q (MATRIX_PROCESSING=$MATRIX_PROCESSING)"
flush(stdout); flush(stderr)
Q = process_sparse_matrix(Q, MATRIX_PROCESSING)

n_coarse = size(Q, 1)
nnz_Mc = nnz(Mc)
nnz_Q = nnz(Q)
@info "Q: $(n_coarse)×$(size(Q, 2)), nnz=$nnz_Q"
flush(stdout); flush(stderr)

################################################################################
# Linear solver setup (matches solve_periodic_NK.jl:202-217)
################################################################################

@info "Setting up linear solver (LINEAR_SOLVER=$LINEAR_SOLVER)"
flush(stdout); flush(stderr)

if LINEAR_SOLVER == "Pardiso"
    Pardiso.isstructurallysymmetric(Q) || error(
        "Q is not structurally symmetric (nnz=$nnz_Q). " *
            "Use MATRIX_PROCESSING=symdrop or symfill to enforce structural symmetry.",
    )
    matrix_type = Pardiso.REAL_SYM
    @info "Using Pardiso REAL_SYM (mtype=1)"
    @show solver = MKLPardisoIterate(; nprocs, matrix_type)
elseif LINEAR_SOLVER == "ParU"
    @info "Using ParUFactorization (parallel sparse LU)"
    @show solver = ParUFactorization(; reuse_symbolic = true)
elseif LINEAR_SOLVER == "UMFPACK"
    @info "Using UMFPACKFactorization (serial sparse LU)"
    @show solver = UMFPACKFactorization(; reuse_symbolic = true)
end
flush(stdout); flush(stderr)

# Representative RHS for the coarse system: the NK preconditioner solves
# Q u = LUMP*x each GMRES apply (ldiv! in solve_periodic_NK.jl:142). Mirror that.
make_rhs() = LUMP_AND_SPRAY ? LUMP * randn(size(M, 1)) : randn(n_coarse)

################################################################################
# Benchmark: init (symbolic) → 1st solve (numeric factorize + solve) → 2nd solve
################################################################################

bytes_per_GB = 1024.0^3

GC.gc()
rss0 = Sys.maxrss()
@info "maxrss before init: $(round(rss0 / bytes_per_GB; digits = 2)) GB"
flush(stdout); flush(stderr)

prob = LinearProblem(Q, ones(n_coarse))

@info "Timing init (symbolic setup)"
flush(stdout); flush(stderr)
res_init = @timed cache = init(prob, solver, rtol = 1.0e-12)
@info @sprintf("init: %.3f s, alloc %.2f GB", res_init.time, res_init.bytes / bytes_per_GB)

@info "Timing 1st solve (numeric factorization + solve)"
flush(stdout); flush(stderr)
cache.b = make_rhs()
res_fact = @timed solve!(cache)
rss1 = Sys.maxrss()
@info @sprintf(
    "1st solve (factorize+solve): %.3f s, alloc %.2f GB, maxrss %.2f GB",
    res_fact.time, res_fact.bytes / bytes_per_GB, rss1 / bytes_per_GB,
)
flush(stdout); flush(stderr)

@info "Timing 2nd solve (reuses factorization)"
flush(stdout); flush(stderr)
cache.b = make_rhs()
res_solve = @timed solve!(cache)
rss2 = Sys.maxrss()
@info @sprintf(
    "2nd solve (solve only): %.3f s, alloc %.2f GB, maxrss %.2f GB",
    res_solve.time, res_solve.bytes / bytes_per_GB, rss2 / bytes_per_GB,
)
flush(stdout); flush(stderr)

maxrss_GB = Sys.maxrss() / bytes_per_GB

################################################################################
# Report
################################################################################

@info "================ PRECONDITIONER SOLVE BENCHMARK SUMMARY ================"
@info "LUMP_AND_SPRAY         = $coarse_tag (di=$(ls.di), dj=$(ls.dj))"
@info "LINEAR_SOLVER          = $LINEAR_SOLVER  (nprocs=$nprocs)"
@info "MATRIX_PROCESSING      = $MATRIX_PROCESSING"
@info "n_fine                 = $Nidx,  nnz(M)  = $(nnz(M))"
@info "n_coarse               = $n_coarse,  nnz(Mc) = $nnz_Mc,  nnz(Q) = $nnz_Q"
@info @sprintf("init time              = %.3f s", res_init.time)
@info @sprintf("factorize+solve time   = %.3f s", res_fact.time)
@info @sprintf("solve-only time        = %.3f s", res_solve.time)
@info @sprintf("factorize alloc        = %.2f GB", res_fact.bytes / bytes_per_GB)
@info @sprintf("maxrss (process peak)  = %.2f GB", maxrss_GB)
@info "GIT_COMMIT             = $git_commit"
@info "NOTE: authoritative peak memory is PBS resources_used.mem (one config per job)."
@info "======================================================================="
flush(stdout); flush(stderr)

# Per-config TSV (header + one row) — race-free across the concurrent sweep jobs.
# Combine afterwards with e.g. `column -t <(cat benchmarks/precond_solve_*.tsv)`.
header = join(
    [
        "tag", "di", "dj", "n_fine", "nnz_M", "n_coarse", "nnz_Mc", "nnz_Q",
        "LINEAR_SOLVER", "MATRIX_PROCESSING", "nprocs",
        "t_init_s", "t_factorize_s", "t_solve_s", "factorize_alloc_GB", "maxrss_GB",
        "git_commit",
    ], "\t",
)
row = join(
    [
        coarse_tag, ls.di, ls.dj, Nidx, nnz(M), n_coarse, nnz_Mc, nnz_Q,
        LINEAR_SOLVER, MATRIX_PROCESSING, nprocs,
        @sprintf("%.3f", res_init.time), @sprintf("%.3f", res_fact.time),
        @sprintf("%.3f", res_solve.time), @sprintf("%.2f", res_fact.bytes / bytes_per_GB),
        @sprintf("%.2f", maxrss_GB),
        git_commit,
    ], "\t",
)
tsv_file = joinpath(bench_dir, "$(bench_tag).tsv")
open(tsv_file, "w") do io
    println(io, header)
    println(io, row)
end
@info "Wrote benchmark row to $tsv_file"

jld_file = joinpath(bench_dir, "$(bench_tag).jld2")
jldsave(
    jld_file;
    tag = coarse_tag, di = ls.di, dj = ls.dj,
    n_fine = Nidx, nnz_M = nnz(M), n_coarse = n_coarse, nnz_Mc = nnz_Mc, nnz_Q = nnz_Q,
    LINEAR_SOLVER, MATRIX_PROCESSING, nprocs,
    t_init_s = res_init.time, t_factorize_s = res_fact.time, t_solve_s = res_solve.time,
    factorize_alloc_GB = res_fact.bytes / bytes_per_GB, maxrss_GB,
    git_commit, model_config,
)
@info "Saved benchmark result to $jld_file"

@info "benchmark_precond_solve.jl complete"
flush(stdout); flush(stderr)
