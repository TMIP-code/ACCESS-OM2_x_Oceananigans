"""
Pardiso-under-MPI test (step 1 of the partitioned-NK plan).

Loads the transport matrix M on rank 0, computes the coarsened preconditioner
matrix Mc = stop_time · LUMP · M · SPRAY (the same shape NK uses inside
TMPreconditioner.ldiv!), and solves Mc · x = b with MKLPardisoIterate while
ranks > 0 sit in MPI.Barrier. Sweeps `nprocs` (the Pardiso thread count) so we
can pick a validated value to pass to MKLPardisoIterate in step 5.

When invoked without mpiexec (singleton MPI world) the test still runs and
serves as a serial baseline for the residual + wall-clock comparison.

Environment variables (in addition to env_defaults.sh):
  PARDISO_NPROCS_SWEEP – comma-separated nprocs values (default: "12,24")
  TM_SOURCE            – const | avg (default: const)
"""

using MPI
using JLD2
using SparseArrays
using LinearAlgebra
using Statistics
using Printf
using Random
using Oceananigans
using Oceananigans.Architectures: CPU
using OceanTransportMatrixBuilder
import Pardiso
using LinearSolve

include("../src/shared_functions.jl")

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)
const MODE = NRANKS > 1 ? "MPI[$NRANKS]" : "Serial(singleton)"

if RANK == 0
    @info "Pardiso-under-MPI test" mode = MODE ranks = NRANKS
    flush(stdout); flush(stderr)
end

# Configuration
(; parentmodel, experiment_dir, outputdir) = load_project_config()
(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

TM_SOURCE = get(ENV, "TM_SOURCE", "const")
matrices_dir = joinpath(outputdir, "TM", model_config, TM_SOURCE)
M_file = joinpath(matrices_dir, "M.jld2")

NPROCS_SWEEP = parse.(Int, split(get(ENV, "PARDISO_NPROCS_SWEEP", "12,24"), ","))

if RANK == 0
    @info "Configuration" TM_SOURCE M_file NPROCS_SWEEP
    @info "Thread env" MKL_NUM_THREADS = get(ENV, "MKL_NUM_THREADS", "(unset)") OMP_NUM_THREADS = get(ENV, "OMP_NUM_THREADS", "(unset)") JULIA_NUM_THREADS = get(ENV, "JULIA_NUM_THREADS", "(unset)")
    flush(stdout); flush(stderr)
end

# Peak resident-set size from /proc/self/status (VmHWM, in kB)
function peak_rss_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end

# Build Mc and b on rank 0 only
Mc = sparse(Int[], Int[], Float64[], 0, 0)
b = Float64[]

if RANK == 0
    @info "[rank 0] Loading M from $M_file"
    flush(stdout); flush(stderr)
    M = load(M_file, "M")
    @info "[rank 0] M loaded" rows = size(M, 1) cols = size(M, 2) nnz = nnz(M)

    grid_file = joinpath(experiment_dir, "grid.jld2")
    @info "[rank 0] Loading grid from $grid_file"
    flush(stdout); flush(stderr)
    grid = load_tripolar_grid(grid_file, CPU())
    (; wet3D, idx, Nidx) = compute_wet_mask(grid)
    @assert Nidx == size(M, 1) "Mismatch: wet cells ($Nidx) != matrix rows ($(size(M, 1)))"

    v1D = interior(compute_volume(grid))[idx]

    @info "[rank 0] Computing LUMP/SPRAY (di=2, dj=2, dk=1)"
    flush(stdout); flush(stderr)
    LUMP, SPRAY, _ = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
    year = 365.25 * 86400.0
    stop_time = year

    global Mc = LUMP * M * SPRAY
    Mc.nzval .*= stop_time
    @info "[rank 0] Mc = stop_time · LUMP · M · SPRAY" rows = size(Mc, 1) cols = size(Mc, 2) nnz = nnz(Mc)
    flush(stdout); flush(stderr)

    global b = randn(MersenneTwister(0xCAFE), size(Mc, 1))
end

MPI.Barrier(COMM)

# Sweep nprocs; rank > 0 sits in barriers
for np in NPROCS_SWEEP
    MPI.Barrier(COMM)
    if RANK == 0
        @info "==== Pardiso solve with nprocs=$np ($MODE) ===="
        flush(stdout); flush(stderr)

        sym = Pardiso.isstructurallysymmetric(Mc)
        matrix_type = sym ? Pardiso.REAL_SYM : Pardiso.REAL_NONSYM
        @info "[nprocs=$np] structurally symmetric: $sym → matrix_type=$(sym ? "REAL_SYM" : "REAL_NONSYM")"

        solver = MKLPardisoIterate(; nprocs = np, matrix_type)

        rss_before = peak_rss_kb()

        t0 = time()
        cache = init(LinearProblem(Mc, copy(b)), solver, rtol = 1.0e-12)
        sol1 = solve!(cache)
        t_first = time() - t0

        cache.b = copy(b)
        t0 = time()
        sol2 = solve!(cache)
        t_second = time() - t0

        x = sol2.u
        r = b - Mc * x
        res_norm = norm(r)
        b_norm = norm(b)
        rel = res_norm / b_norm
        rss_after = peak_rss_kb()

        @info "[nprocs=$np] timing" t_first_s = round(t_first, digits = 2) t_second_s = round(t_second, digits = 2)
        @info "[nprocs=$np] residual" abs_res = res_norm rel_res = rel b_norm = b_norm
        @info "[nprocs=$np] memory" peak_rss_before_kB = rss_before peak_rss_after_kB = rss_after delta_kB = rss_after - rss_before
        flush(stdout); flush(stderr)
    end
    MPI.Barrier(COMM)
end

MPI.Barrier(COMM)
if RANK == 0
    @info "Pardiso-under-MPI test complete"
    flush(stdout); flush(stderr)
end

MPI.Finalize()
