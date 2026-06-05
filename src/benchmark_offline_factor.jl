"""
Offline preconditioner-factorization toy (Phase 1, see docs/offline_factorization.md).

Proves the **save → load → reuse** cycle for the NK lump-and-spray preconditioner
matrix Q: factorize Q in one process, save the factor to disk, then in a *fresh*
process load the factor + LUMP/SPRAY and re-apply Q⁻¹ — asserting the reloaded
solve matches an in-process reference to relative error < 1e-8 (the correctness gate).

Q is built by the shared `build_precond_Q` helper (the exact same Q `solve_periodic_NK.jl`
factorizes), so a factor saved here is a valid preconditioner for the in-job NK solve.

Solvers (OFFLINE_SOLVER):
  UMFPACK     – stdlib SuiteSparse; save the plain-Julia pieces L,U,p,q,Rs (the C
                handle is not serializable). Trusted baseline + correctness reference.
  PureUMFPACK – pure-Julia LU; the factor object is plain data → jldsave directly.
                (unregistered; experimental)
  MUMPS       – MUMPS.jl direct, JOB=7 save / JOB=8 restore. Parallel; scales to OM2-01.

Phases (OFFLINE_PHASE):
  factor – build Q, save Q/LUMP/SPRAY + a test RHS b and reference solution u_ref,
           factorize with OFFLINE_SOLVER and save the factor. Records factorize time
           + Sys.maxrss + factor file size.
  reuse  – fresh process: load the factor + LUMP/SPRAY + b/u_ref, apply Q⁻¹, assert
           the correctness gate. Records load time + solve time + Sys.maxrss.
  both   – run factor then reuse in one process (convenient interactively; note the
           reuse-phase maxrss is contaminated by the factorization peak, so prefer the
           two separate phases for honest memory numbers — that is what the PBS wrapper does).

Usage — interactive:
```
qsub -I -P y99 -q normal -l ncpus=12 -l mem=48GB -l walltime=01:00:00 \\
     -l storage=gdata/xp65+gdata/ik11+gdata/cj50+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
PARENT_MODEL=ACCESS-OM2-1 LUMP_AND_SPRAY=2x2 OFFLINE_SOLVER=UMFPACK julia --project src/benchmark_offline_factor.jl
```

Environment variables:
  PARENT_MODEL      – model resolution tag (toy: ACCESS-OM2-1)
  MODEL_CONFIG      – config tag locating the TM dir (set by env_defaults.sh)
  LUMP_AND_SPRAY    – AxB coarsening (toy: 2x2)
  MATRIX_PROCESSING – raw | symfill | dropzeros | symdrop (default: raw — UMFPACK/MUMPS
                      unsymmetric LU need no structural symmetry)
  TM_SOURCE         – const | avg (default: const)
  OFFLINE_SOLVER    – UMFPACK | PureUMFPACK | MUMPS (default: UMFPACK)
  OFFLINE_PHASE     – factor | reuse | both (default: both)
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using LinearAlgebra
using SparseArrays
using JLD2
using Printf
using OceanTransportMatrixBuilder

@info "Packages loaded"
flush(stdout); flush(stderr)

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, outputdir) = load_project_config()

year = 365.25 * 86400  # seconds — matches NK stop_time for N_MONTHS=12

model_config = require_env("MODEL_CONFIG")
git_commit = get(ENV, "GIT_COMMIT", "unknown")

TM_SOURCE = get(ENV, "TM_SOURCE", "const")
(TM_SOURCE ∈ ("const", "avg")) || error("TM_SOURCE must be const or avg (got: $TM_SOURCE)")

MATRIX_PROCESSING = get(ENV, "MATRIX_PROCESSING", "raw")
(MATRIX_PROCESSING ∈ ("raw", "symfill", "dropzeros", "symdrop")) ||
    error("MATRIX_PROCESSING must be raw|symfill|dropzeros|symdrop (got: $MATRIX_PROCESSING)")

ls = parse_lump_and_spray()
coarse_tag = ls.on ? ls.tag : "full"

solver = get(ENV, "OFFLINE_SOLVER", "UMFPACK")
(solver ∈ ("UMFPACK", "PureUMFPACK", "MUMPS")) ||
    error("OFFLINE_SOLVER must be UMFPACK|PureUMFPACK|MUMPS (got: $solver)")

phase = get(ENV, "OFFLINE_PHASE", "both")
(phase ∈ ("factor", "reuse", "both")) ||
    error("OFFLINE_PHASE must be factor|reuse|both (got: $phase)")

matrices_dir = joinpath(outputdir, "TM", model_config)
M_dir = joinpath(matrices_dir, TM_SOURCE)
# Artifacts live beside the coarsened LUMP/SPRAY/Mc, under an offline-bench subdir.
art_dir = joinpath(matrices_dir, coarse_tag, "offline_bench")
mkpath(art_dir)

bytes_per_GB = 1024.0^3
gb(x) = x / bytes_per_GB

@info "Offline-factorization toy configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- MODEL_CONFIG      = $model_config"
@info "- TM_SOURCE         = $TM_SOURCE"
@info "- LUMP_AND_SPRAY    = $coarse_tag (di=$(ls.di), dj=$(ls.dj), dk=$(ls.dk))"
@info "- MATRIX_PROCESSING = $MATRIX_PROCESSING"
@info "- OFFLINE_SOLVER    = $solver"
@info "- OFFLINE_PHASE     = $phase"
@info "- art_dir           = $art_dir"
flush(stdout); flush(stderr)

# Lazily load the optional-dep solver packages only when selected, so a UMFPACK-only
# run needs neither PureUMFPACK nor MPI/MUMPS. (UMFPACK helpers live in matrix.jl.)
if solver == "PureUMFPACK"
    using PureUMFPACK
elseif solver == "MUMPS"
    using MPI
    using MUMPS
    MPI.Initialized() || MPI.Init()
end

# MUMPS save/restore writes to a directory with a fixed prefix; both phases must agree.
const MUMPS_SAVE_PREFIX = "offlinefactor"

# Artifact paths shared between phases.
Q_file = joinpath(art_dir, "Q.jld2")
LUMP_file = joinpath(art_dir, "LUMP.jld2")
SPRAY_file = joinpath(art_dir, "SPRAY.jld2")
rhs_file = joinpath(art_dir, "test_rhs_ref.jld2")
factor_path(s) = joinpath(art_dir, s == "MUMPS" ? "factor_MUMPS" : "factor_$(s).jld2")

################################################################################
# Per-solver factorize+save and load+apply dispatch.
#
# Each returns/uses only what the cycle needs. UMFPACK lives in shared_utils/matrix.jl
# (save_umfpack_factor / load_umfpack_factor / apply_umfpack_factor); the optional-dep
# solvers (PureUMFPACK, MUMPS) are loaded lazily here so a UMFPACK-only run needs neither.
################################################################################

function factorize_and_save(solver, Q, path)
    if solver == "UMFPACK"
        F = lu(Q)
        save_umfpack_factor(path, F)
        return F   # also usable as the in-process reference
    elseif solver == "PureUMFPACK"
        # Use the supernodal multifrontal (BLAS-3) kernel: the default :gplu is the
        # unblocked left-looking LU, 4-20x slower than UMFPACK on 3D-like problems —
        # it overran walltime on this 710k-row ocean matrix. multifrontal is ~parity.
        # PureLU is plain Julia data → serialize the whole factor directly. (A is the
        # coarse Q, small vs the L/U factors, so we keep it rather than strip it.)
        F = PureUMFPACK.splu(Q; method = :multifrontal)
        jldsave(path; F)
        return F
    elseif solver == "MUMPS"
        # MUMPS native save/restore: factorize, then JOB=7 (SAVE_DATA) to `path`.
        mkpath(path)
        m = MUMPS.Mumps{Float64}(MUMPS.mumps_unsymmetric)
        MUMPS.associate_matrix!(m, Q)
        MUMPS.factorize!(m)
        MUMPS.set_save_dir!(m, path)
        MUMPS.set_save_prefix!(m, MUMPS_SAVE_PREFIX)
        MUMPS.set_job!(m, MUMPS.SAVE_DATA)
        MUMPS.invoke_mumps!(m)
        MUMPS.finalize!(m)
        return m
    end
end

function load_and_apply(solver, path, b)
    if solver == "UMFPACK"
        fac = load_umfpack_factor(path)
        return apply_umfpack_factor(fac, b)
    elseif solver == "PureUMFPACK"
        F = load(path, "F")
        return PureUMFPACK.solve(F, b)
    elseif solver == "MUMPS"
        # JOB=8 (RESTORE_DATA) into a fresh instance, then a pure SOLVE on the
        # restored factors. (The constructor already runs JOB=-1 INITIALIZE.)
        m = MUMPS.Mumps{Float64}(MUMPS.mumps_unsymmetric)
        MUMPS.set_save_dir!(m, path)
        MUMPS.set_save_prefix!(m, MUMPS_SAVE_PREFIX)
        MUMPS.set_job!(m, MUMPS.RESTORE_DATA)
        MUMPS.invoke_mumps!(m)
        MUMPS.associate_rhs!(m, b)
        MUMPS.mumps_solve!(m; rhs_changed = true)
        x = copy(MUMPS.get_solution(m))
        MUMPS.finalize!(m)
        return x
    end
end

################################################################################
# Phase: factor
################################################################################

results = Dict{Symbol, Any}()

if phase ∈ ("factor", "both")
    @info "[factor] Loading grid, wet mask, volumes"
    flush(stdout); flush(stderr)
    grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
    (; wet3D, idx, Nidx) = compute_wet_mask(grid)
    v1D = interior(compute_volume(grid))[idx]
    @info "[factor] wet cells (fine) = $Nidx"

    M_file = joinpath(M_dir, "M.jld2")
    @info "[factor] Loading M from $M_file"
    flush(stdout); flush(stderr)
    M = load(M_file, "M")
    @assert Nidx == size(M, 1) "wet cells ($Nidx) ≠ matrix rows ($(size(M, 1)))"

    # Build Q exactly as NK does (shared helper).
    (; Q, LUMP, SPRAY) = build_precond_Q(M, wet3D, v1D, ls, year, MATRIX_PROCESSING)
    n_coarse = size(Q, 1)
    @info "[factor] Q: $(n_coarse)×$(size(Q, 2)), nnz=$(nnz(Q))"

    # Persist Q + coarsening operators + a representative coarse RHS and reference
    # solution for the reuse-phase correctness gate. The NK preconditioner solves
    # Q u = LUMP*x each GMRES apply, so draw b that way.
    x_fine = randn(size(M, 1))
    b = ls.on ? LUMP * x_fine : x_fine
    Fref = lu(Q)
    u_ref = Fref \ b
    @info @sprintf(
        "[factor] reference residual ||Q u_ref - b||/||b|| = %.2e",
        norm(Q * u_ref - b) / norm(b)
    )

    jldsave(Q_file; Q)
    jldsave(LUMP_file; LUMP = ls.on ? LUMP : I)
    jldsave(SPRAY_file; SPRAY = ls.on ? SPRAY : I)
    jldsave(rhs_file; b, u_ref, n_coarse, n_fine = Nidx, nnz_Q = nnz(Q))

    # Factorize with the chosen solver and save the factor.
    @info "[factor] Factorizing with $solver"
    flush(stdout); flush(stderr)
    GC.gc()
    fpath = factor_path(solver)
    res_fact = @timed factorize_and_save(solver, Q, fpath)
    maxrss_factor = Sys.maxrss()
    fsize = isdir(fpath) ?
        sum(filesize(joinpath(fpath, f)) for f in readdir(fpath); init = 0) :
        filesize(fpath)
    @info @sprintf(
        "[factor] %s: factorize+save %.3f s, maxrss %.2f GB, factor on disk %.3f GB",
        solver, res_fact.time, gb(maxrss_factor), gb(fsize),
    )

    results[:t_factorize_s] = res_fact.time
    results[:maxrss_factor_GB] = gb(maxrss_factor)
    results[:factor_size_GB] = gb(fsize)
    results[:n_coarse] = n_coarse
    results[:n_fine] = Nidx
    results[:nnz_Q] = nnz(Q)
    flush(stdout); flush(stderr)
end

################################################################################
# Phase: reuse  (fresh process when run as its own phase)
################################################################################

if phase ∈ ("reuse", "both")
    @info "[reuse] Loading saved LUMP/SPRAY + test RHS/reference"
    flush(stdout); flush(stderr)
    LUMP = load(LUMP_file, "LUMP")
    SPRAY = load(SPRAY_file, "SPRAY")
    rhs = load(rhs_file)
    b = rhs["b"]
    u_ref = rhs["u_ref"]

    GC.gc()
    fpath = factor_path(solver)
    @info "[reuse] Loading + applying factor from $fpath"
    flush(stdout); flush(stderr)
    res_load = @timed begin
        u = load_and_apply(solver, fpath, b)
        u
    end
    u = res_load.value
    maxrss_reuse = Sys.maxrss()

    # Correctness gate: reloaded solve must match the in-process reference.
    rel_err = norm(u - u_ref) / norm(u_ref)
    # Full preconditioner action P x = SPRAY*(Q⁻¹ LUMP x) - x is linear in u, so a
    # matching u guarantees a matching action; report the action norm for sanity.
    action = SPRAY * u
    @info @sprintf(
        "[reuse] %s: load+apply %.3f s, maxrss %.2f GB, rel-err vs ref = %.3e",
        solver, res_load.time, gb(maxrss_reuse), rel_err,
    )

    gate = 1.0e-8
    if rel_err < gate
        @info @sprintf("[reuse] CORRECTNESS GATE PASSED (rel-err %.3e < %.0e)", rel_err, gate)
    else
        error(@sprintf("[reuse] CORRECTNESS GATE FAILED: rel-err %.3e ≥ %.0e", rel_err, gate))
    end

    results[:t_loadapply_s] = res_load.time
    results[:maxrss_reuse_GB] = gb(maxrss_reuse)
    results[:rel_err] = rel_err
    results[:reuse_correct] = rel_err < gate
    results[:action_norm] = norm(action)
    flush(stdout); flush(stderr)
end

################################################################################
# Report — one TSV row per (solver, phase) invocation; combine across runs.
################################################################################

g(k) = get(results, k, "")
# Runtime format string ⇒ use Printf.Format (the @sprintf macro needs a literal).
fmt(k, f) = haskey(results, k) ? Printf.format(Printf.Format(f), results[k]) : ""

@info "================ OFFLINE FACTORIZATION TOY SUMMARY ================"
@info "solver=$solver phase=$phase tag=$coarse_tag MATRIX_PROCESSING=$MATRIX_PROCESSING"
haskey(results, :n_coarse) && @info "n_fine=$(g(:n_fine)) n_coarse=$(g(:n_coarse)) nnz_Q=$(g(:nnz_Q))"
haskey(results, :t_factorize_s) && @info @sprintf(
    "factorize+save = %.3f s, maxrss %.2f GB, factor %.3f GB",
    results[:t_factorize_s], results[:maxrss_factor_GB], results[:factor_size_GB]
)
haskey(results, :t_loadapply_s) && @info @sprintf(
    "load+apply = %.3f s, maxrss %.2f GB, rel-err %.3e, correct=%s",
    results[:t_loadapply_s], results[:maxrss_reuse_GB], results[:rel_err], results[:reuse_correct]
)
@info "NOTE: authoritative peak memory is PBS resources_used.mem (one phase per job)."
@info "GIT_COMMIT=$git_commit"
@info "=================================================================="

header = join(
    [
        "solver", "phase", "tag", "di", "dj", "MATRIX_PROCESSING",
        "n_fine", "n_coarse", "nnz_Q",
        "t_factorize_s", "maxrss_factor_GB", "factor_size_GB",
        "t_loadapply_s", "maxrss_reuse_GB", "rel_err", "reuse_correct",
        "git_commit",
    ], "\t"
)
row = join(
    [
        solver, phase, coarse_tag, ls.di, ls.dj, MATRIX_PROCESSING,
        g(:n_fine), g(:n_coarse), g(:nnz_Q),
        fmt(:t_factorize_s, "%.3f"), fmt(:maxrss_factor_GB, "%.2f"), fmt(:factor_size_GB, "%.4f"),
        fmt(:t_loadapply_s, "%.3f"), fmt(:maxrss_reuse_GB, "%.2f"), fmt(:rel_err, "%.3e"), g(:reuse_correct),
        git_commit,
    ], "\t"
)
tsv_file = joinpath(art_dir, "offline_factor_$(solver)_$(phase).tsv")
open(tsv_file, "w") do io
    println(io, header)
    println(io, row)
end
@info "Wrote benchmark row to $tsv_file"

@info "benchmark_offline_factor.jl complete"
flush(stdout); flush(stderr)
