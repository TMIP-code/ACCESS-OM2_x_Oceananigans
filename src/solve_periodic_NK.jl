"""
Solve for the periodic steady-state age using Newton-GMRES.

Finds x such that G(x) = Φ(x) - x = 0, where Φ(x) is the result of running
the model for 1 year from initial condition x. Uses either a lump-and-spray
preconditioner (Bardin et al., 2014) or a direct Q⁻¹ - I preconditioner. The
Jacobian-vector product is computed exactly via the linear forward map
(`Φ!(·; source_rate = 0)`).

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=04:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/solve_periodic_NK.jl")
```

Environment variables (in addition to setup_model.jl):
  LINEAR_SOLVER  – Pardiso | ParU | UMFPACK  (default: Pardiso)
                   Pardiso: MKL Pardiso iterative solver
                   ParU:    ParU parallel sparse LU factorization
                   UMFPACK: UMFPACK sparse LU factorization (ships with Julia)
  LUMP_AND_SPRAY – no | AxB  (default: no)
                   no:    direct preconditioner P = Q⁻¹ - I where Q = stop_time * M
                   AxB:   lump-and-spray coarsening with di=A, dj=B, dk=1
                          (Bardin et al., 2014). Example: 5x5, 4x4, 2x2.
  MATRIX_PROCESSING – raw | symfill | dropzeros | symdrop  (default: raw)
                   Processing applied to Q_precond before Pardiso factorization.
                   raw:       no processing (Pardiso requires structurally symmetric M)
                   symfill:   add zero entries at (j,i) for every existing (i,j)
                   dropzeros: remove stored zeros
                   symdrop:   keep (i,j) only if (j,i) also exists, then drop zeros
  TM_SOURCE      – const | avg  (default: const)
                   Subdirectory under TM/{model_config}/ to load M from.
"""

include("setup_model.jl")

using NonlinearSolve
using LinearAlgebra
using SparseArrays
using OceanTransportMatrixBuilder
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using KernelAbstractions: @kernel, @index
import Pardiso
import ParU_jll
using LinearSolve: ParUFactorization, UMFPACKFactorization
const nprocs = parse(Int, get(ENV, "PARDISO_NPROCS", ENV["PBS_NCPUS"]))

################################################################################
# Configuration
################################################################################

LINEAR_SOLVER = require_env("LINEAR_SOLVER")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) || error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

# Offline preconditioner reuse (Phase 2): if PRECOND_FACTOR points at an offline_bench
# dir (from benchmark_offline_factor.jl), rank 0 loads LUMP/SPRAY + RESTORES the saved
# MUMPS factor (JOB=8) instead of building Q and factorizing in-job. The MUMPS instance
# runs on a 1-rank COMM_SELF communicator on rank 0 — decoupled from NK's GPU-partition
# MPI ranks (whose worker loop never calls MUMPS), so its collective calls can't deadlock
# and the single-rank-saved factor restores regardless of NK's rank count. LINEAR_SOLVER
# is then ignored for the factorization (still used for output tags).
PRECOND_FACTOR = get(ENV, "PRECOND_FACTOR", "")
if !isempty(PRECOND_FACTOR)
    import MUMPS
end

ls = parse_lump_and_spray()
LUMP_AND_SPRAY = ls.on
lumpspray_tag = ls.tag

MATRIX_PROCESSING = require_env("MATRIX_PROCESSING")
(MATRIX_PROCESSING ∈ ("raw", "symfill", "dropzeros", "symdrop")) || error("MATRIX_PROCESSING must be one of: raw, symfill, dropzeros, symdrop (got: $MATRIX_PROCESSING)")

TM_SOURCE = require_env("TM_SOURCE")
(TM_SOURCE ∈ ("const", "avg")) || error("TM_SOURCE must be one of: const, avg (got: $TM_SOURCE)")

TRAF = lowercase(require_env("TRAF")) == "yes"
TRAF_TM_SOURCE = require_env("TRAF_TM_SOURCE")
if TRAF
    TM_SOURCE == "const" || error(
        "TRAF=yes is only supported with TM_SOURCE=const in the first cut (got TM_SOURCE=$TM_SOURCE). " *
            "Snapshot/avg-matrix support for TRAF is a follow-up.",
    )
    TRAF_TM_SOURCE in ("invVMtV", "M_traf") ||
        error("TRAF_TM_SOURCE must be invVMtV or M_traf (got: $TRAF_TM_SOURCE)")
end
M_basename = if !TRAF
    "M.jld2"
elseif TRAF_TM_SOURCE == "invVMtV"
    "invVMtV.jld2"
else  # "M_traf"
    "M_traf.jld2"
end

TM_MODEL_CONFIG = let v = get(ENV, "TM_MODEL_CONFIG", "")
    isempty(v) ? model_config : v
end
matrices_dir = joinpath(outputdir, "TM", TM_MODEL_CONFIG)
if TM_MODEL_CONFIG != model_config
    @warn "NK using TM from a different model_config (preconditioner approximation only)" tm_config = TM_MODEL_CONFIG nk_config = model_config
end

omega = parse_omega()
omega_suffix = omega.suffix

@info "Newton-GMRES periodic solver configuration"
@info "- LINEAR_SOLVER = $LINEAR_SOLVER"
@info "- TM_SOURCE = $TM_SOURCE"
@info "- LUMP_AND_SPRAY = $LUMP_AND_SPRAY (di=$(ls.di), dj=$(ls.dj), dk=$(ls.dk), tag: $lumpspray_tag)"
@info "- MATRIX_PROCESSING = $MATRIX_PROCESSING"
@info "- OMEGA = $(omega.tag) (suffix='$(omega_suffix)')"
@info "- matrices_dir = $matrices_dir"
flush(stdout); flush(stderr)

################################################################################
# Solver output directory (used by periodic_solver_common.jl for trace files)
################################################################################

px = parse(Int, require_env("PARTITION_X"))
py = parse(Int, require_env("PARTITION_Y"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
nk_dirname = "NK$(ls.dir_suffix)"
solver_output_dir = isempty(gpu_tag) ?
    joinpath(outputdir, "periodic", model_config, nk_dirname) :
    joinpath(outputdir, "periodic", model_config, gpu_tag, nk_dirname)
mkpath(solver_output_dir)

################################################################################
# Common solver infrastructure (simulation, wet mask, buffers, Φ!, G!)
################################################################################

include("periodic_solver_common.jl")

################################################################################
# Rank-0: M load, preconditioner, JVP setup, Newton solve, save.
# Rank > 0: worker loop joining each Φ!_body call via MPI.Bcast!.
################################################################################

# TMPreconditioner struct + ldiv! methods live at module scope (extend Base/LA).
# Only rank 0 instantiates it; rank > 0 never references it.
if !@isdefined(TMPreconditioner)
    struct TMPreconditioner
        prob
    end
end
Base.eltype(::TMPreconditioner) = Float64
function LinearAlgebra.ldiv!(Pl::TMPreconditioner, x::AbstractVector)
    Pl.prob.b = LUMP * x
    solve!(Pl.prob)
    x .= SPRAY * Pl.prob.u .- x
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, Pl::TMPreconditioner, x::AbstractVector)
    mul!(Pl.prob.b, LUMP, x)
    solve!(Pl.prob)
    mul!(y, SPRAY, Pl.prob.u)
    y .-= x
    return y
end

# Offline-factor variant: same action P = S Q⁻¹ L − I, but Q⁻¹ is applied by a restored
# MUMPS factor (JOB=8) held on rank 0's COMM_SELF instance. Only rank 0 ever calls these.
#
# Fieldless on purpose: the restored Mumps lives in the module global MUMPS_PRECOND (set
# on rank 0 below), like ldiv! already reads global LUMP/SPRAY. A field holding the raw
# Mumps would make NonlinearSolve's trace printer (clean_sprint_struct) recurse into the
# Mumps struct and hit its uninitialized fields (UndefRefError); a fieldless struct is
# printed as a leaf.
if !@isdefined(MumpsPreconditioner)
    struct MumpsPreconditioner end
end
Base.eltype(::MumpsPreconditioner) = Float64
function LinearAlgebra.ldiv!(::MumpsPreconditioner, x::AbstractVector)
    MUMPS.associate_rhs!(MUMPS_PRECOND, LUMP * x)
    MUMPS.mumps_solve!(MUMPS_PRECOND; rhs_changed = true)
    x .= SPRAY * MUMPS.get_solution(MUMPS_PRECOND) .- x
    return x
end
function LinearAlgebra.ldiv!(y::AbstractVector, ::MumpsPreconditioner, x::AbstractVector)
    MUMPS.associate_rhs!(MUMPS_PRECOND, LUMP * x)
    MUMPS.mumps_solve!(MUMPS_PRECOND; rhs_changed = true)
    y .= SPRAY * MUMPS.get_solution(MUMPS_PRECOND) .- x
    return y
end

if rank == 0
    if isempty(PRECOND_FACTOR)
        ########################################################################
        # In-job path: load M, build Q, factorize (default).
        ########################################################################

        M_file = joinpath(matrices_dir, TM_SOURCE, M_basename)
        @info "[rank 0] Loading transport matrix from $M_file"
        flush(stdout); flush(stderr)
        M = load(M_file, "M")
        @info "[rank 0] Loaded M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"
        flush(stdout); flush(stderr)

        @assert Nidx_global == size(M, 1) "Mismatch: global wet cells ($Nidx_global) ≠ matrix rows ($(size(M, 1)))"

        # Build Q exactly as the offline factorizer does (shared single source of truth).
        precond = build_precond_Q(M, wet3D_global, v1D, ls, stop_time, MATRIX_PROCESSING)
        LUMP = precond.LUMP
        SPRAY = precond.SPRAY
        Q_precond = precond.Q
        precond = nothing
        flush(stdout); flush(stderr)

        # Free the raw transport matrix before factorization: the Jv's use the GPU
        # forward model, not M, so the tens-of-GB sparse M on rank 0 is dead weight
        # during the (memory-peak) factorization.
        M = nothing
        GC.gc()

        @info "Setting up preconditioner (LINEAR_SOLVER=$LINEAR_SOLVER)"
        flush(stdout); flush(stderr)

        if LINEAR_SOLVER == "Pardiso"
            error_msg = """
                Q_precond is not structurally symmetric (nnz=$(nnz(Q_precond))).
                Use MATRIX_PROCESSING=symdrop or symfill to enforce structural symmetry.
            """
            Pardiso.isstructurallysymmetric(Q_precond) || error(error_msg)
            matrix_type = Pardiso.REAL_SYM
            @info "Using Pardiso REAL_SYM (mtype=1)"
            @show linear_solver = MKLPardisoIterate(; nprocs, matrix_type)
        elseif LINEAR_SOLVER == "ParU"
            @info "Using ParUFactorization (parallel sparse LU)"
            @show linear_solver = ParUFactorization(; reuse_symbolic = true)
        elseif LINEAR_SOLVER == "UMFPACK"
            @info "Using UMFPACKFactorization (serial sparse LU)"
            @show linear_solver = UMFPACKFactorization(; reuse_symbolic = true)
        end

        # P = S Q⁻¹ L - I  (Bardin et al., 2014)
        # When LUMP_AND_SPRAY=no, LUMP = SPRAY = I, so P = Q⁻¹ - I
        Plprob = LinearProblem(Q_precond, ones(size(Q_precond, 1)))
        Plprob = init(Plprob, linear_solver, rtol = 1.0e-12)
        Pl = TMPreconditioner(Plprob)
    else
        ########################################################################
        # Offline-reuse path: load LUMP/SPRAY + RESTORE the saved MUMPS factor.
        # No M load, no factorization — just JOB=8 restore on a COMM_SELF instance.
        ########################################################################

        @info "[rank 0] PRECOND_FACTOR mode: loading offline factor from $PRECOND_FACTOR"
        flush(stdout); flush(stderr)
        LUMP = load(joinpath(PRECOND_FACTOR, "LUMP.jld2"), "LUMP")
        SPRAY = load(joinpath(PRECOND_FACTOR, "SPRAY.jld2"), "SPRAY")
        @assert size(LUMP, 2) == Nidx_global "LUMP fine dim ($(size(LUMP, 2))) ≠ global wet cells ($Nidx_global)"

        # 1-rank MUMPS on rank 0 only (COMM_SELF): matches the single-rank-saved factor
        # and stays off NK's GPU-partition COMM_WORLD, so it can't collide with the
        # workers' Φ! loop.
        comm_self = MPI.API.MPI_Comm_c2f(MPI.COMM_SELF.val)
        MUMPS_PRECOND = MUMPS.Mumps{Float64}(MUMPS.mumps_unsymmetric, 1, comm_self)
        MUMPS.set_save_dir!(MUMPS_PRECOND, joinpath(PRECOND_FACTOR, "factor_MUMPS"))
        MUMPS.set_save_prefix!(MUMPS_PRECOND, "offlinefactor")
        MUMPS.set_job!(MUMPS_PRECOND, MUMPS.RESTORE_DATA)
        MUMPS.invoke_mumps!(MUMPS_PRECOND)
        @info "[rank 0] MUMPS factor restored (COMM_SELF, 1-rank); using MumpsPreconditioner"
        Pl = MumpsPreconditioner()
    end

    Pr = I
    precs = Returns((Pl, Pr))

    @info "Preconditioner ready"
    flush(stdout); flush(stderr)

    ############################################################################
    # JVP setup (exact JVP via linear tracer)
    ############################################################################

    @info "Using exact JVP via linear tracer (linΦ!)"
    flush(stdout); flush(stderr)

    f! = NonlinearFunction(G!; jvp = jvp!)
    newton_solver = NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, gmres_restart = 50, rtol = 1.0e-4),
    )

    ############################################################################
    # Nonlinear solve: Newton-GMRES
    ############################################################################

    @info "Solving nonlinear problem with Newton-GMRES"
    @info "- JVP method: exact (linear tracer)"
    @info "- Preconditioner: $(LUMP_AND_SPRAY ? "lump-and-spray (Bardin et al., 2014)" : "direct Q⁻¹ - I")"
    @info "- Linear solver: $LINEAR_SOLVER"
    @info "- abstol = 0.001 years (volume-weighted RMS norm)"
    flush(stdout); flush(stderr)

    age_init_vec = load_initial_age(idx_global, Nidx_global, outputdir, model_config; year, solver_output_dir)
    nonlinearprob = NonlinearProblem(f!, age_init_vec)

    NK_MAXITERS = parse(Int, get(ENV, "NK_MAXITERS", "1000"))
    @info "Newton-GMRES maxiters = $NK_MAXITERS"

    @time sol = solve(
        nonlinearprob,
        newton_solver;
        # internalnorm = vol_norm,
        show_trace = Val(true),
        reltol = Inf,
        abstol = 0.0001year,
        maxiters = NK_MAXITERS,
        verbose = true,
    )

    @info "Newton-GMRES solve complete" retcode = sol.retcode total_Φ_calls = Φ_call_count[] total_G_calls = G_call_count[] total_jvp_calls = jvp_call_count[]
    flush(stdout); flush(stderr)

    # Signal workers (rank > 0) that the solve is done
    arch isa Distributed && MPI.Bcast!(zeros(2), 0, COMM)

    ############################################################################
    # Save result
    ############################################################################

    @info "Saving steady-state age"
    flush(stdout); flush(stderr)

    age_steady_3D = zeros(Float64, Nx′_global, Ny′_global, Nz′_global)
    age_steady_3D[idx_global] .= sol.u

    vol_mean = sum(sol.u .* v1D) / sum(v1D) / year
    @info "Volume-weighted mean periodic steady age: $vol_mean years"

    steady_dir = solver_output_dir
    steady_file = joinpath(steady_dir, "age_$(LINEAR_SOLVER)_$(lumpspray_tag)$(omega_suffix).jld2")
    jldsave(steady_file; age = age_steady_3D, wet3D = wet3D_global, idx = idx_global)
    @info "Saved steady-state age to $steady_file"
    flush(stdout); flush(stderr)

    @info "solve_periodic_NK.jl complete"
    flush(stdout); flush(stderr)
else
    # Rank > 0 worker loop: block on Bcast between Φ! invocations; when rank 0
    # signals end-of-solve (continue_flag = 0), exit.
    ctrl = zeros(2)        # [continue_flag, source_rate]
    dummy = Float64[]
    while true
        MPI.Bcast!(ctrl, 0, COMM)
        ctrl[1] == 0.0 && break
        Φ!_body(dummy, dummy; source_rate = ctrl[2])
    end
    @info "[rank $rank] worker loop exited; solve done"
    flush(stdout); flush(stderr)
end
