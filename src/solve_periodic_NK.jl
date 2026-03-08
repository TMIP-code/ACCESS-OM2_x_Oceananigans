"""
Solve for the periodic steady-state age using Newton-GMRES.

Finds x such that G(x) = Φ(x) - x = 0, where Φ(x) is the result of running
the model for 1 year from initial condition x. Uses either a lump-and-spray
preconditioner (Bardin et al., 2014) or a direct Q⁻¹ - I preconditioner, and
either matrix-based or finite-difference JVP for GMRES.

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
  JVP_METHOD     – exact | matrix | finitediff  (default: exact)
                   exact:      exact JVP via linear tracer linΦ! (1 simulation per JVP)
                   matrix:     approximate JVP using transport matrix M (fast, sparse matvec)
                   finitediff: finite-difference JVP via AutoFiniteDiff (slow, extra G! evals)
  LINEAR_SOLVER  – Pardiso | ParU | UMFPACK  (default: Pardiso)
                   Pardiso: MKL Pardiso iterative solver
                   ParU:    ParU parallel sparse LU factorization
                   UMFPACK: UMFPACK sparse LU factorization (ships with Julia)
  LUMP_AND_SPRAY – yes | no  (default: no)
                   yes: lump-and-spray coarsening (Bardin et al., 2014)
                   no:  direct preconditioner P = Q⁻¹ - I where Q = stop_time * M
  TM_SOURCE      – const | avg24 | avg12a | avg12b  (default: const)
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
const nprocs = 12

################################################################################
# Configuration
################################################################################

JVP_METHOD = get(ENV, "JVP_METHOD", "exact")
(JVP_METHOD ∈ ("exact", "matrix", "finitediff")) || error("JVP_METHOD must be one of: exact, matrix, finitediff (got: $JVP_METHOD)")

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) || error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
lumpspray_tag = LUMP_AND_SPRAY ? "LSprec" : "prec"

TM_SOURCE = get(ENV, "TM_SOURCE", "const")
(TM_SOURCE ∈ ("const", "avg24", "avg12a", "avg12b")) || error("TM_SOURCE must be one of: const, avg24, avg12a, avg12b (got: $TM_SOURCE)")

matrices_dir = joinpath(outputdir, "TM", model_config)

@info "Newton-GMRES periodic solver configuration"
@info "- JVP_METHOD  = $JVP_METHOD"
@info "- LINEAR_SOLVER = $LINEAR_SOLVER"
@info "- TM_SOURCE = $TM_SOURCE"
@info "- LUMP_AND_SPRAY = $LUMP_AND_SPRAY (tag: $lumpspray_tag)"
@info "- matrices_dir = $matrices_dir"
flush(stdout); flush(stderr)

################################################################################
# Load pre-built transport matrix M from disk
################################################################################

M_file = joinpath(matrices_dir, TM_SOURCE, "M.jld2")
@info "Loading transport matrix from $M_file"
flush(stdout); flush(stderr)
M = load(M_file, "M")
@info "Loaded M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"
flush(stdout); flush(stderr)

################################################################################
# Solver output directory (used by periodic_solver_common.jl for trace files)
################################################################################

solver_output_dir = joinpath(outputdir, "periodic", model_config, "NK")
mkpath(solver_output_dir)

################################################################################
# Common solver infrastructure (simulation, wet mask, buffers, Φ!, G!)
################################################################################

include("periodic_solver_common.jl")

@assert Nidx == size(M, 1) "Mismatch: wet cells ($Nidx) != matrix rows ($(size(M, 1)))"

################################################################################
# LUMP, SPRAY, and preconditioner matrix Q_precond
################################################################################

if LUMP_AND_SPRAY
    @info "Computing LUMP and SPRAY matrices"
    flush(stdout); flush(stderr)
    LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
    Mc = LUMP * M * SPRAY
    @info "Coarsened Jacobian Mc: $(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc))"
    Q_precond = copy(Mc)
    Q_precond.nzval .*= stop_time
else
    @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no); using full Q = stop_time * M"
    LUMP = I
    SPRAY = I
    Q_precond = copy(M)
    Q_precond.nzval .*= stop_time
end
flush(stdout); flush(stderr)

################################################################################
# Preconditioner setup
################################################################################

@info "Setting up preconditioner (LINEAR_SOLVER=$LINEAR_SOLVER)"
flush(stdout); flush(stderr)

if LINEAR_SOLVER == "Pardiso"
    error_msg = """
        Q_precond is not structurally symmetric (nnz=$(nnz(Q_precond))).
        The transport matrix M should have symmetric sparsity from matrix_setup.jl.
        Check that no operation (dropzeros, scalar multiply, etc.) broke the pattern.
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
if !@isdefined(TMPreconditioner)
    struct TMPreconditioner
        prob
    end
end

Plprob = LinearProblem(Q_precond, ones(size(Q_precond, 1)))
Plprob = init(Plprob, linear_solver, rtol = 1.0e-12)
Pl = TMPreconditioner(Plprob)

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

Pr = I
precs = Returns((Pl, Pr))

@info "Preconditioner ready"
flush(stdout); flush(stderr)

################################################################################
# JVP setup
################################################################################

if JVP_METHOD == "matrix"
    @info "Using matrix-based JVP: J ≈ stop_time * M (sparse matvec)"
    flush(stdout); flush(stderr)

    # M = ∂x/∂t
    # ϕ(x(t)) = x(t + 1year) = x(t) + ∫ ∂x/∂t dt ≈ x(t) + Δt M x(t)
    # G(x) = ϕ(x) - x ≈ Δt M x(t)
    # The true Jacobian of G(x) = Φ(x) - x is J_G = exp(M*T) - I ≈ M*T
    # (first-order approximation). Using this avoids expensive G! evaluations
    # during GMRES iterations (sparse matvec vs full year simulation).
    MT = copy(M)
    MT.nzval .*= stop_time

    function approximate_jvp!(Jv, v, u, p)
        return mul!(Jv, MT, v)
    end

    f! = NonlinearFunction(G!; jvp = approximate_jvp!)
    newton_solver = NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, gmres_restart = 50, rtol = 1.0e-4),
    )

elseif JVP_METHOD == "exact"
    @info "Using exact JVP via linear tracer (linΦ!)"
    flush(stdout); flush(stderr)

    f! = NonlinearFunction(G!; jvp = jvp!)
    newton_solver = NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, gmres_restart = 50, rtol = 1.0e-4),
    )

elseif JVP_METHOD == "finitediff"
    @info "Using finite-difference JVP (AutoFiniteDiff)"
    @warn "This requires extra G! evaluations per GMRES iteration — much slower than matrix JVP"
    flush(stdout); flush(stderr)

    f! = NonlinearFunction(G!)
    newton_solver = NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = precs, gmres_restart = 50, rtol = 1.0e-4),
        jvp_autodiff = AutoFiniteDiff(),
    )
end

################################################################################
# Nonlinear solve: Newton-GMRES
################################################################################

@info "Solving nonlinear problem with Newton-GMRES"
@info "- JVP method: $JVP_METHOD"
@info "- Preconditioner: $(LUMP_AND_SPRAY ? "lump-and-spray (Bardin et al., 2014)" : "direct Q⁻¹ - I")"
@info "- Linear solver: $LINEAR_SOLVER"
@info "- abstol = 0.001 years (volume-weighted RMS norm)"
flush(stdout); flush(stderr)

age_init_vec = load_initial_age(idx, Nidx, outputdir, model_config; year)
nonlinearprob = NonlinearProblem(f!, age_init_vec)

@time sol = solve(
    nonlinearprob,
    newton_solver;
    # internalnorm = vol_norm,
    show_trace = Val(true),
    reltol = Inf,
    abstol = 0.0001year,
    verbose = true,
)

@info "Newton-GMRES solve complete" retcode = sol.retcode total_G_calls = g_call_count[] total_jvp_calls = jvp_call_count[]
flush(stdout); flush(stderr)

################################################################################
# Save result
################################################################################

@info "Saving steady-state age"
flush(stdout); flush(stderr)

age_steady_3D = zeros(Float64, Nx′, Ny′, Nz′)
age_steady_3D[idx] .= sol.u

vol_mean = sum(sol.u .* v1D) / sum(v1D) / year
@info "Volume-weighted mean periodic steady age: $vol_mean years"

steady_dir = solver_output_dir
steady_file = joinpath(steady_dir, "age_$(LINEAR_SOLVER)_$(lumpspray_tag).jld2")
jldsave(steady_file; age = age_steady_3D, wet3D, idx)
@info "Saved steady-state age to $steady_file"
flush(stdout); flush(stderr)

@info "solve_periodic_NK.jl complete"
flush(stdout); flush(stderr)
