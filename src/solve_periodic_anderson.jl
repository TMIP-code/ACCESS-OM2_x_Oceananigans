"""
Solve for the periodic steady-state age using fixed-point acceleration.

Finds x such that Φ(x) = x, where Φ(x) is the result of running the model
for 1 year from initial condition x. This is equivalent to G(x) = Φ(x) - x = 0
but uses fixed-point acceleration methods (SpeedMapping/Anderson) instead of
Newton-GMRES, avoiding the need for a transport matrix or preconditioner.

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=04:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/solve_periodic_anderson.jl")
```

Environment variables (in addition to setup_model.jl):
  AA_SOLVER            – SpeedMapping | NLsolve | SIAMFANL | FixedPoint  (default: SpeedMapping)
                         SpeedMapping: Alternating Cyclic Extrapolation (SpeedMapping.jl)
                         NLsolve:      Anderson acceleration (NLsolve.jl)
                         SIAMFANL:     Anderson acceleration (SIAMFANLEquations.jl)
                         FixedPoint:   Anderson acceleration (FixedPointAcceleration.jl)
  NLSAA_M            – Anderson history size (default: 10; literature recommends 5–10)
  NLSAA_BETA         – Anderson damping parameter (default: 1.0; try 0.5 for slow convergence)
  SMAA_SIGMA_MIN     – SpeedMapping minimum σ (default: 0.0; setting to 1 may avoid stalling)
  SMAA_STABILIZE     – Stabilization mapping before extrapolation (default: no; yes/no)
  SMAA_CHECK_OBJ     – Restart at best past iterate on NaN/Inf (default: no; yes/no)
  SMAA_ORDERS        – Alternating order sequence (default: 332; each digit 1–3)
  FPAA_M             – FixedPointAcceleration Anderson history size (default: 10)
  WARM_START_FILE    – JLD2 file with an "age" field to use as initial guess (default: empty = zeros)
"""

include("setup_model.jl")

using NonlinearSolve             # for SpeedMappingJL / FixedPointAccelerationJL wrappers
using SpeedMapping               # required for NonlinearSolve's SpeedMappingJL() extension
using NLsolve                    # called directly (NonlinearSolve wrapper has dense Jacobian bug)
using SIAMFANLEquations          # called directly (NonlinearSolve wrapper has dense N×N allocation bug)
using FixedPointAcceleration     # required for NonlinearSolve's FixedPointAccelerationJL() extension
using LinearAlgebra

################################################################################
# Configuration
################################################################################

AA_SOLVER = get(ENV, "AA_SOLVER", "SpeedMapping")
(AA_SOLVER ∈ ("SpeedMapping", "NLsolve", "SIAMFANL", "FixedPoint")) || error("AA_SOLVER must be one of: SpeedMapping, NLsolve, SIAMFANL, FixedPoint (got: $AA_SOLVER)")

NLSAA_M = parse(Int, get(ENV, "NLSAA_M", "10"))
NLSAA_BETA = parse(Float64, get(ENV, "NLSAA_BETA", "1.0"))

SMAA_SIGMA_MIN = parse(Float64, get(ENV, "SMAA_SIGMA_MIN", "0.0"))
SMAA_STABILIZE = lowercase(get(ENV, "SMAA_STABILIZE", "no")) == "yes"
SMAA_CHECK_OBJ = lowercase(get(ENV, "SMAA_CHECK_OBJ", "no")) == "yes"
SMAA_ORDERS = parse.(Int, collect(get(ENV, "SMAA_ORDERS", "332")))

FPAA_M = parse(Int, get(ENV, "FPAA_M", "10"))

@info "Fixed-point periodic solver configuration"
@info "- AA_SOLVER = $AA_SOLVER"
@info "- NLSAA_M = $NLSAA_M (Anderson history size)"
@info "- NLSAA_BETA = $NLSAA_BETA (Anderson damping)"
@info "- SMAA_SIGMA_MIN = $SMAA_SIGMA_MIN"
@info "- SMAA_STABILIZE = $SMAA_STABILIZE"
@info "- SMAA_CHECK_OBJ = $SMAA_CHECK_OBJ"
@info "- SMAA_ORDERS = $SMAA_ORDERS"
@info "- FPAA_M = $FPAA_M (FixedPoint Anderson history size)"
flush(stdout); flush(stderr)

################################################################################
# Common solver infrastructure (simulation, wet mask, buffers, Φ!, G!)
################################################################################

include("periodic_solver_common.jl")

################################################################################
# Initial age (INITIAL_AGE env var — see periodic_solver_common.jl)
################################################################################

age_init_vec = load_initial_age(idx, Nidx, outputdir, model_config; year)

################################################################################
# Nonlinear solve: fixed-point acceleration
################################################################################

@info "Solving nonlinear problem with fixed-point acceleration ($AA_SOLVER)"
flush(stdout); flush(stderr)

if AA_SOLVER in ("SpeedMapping", "FixedPoint")
    # --- Via NonlinearSolve (wrappers are fine) ---
    @info "- abstol = 0.001 years (volume-weighted RMS norm)"
    flush(stdout); flush(stderr)

    f! = NonlinearFunction(G!)
    nonlinearprob = NonlinearProblem(f!, age_init_vec, nothing)

    if AA_SOLVER == "SpeedMapping"
        @info "Using SpeedMappingJL (Alternating Cyclic Extrapolation)"
        flush(stdout); flush(stderr)
        solver = SpeedMappingJL(; σ_min = SMAA_SIGMA_MIN, stabilize = SMAA_STABILIZE, check_obj = SMAA_CHECK_OBJ, orders = SMAA_ORDERS)
    else
        @info "Using FixedPointAccelerationJL with Anderson acceleration (m=$FPAA_M)"
        flush(stdout); flush(stderr)
        solver = FixedPointAccelerationJL(; algorithm = :Anderson, m = FPAA_M)
    end

    @time sol = solve(
        nonlinearprob,
        solver;
        internalnorm = vol_norm,
        show_trace = Val(true),
        reltol = Inf,
        abstol = 0.001,
        maxiters = 1000,
        verbose = true,
    )

    sol_vec = sol.u
    retcode_str = string(sol.retcode)

elseif AA_SOLVER == "NLsolve"
    # --- Direct call (bypasses NonlinearSolve dense Jacobian allocation bug) ---
    @info "Using NLsolve.nlsolve directly with Anderson acceleration (m=$NLSAA_M, beta=$NLSAA_BETA)"
    @info "- ftol = $(0.001 * year) seconds (inf-norm ≈ 0.001 years)"
    flush(stdout); flush(stderr)

    G_nlsolve!(F, x) = G!(F, x, nothing)
    @time result = NLsolve.nlsolve(
        G_nlsolve!, age_init_vec;
        method = :anderson,
        m = NLSAA_M,
        beta = NLSAA_BETA,
        ftol = 0.001 * year,
        iterations = 1000,
        show_trace = true,
    )

    sol_vec = result.zero
    retcode_str = NLsolve.converged(result) ? "Success" : "MaxIters"

elseif AA_SOLVER == "SIAMFANL"
    # --- Direct call (bypasses NonlinearSolve dense N×N allocation bug) ---
    @info "Using SIAMFANLEquations.aasol directly with Anderson acceleration (m=$NLSAA_M, beta=$NLSAA_BETA)"
    @info "- atol = $(0.001 * year) seconds (2-norm ≈ 0.001 years)"
    flush(stdout); flush(stderr)

    Φ_siamfanl!(G, x) = Φ!(G, x, nothing)
    Vstore = zeros(Float64, Nidx, 3 * NLSAA_M + 3)
    @time result = SIAMFANLEquations.aasol(
        Φ_siamfanl!, age_init_vec, NLSAA_M, Vstore;
        maxit = 1000,
        rtol = 0.0,
        atol = 0.001 * year,
        beta = NLSAA_BETA,
    )

    sol_vec = result.solution
    retcode_str = result.idid ? "Success" : "Failure(errcode=$(result.errcode))"
end

@info "Fixed-point solve complete" retcode = retcode_str total_G_calls = g_call_count[]
flush(stdout); flush(stderr)

################################################################################
# Save result
################################################################################

@info "Saving steady-state age"
flush(stdout); flush(stderr)

age_steady_3D = zeros(Float64, Nx′, Ny′, Nz′)
age_steady_3D[idx] .= sol_vec

vol_mean = sum(sol_vec .* v1D) / sum(v1D) / year
@info "Volume-weighted mean periodic steady age: $vol_mean years"

steady_dir = joinpath(outputdir, "age", model_config)
mkpath(steady_dir)
steady_file = joinpath(steady_dir, "age_$(AA_SOLVER).jld2")
jldsave(steady_file; age = age_steady_3D, wet3D, idx)
@info "Saved steady-state age to $steady_file"
flush(stdout); flush(stderr)

@info "solve_periodic_anderson.jl complete"
flush(stdout); flush(stderr)
