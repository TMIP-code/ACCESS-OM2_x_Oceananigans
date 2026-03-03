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
  ACCELERATION_METHOD – speedmapping | anderson  (default: speedmapping)
                        speedmapping: Alternating Cyclic Extrapolation (SpeedMapping.jl)
                        anderson:     Anderson acceleration (NLsolve.jl)
  AA_M               – Anderson history size (default: 10; literature recommends 5–10)
  AA_BETA            – Anderson damping parameter (default: 1.0; try 0.5 for slow convergence)
  WARM_START_FILE    – JLD2 file with an "age" field to use as initial guess (default: empty = zeros)
"""

include("setup_model.jl")

using NonlinearSolve
using SpeedMapping  # required for NonlinearSolve's SpeedMappingJL() extension
using NLsolve       # required for NonlinearSolve's NLsolveJL() extension
using LinearAlgebra

# Verify package extensions loaded (Julia extension trigger can fail on HPC)
if !isdefined(NonlinearSolve, :SpeedMappingJL)
    error("SpeedMapping.jl extension not loaded by NonlinearSolve. Try: using Pkg; Pkg.precompile() and restart Julia.")
end
if !isdefined(NonlinearSolve, :NLsolveJL)
    error("NLsolve.jl extension not loaded by NonlinearSolve. Try: using Pkg; Pkg.precompile() and restart Julia.")
end

################################################################################
# Configuration
################################################################################

ACCELERATION_METHOD = get(ENV, "ACCELERATION_METHOD", "speedmapping")
(ACCELERATION_METHOD ∈ ("speedmapping", "anderson")) || error("ACCELERATION_METHOD must be one of: speedmapping, anderson (got: $ACCELERATION_METHOD)")

AA_M = parse(Int, get(ENV, "AA_M", "10"))
AA_BETA = parse(Float64, get(ENV, "AA_BETA", "1.0"))

@info "Fixed-point periodic solver configuration"
@info "- ACCELERATION_METHOD = $ACCELERATION_METHOD"
@info "- AA_M = $AA_M (Anderson history size)"
@info "- AA_BETA = $AA_BETA (Anderson damping)"
flush(stdout)

################################################################################
# Common solver infrastructure (simulation, wet mask, buffers, Φ!, G!)
################################################################################

include("periodic_solver_common.jl")

################################################################################
# Warm-start from previous solution (optional)
################################################################################

age_init_vec = zeros(Nidx)

WARM_START_FILE = get(ENV, "WARM_START_FILE", "")
if !isempty(WARM_START_FILE)
    if isfile(WARM_START_FILE)
        @info "Loading warm-start initial guess from $WARM_START_FILE"
        flush(stdout)
        warm_data = jldopen(WARM_START_FILE)
        age_warm = warm_data["age"]
        close(warm_data)
        age_init_vec .= view(age_warm, idx)
        @info "Warm-start loaded" norm = norm(age_init_vec) max = maximum(abs, age_init_vec)
    else
        @warn "WARM_START_FILE not found: $WARM_START_FILE — starting from zeros"
    end
else
    @info "Starting from zero initial guess (set WARM_START_FILE to warm-start)"
end
flush(stdout)

################################################################################
# Nonlinear solve: fixed-point acceleration
################################################################################

@info "Solving nonlinear problem with fixed-point acceleration ($ACCELERATION_METHOD)"
@info "- abstol = $(0.001 * stop_time) seconds ($(0.001 * stop_time / day) days)"
flush(stdout)

f! = NonlinearFunction(G!)
nonlinearprob = NonlinearProblem(f!, age_init_vec, [])

if ACCELERATION_METHOD == "speedmapping"
    @info "Using SpeedMappingJL (Alternating Cyclic Extrapolation)"
    flush(stdout)
    solver = SpeedMappingJL()
elseif ACCELERATION_METHOD == "anderson"
    @info "Using NLsolveJL with Anderson acceleration (m=$AA_M, beta=$AA_BETA)"
    flush(stdout)
    solver = NLsolveJL(; method = :anderson, m = AA_M, beta = AA_BETA)
end

@time sol = solve(
    nonlinearprob,
    solver;
    show_trace = Val(true),
    reltol = Inf,
    abstol = 0.001 * stop_time,
    maxiters = 1000,
    verbose = true,
)

@info "Fixed-point solve complete" retcode = sol.retcode total_G_calls = g_call_count[]
flush(stdout)

################################################################################
# Save result
################################################################################

@info "Saving steady-state age"
flush(stdout)

age_steady_3D = zeros(Float64, Nx′, Ny′, Nz′)
age_steady_3D[idx] .= sol.u

steady_dir = joinpath(outputdir, "age", model_config)
mkpath(steady_dir)
steady_file = joinpath(steady_dir, "age_$(ACCELERATION_METHOD).jld2")
jldsave(steady_file; age = age_steady_3D, wet3D, idx)
@info "Saved steady-state age to $steady_file"
flush(stdout)

@info "solve_periodic_anderson.jl complete"
flush(stdout)
