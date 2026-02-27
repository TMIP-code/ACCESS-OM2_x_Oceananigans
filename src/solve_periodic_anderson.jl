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
"""

include("setup_model.jl")

using NonlinearSolve
using SpeedMapping  # required for NonlinearSolve's SpeedMappingJL() extension
using NLsolve       # required for NonlinearSolve's NLsolveJL() extension
using LinearAlgebra
using Oceananigans.Simulations: reset!

################################################################################
# Configuration
################################################################################

ACCELERATION_METHOD = get(ENV, "ACCELERATION_METHOD", "speedmapping")
(ACCELERATION_METHOD ∈ ("speedmapping", "anderson")) || error("ACCELERATION_METHOD must be one of: speedmapping, anderson (got: $ACCELERATION_METHOD)")

@info "Fixed-point periodic solver configuration"
@info "- ACCELERATION_METHOD = $ACCELERATION_METHOD"
flush(stdout)

################################################################################
# Simulation (minimal output — no field writers during periodic solve)
################################################################################

@info "Creating simulation (no output writers)"
flush(stdout)

set!(model, age = Returns(0.0))

simulation = Simulation(model; Δt, stop_time)

function progress_message(sim)
    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year)
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout)
    return @info @sprintf(
        "  sim iter: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d), mean(age) = %.1e, wall: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age, idx_max.I..., mean_age, walltime
    )
end

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

################################################################################
# Compute wet cell indexing
################################################################################

@info "Computing wet cell mask"
flush(stdout)

fNaN = CenterField(grid)
mask_immersed_field!(fNaN, NaN)
wet3D = .!isnan.(interior(on_architecture(CPU(), fNaN)))
idx = findall(wet3D)
Nidx = length(idx)
@info "Number of wet cells: $Nidx"
Nx′, Ny′, Nz′ = size(wet3D)
flush(stdout)

################################################################################
# Preallocate buffers for G!
################################################################################

age3D_cpu = zeros(Float64, Nx′, Ny′, Nz′)
age3D_gpu = on_architecture(arch, zeros(Float64, Nx′, Ny′, Nz′))

################################################################################
# G! function: 1-year drift (runs simulation on GPU, solver on CPU)
################################################################################

g_call_count = Ref(0)

function G!(dage, age, p)
    g_call_count[] += 1
    call_num = g_call_count[]
    t_start = time()
    @info "G! call #$call_num starting" norm_age = norm(age) max_age = maximum(abs, age)
    flush(stdout)

    # Reset simulation for a fresh 1-year run
    reset!(simulation)
    simulation.stop_time = stop_time

    # CPU vec -> CPU 3D -> preallocated GPU 3D (no new GPU allocation)
    fill!(age3D_cpu, 0)
    age3D_cpu[idx] .= age
    copyto!(age3D_gpu, age3D_cpu)

    # Set model field and run 1-year simulation
    set!(model, age = age3D_gpu)
    run!(simulation)

    # GPU field -> CPU 3D -> CPU vec (drift = final - initial)
    age3D_cpu .= Array(interior(model.tracers.age))
    dage .= view(age3D_cpu, idx) .- age

    elapsed = time() - t_start
    @info "G! call #$call_num done ($(round(elapsed; digits = 1))s)" norm_drift = norm(dage) max_drift = maximum(abs, dage)
    flush(stdout)
    return dage
end

################################################################################
# Nonlinear solve: fixed-point acceleration
################################################################################

@info "Solving nonlinear problem with fixed-point acceleration ($ACCELERATION_METHOD)"
@info "- abstol = $(0.001 * stop_time)"
flush(stdout)

age_init_vec = zeros(Nidx)
f! = NonlinearFunction(G!)
nonlinearprob = NonlinearProblem(f!, age_init_vec, [])

if ACCELERATION_METHOD == "speedmapping"
    @info "Using SpeedMappingJL (Alternating Cyclic Extrapolation)"
    flush(stdout)
    solver = SpeedMappingJL()
elseif ACCELERATION_METHOD == "anderson"
    @info "Using NLsolveJL with Anderson acceleration"
    flush(stdout)
    solver = NLsolveJL(; method = :anderson, m = 40)
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

steady_dir = joinpath(outputdir, "age", run_mode_tag)
mkpath(steady_dir)
steady_file = joinpath(steady_dir, "steady_age_$(ACCELERATION_METHOD).jld2")
jldsave(steady_file; age = age_steady_3D, wet3D, idx)
@info "Saved steady-state age to $steady_file"
flush(stdout)

@info "solve_periodic_anderson.jl complete"
flush(stdout)
