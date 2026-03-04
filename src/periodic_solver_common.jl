"""
Common infrastructure for periodic steady-state solvers.

Included after `setup_model.jl` — assumes its globals are in scope:
  model, grid, arch, Δt, stop_time, prescribed_Δt, progress_message,
  year, on_architecture, compute_wet_mask
"""

using LinearAlgebra: norm
using Oceananigans.Simulations: reset!
using Printf: @sprintf

################################################################################
# Trace solver history configuration
################################################################################

TRACE_SOLVER_HISTORY = lowercase(get(ENV, "TRACE_SOLVER_HISTORY", "no")) == "yes"

if TRACE_SOLVER_HISTORY
    trace_dir = joinpath(outputdir, "age", model_config, "trace")
    mkpath(trace_dir)
    @info "TRACE_SOLVER_HISTORY enabled — saving iterates to $trace_dir"
else
    trace_dir = ""
    @info "TRACE_SOLVER_HISTORY disabled (set TRACE_SOLVER_HISTORY=yes to enable)"
end
flush(stdout)

################################################################################
# Simulation (no output writers)
################################################################################

@info "Creating simulation (no output writers)"
flush(stdout)

set!(model, age = Returns(0.0))

simulation = Simulation(model; Δt, stop_time)

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

################################################################################
# Compute wet cell mask & preallocate CPU/GPU buffers
################################################################################

@info "Computing wet cell mask"
flush(stdout)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
@info "Number of wet cells: $Nidx"
Nx′, Ny′, Nz′ = size(wet3D)
flush(stdout)

age3D_cpu = zeros(Float64, Nx′, Ny′, Nz′)
age3D_gpu = on_architecture(arch, zeros(Float64, Nx′, Ny′, Nz′))

################################################################################
# Forward map Φ! and residual G!
################################################################################

g_call_count = Ref(0)

"""
    Φ!(age_out, age_in, p)

1-year forward map: runs the simulation for 1 year starting from `age_in`
(wet-cell vector) and writes the final age into `age_out` (wet-cell vector).
"""
function Φ!(age_out, age_in, p)
    g_call_count[] += 1
    call_num = g_call_count[]
    t_start = time()
    @info "Φ! call #$call_num starting" norm_age = norm(age_in) max_age = maximum(abs, age_in) / year
    flush(stdout)

    # Reset simulation for a fresh 1-year run
    reset!(simulation)
    simulation.stop_time = stop_time

    # CPU vec → CPU 3D → GPU 3D
    fill!(age3D_cpu, 0)
    age3D_cpu[idx] .= age_in
    copyto!(age3D_gpu, age3D_cpu)

    # Set initial condition and attach trace writer
    set!(model, age = age3D_gpu)

    if TRACE_SOLVER_HISTORY
        iter_str = @sprintf("%04d", call_num)
        trace_prefix = joinpath(trace_dir, "age_trace_iter_$(iter_str)")
        simulation.output_writers[:trace] = JLD2Writer(
            model, Dict("age" => model.tracers.age);
            schedule = TimeInterval(stop_time),
            filename = trace_prefix,
            overwrite_existing = true,
        )
    end

    # Run 1-year simulation
    run!(simulation)

    # Remove trace writer before next iteration
    if TRACE_SOLVER_HISTORY
        delete!(simulation.output_writers, :trace)
    end

    # GPU field → CPU 3D → CPU vec  (copyto! avoids intermediate Array allocation)
    copyto!(age3D_cpu, interior(model.tracers.age))
    age_out .= view(age3D_cpu, idx)

    elapsed = time() - t_start
    @info "Φ! call #$call_num done ($(round(elapsed; digits = 1))s)"
    flush(stdout)
    return age_out
end

"""
    G!(dage, age, p)

1-year drift residual: G(x) = Φ(x) − x.
"""
function G!(dage, age, p)
    Φ!(dage, age, p)
    dage .-= age
    @info "G! residual" norm_drift = norm(dage) max_drift = maximum(abs, dage) / year mean_drift = mean(abs, dage) / year
    flush(stdout)
    return dage
end
