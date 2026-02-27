"""
Run a standalone 1-year offline age simulation.

This is a lightweight test/debug script that runs the model for one year
and saves full output (age, u, v, w, eta).

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/run_1year.jl")
```

Environment variables:
  PARENT_MODEL    – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION   – wdiagnosed | wprescribed  (default: wdiagnosed)
"""

include("setup_model.jl")

################################################################################
# Initial condition
################################################################################

@info "Setting initial condition: age = 0"
flush(stdout)

set!(model, age = Returns(0.0))

################################################################################
# Simulation
################################################################################

@info "Creating simulation"
flush(stdout)

simulation = Simulation(
    model;
    Δt,
    stop_time,
)

function progress_message(sim)
    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year) # in years
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout)
    return @info @sprintf(
        "Iteration: %04d, time: %1.3f, Δt: %.2e, max(age)/time = %.1e at (%d, %d, %d), mean(age) = %.1e, wall time: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age / (time(sim) / year), idx_max.I..., mean_age, walltime
    )
end

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

output_fields = Dict(
    "age" => model.tracers.age,
    "u" => model.velocities.u,
    "v" => model.velocities.v,
    "w" => model.velocities.w,
    "eta" => model.free_surface.displacement,
)

age_output_dir = joinpath(outputdir, "age", run_mode_tag)
mkpath(age_output_dir)
output_prefix = joinpath(age_output_dir, "offline_age_$(parentmodel)_$(arch_str)_$(run_suffix)")

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(prescribed_Δt / 2),
    filename = output_prefix,
    overwrite_existing = true,
)

@info "Running 1-year simulation"
@info "Output prefix: $output_prefix"
flush(stdout)

run!(simulation)

@info "1-year simulation complete"
flush(stdout)
