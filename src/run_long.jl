"""
Run a long brute-force offline age simulation (default: 3000 years).

This script runs the model for many thousands of years to approach
steady-state age, saving checkpoints every 100 years.

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=48:00:00 -l ncpus=12 -l ngpus=1 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/run_long.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  NYEARS           – total simulation length in years  (default: 3000)
"""

include("setup_model.jl")

using LinearAlgebra: norm

################################################################################
# Configuration
################################################################################

NYEARS = parse(Int, get(ENV, "NYEARS", "3000"))
CHECKPOINT_INTERVAL_YEARS = 100

# Override stop_time for long simulation
stop_time = NYEARS * 12 * prescribed_Δt
checkpoint_interval = CHECKPOINT_INTERVAL_YEARS * 12 * prescribed_Δt

@info "Long simulation configuration"
@info "- NYEARS = $NYEARS"
@info "- CHECKPOINT_INTERVAL = $CHECKPOINT_INTERVAL_YEARS years"
@info "- stop_time = $(stop_time / year) years"
flush(stdout)

################################################################################
# Initial condition
################################################################################

@info "Setting initial condition: age = 0"
flush(stdout)

set!(model, age = Returns(0.0))

################################################################################
# Wet cell mask (for checkpoint saving)
################################################################################

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)

grid_cpu = on_architecture(CPU(), grid)
v1D = interior(compute_volume(grid_cpu))[idx]
inv_sumv = 1 / sum(v1D)

@info "Number of wet cells: $Nidx"
flush(stdout)

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
        "Iteration: %04d, time: %1.3f yr, Δt: %.2e yr, max(age) = %.1e yr at (%d, %d, %d), mean(age) = %.1e yr, wall: %s\n",
        iteration(sim), time(sim) / year, sim.Δt / year, max_age, idx_max.I..., mean_age, walltime
    )
end

# Progress every year
add_callback!(simulation, progress_message, TimeInterval(12 * prescribed_Δt))

################################################################################
# Checkpoint saving callback
################################################################################

age_output_dir = joinpath(outputdir, "age", model_config)
mkpath(age_output_dir)

prev_mean_age = Ref(0.0)

function save_checkpoint(sim)
    elapsed_years = round(Int, time(sim) / year)

    # Extract age field to CPU
    age_data = Array(interior(model.tracers.age))
    age_wet = age_data[idx]

    # Volume-weighted mean age (in years)
    mean_age = sum(age_wet .* v1D) * inv_sumv / year
    max_age = maximum(age_wet) / year

    # Convergence: relative change since last checkpoint
    if prev_mean_age[] > 0
        rel_change = abs(mean_age - prev_mean_age[]) / prev_mean_age[]
        @info @sprintf(
            "CHECKPOINT at year %d: mean_age = %.2f yr, max_age = %.2f yr, Δ(mean_age)/mean_age = %.2e",
            elapsed_years, mean_age, max_age, rel_change
        )
    else
        @info @sprintf(
            "CHECKPOINT at year %d: mean_age = %.2f yr, max_age = %.2f yr",
            elapsed_years, mean_age, max_age
        )
    end
    prev_mean_age[] = mean_age

    # Save 3D age field
    age_3D = zeros(Float64, Nx′, Ny′, Nz′)
    age_3D[idx] .= age_wet

    checkpoint_file = joinpath(age_output_dir, "age_long_$(NYEARS)years_checkpoint_$(elapsed_years).jld2")
    jldsave(checkpoint_file; age = age_3D, wet3D, idx, elapsed_years)
    @info "Saved checkpoint to $checkpoint_file"
    return flush(stdout)
end

add_callback!(simulation, save_checkpoint, TimeInterval(checkpoint_interval))

################################################################################
# Run
################################################################################

@info "Running $(NYEARS)-year simulation"
@info "Output directory: $age_output_dir"
flush(stdout)

run!(simulation)

@info "$(NYEARS)-year simulation complete"
flush(stdout)

################################################################################
# Validate age field
################################################################################

@info "Validating age field after $(NYEARS)-year simulation"
flush(stdout)

age_data = Array(interior(model.tracers.age))
elapsed_time = time(simulation)
age_wet = age_data[idx]

# Volume-weighted mean age
final_mean_age = sum(age_wet .* v1D) * inv_sumv / year
final_max_age = maximum(age_wet) / year

# Max age bound
max_age_ratio = maximum(age_wet) / elapsed_time
@info "Max age bound check:" max_age_years = final_max_age ratio_to_elapsed = max_age_ratio
if max_age_ratio > 1.1
    @warn "Max age exceeds 1.1× elapsed time — possible numerical overshoot or bug"
end

# Surface age
surface_mask = wet3D[:, :, end]
surface_ages = age_data[:, :, end][surface_mask]
@info "Surface age:" max_days = maximum(abs, surface_ages) / day mean_days = mean(surface_ages) / day

# Non-negativity
n_negative = count(x -> x < 0, age_wet)
@info "Non-negativity:" n_negative min_age_days = minimum(age_wet) / day
if n_negative > 0
    @warn "Found $n_negative negative age values"
end

# Summary
@info "Final summary:" elapsed_years = NYEARS vol_mean_age_years = final_mean_age max_age_years = final_max_age n_wet_cells = Nidx
flush(stdout)

@info "run_long.jl complete"
flush(stdout)
