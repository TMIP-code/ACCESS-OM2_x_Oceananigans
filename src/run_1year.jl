"""
Run a standalone 1-year offline age simulation.

This is a lightweight test/debug script that runs the model for one year
and saves full output (age, u, v, w, η).

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
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

include("setup_model.jl")

################################################################################
# Initial condition
################################################################################

@info "Setting initial condition: age = 0"
flush(stdout); flush(stderr)

set!(model, age = Returns(0.0))

################################################################################
# Simulation
################################################################################

@info "Creating simulation"
flush(stdout); flush(stderr)

simulation = Simulation(
    model;
    Δt,
    stop_time,
)

function progress_message(sim)
    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year) # in years
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout); flush(stderr)
    return @info @sprintf(
        "Iteration: %04d, time: %1.3f yr, Δt: %.2e yr, max(age)/time = %.1e at (%d, %d, %d), mean(age) = %.1e yr, wall time: %s\n",
        iteration(sim), time(sim) / year, sim.Δt / year, max_age / (time(sim) / year), idx_max.I..., mean_age, walltime
    )
end

add_callback!(simulation, progress_message, TimeInterval(prescribed_Δt))

output_fields = Dict(
    "age" => model.tracers.age,
    "u" => model.velocities.u,
    "v" => model.velocities.v,
    "w" => model.velocities.w,
    "η" => model.free_surface.displacement,
)

age_output_dir = joinpath(outputdir, "age", model_config)
mkpath(age_output_dir)
output_prefix = joinpath(age_output_dir, "age_1year")

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(prescribed_Δt / 2),
    filename = output_prefix,
    overwrite_existing = true,
)

@info "Running 1-year simulation"
@info "Output prefix: $output_prefix"
flush(stdout); flush(stderr)

run!(simulation)

@info "1-year simulation complete"
flush(stdout); flush(stderr)

################################################################################
# Validate age field
################################################################################

using LinearAlgebra: norm

@info "Validating age field after 1-year simulation"
flush(stdout); flush(stderr)

age_data = Array(interior(model.tracers.age))
elapsed_time = time(simulation)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)
age_wet = age_data[idx]

# ── Test 1: Max age bound ────────────────────────────────────────────────
# With source_rate = 1 and divergence-free flow, no cell should exceed age = elapsed_time.
# Violations indicate numerical overshoot (Centered scheme) or a bug.
max_age_val = maximum(age_wet)
max_age_ratio = max_age_val / elapsed_time
@info "Max age bound check:" max_age_years = max_age_val / year ratio_to_elapsed = max_age_ratio
if max_age_ratio > 1.1
    @warn "Max age exceeds 1.1× elapsed time — possible numerical overshoot or bug"
end

# ── Test 2: Surface age should be near zero ──────────────────────────────
surface_mask = wet3D[:, :, end]   # k = Nz = surface
surface_ages = age_data[:, :, end][surface_mask]
max_surface_age = maximum(abs, surface_ages)
mean_surface_age = mean(surface_ages)
@info "Surface age:" max_days = max_surface_age / day mean_days = mean_surface_age / day
if max_surface_age > 1day
    @warn "Surface age exceeds 1 day — surface relaxation may not be working correctly"
end

# ── Test 3: Non-negativity ───────────────────────────────────────────────
n_negative = count(x -> x < 0, age_wet)
min_age_val = minimum(age_wet)
@info "Non-negativity:" n_negative fraction = n_negative / Nidx min_age_days = min_age_val / day
if n_negative > 0
    @warn "Found $n_negative negative age values (min = $(min_age_val / day) days) — advection scheme ($ADVECTION_SCHEME) may produce oscillations"
end

# ── Test 4: Depth-averaged profile ───────────────────────────────────────
z_centers = Array(znodes(grid, Center(), Center(), Center()))
@info "Volume-weighted mean age by depth level:"
flush(stdout); flush(stderr)
for k in Nz′:-1:1
    level_mask = wet3D[:, :, k]
    n_wet = count(level_mask)
    if n_wet == 0
        continue
    end
    level_ages = age_data[:, :, k][level_mask]
    level_mean = mean(level_ages)
    level_max = maximum(level_ages)
    level_min = minimum(level_ages)
    z_val = z_centers[k]
    @info @sprintf(
        "  k=%3d  z=%7.0fm  mean=%6.2f yr  max=%6.2f yr  min=%6.2f yr  n_wet=%d",
        k, z_val, level_mean / year, level_max / year, level_min / year, n_wet
    )
end
flush(stdout); flush(stderr)

# ── Test 5: Hotspot inspection ───────────────────────────────────────────
max_idx = argmax(age_data)
mi, mj, mk = Tuple(max_idx)
@info "Max age hotspot at (i=$mi, j=$mj, k=$mk):" age_years = age_data[mi, mj, mk] / year z_meters = z_centers[mk]
# Print neighbors (±1) to see if it's an isolated spike or regional
for dk in -1:1, dj in -1:1, di in -1:1
    ni, nj, nk = mi + di, mj + dj, mk + dk
    if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′ && wet3D[ni, nj, nk]
        @info @sprintf("  neighbor (%+d,%+d,%+d): age = %6.2f yr", di, dj, dk, age_data[ni, nj, nk] / year)
    end
end
flush(stdout); flush(stderr)

# ── Summary ──────────────────────────────────────────────────────────────
@info "Validation summary:" max_age_years = max_age_val / year mean_wet_age_years = mean(age_wet) / year max_surface_days = max_surface_age / day n_negative n_wet_cells = Nidx
flush(stdout); flush(stderr)

@info "run_1year.jl complete"
@info "Run plot_1year_age.jl on CPU to generate age diagnostic plots"
flush(stdout); flush(stderr)
