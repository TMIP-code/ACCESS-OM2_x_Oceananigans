"""
Diagnose w from continuity using prescribed u, v, η and save as monthly FTS.

Runs a short simulation with `DiagnosticVerticalVelocity` to compute w at each
of the 12 monthly FTS snapshot times. The z-star machinery (∂t_σ, σ) is properly
initialized via `time_step!`, ensuring w is identical to what the age simulation
would compute at runtime with `W_FORMULATION=wdiagnosed`.

Requires: create_grid.jl and create_velocities.jl to have been run first.

Usage:
    julia --project src/diagnose_w.jl

Output:
    preprocessed_inputs/{PM}/{EXP}/{TW}/monthly/w_diagnosed_monthly.jld2
"""

@info "Loading packages for w diagnosis"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.OutputReaders: Cyclical, InMemory, OnDisk
using Oceananigans.Units: days, seconds
year = years = 365.25days
month = months = year / 12

using JLD2
using TOML

include("select_architecture.jl")
include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, monthly_dir, yearly_dir) = load_project_config()
(; VELOCITY_SOURCE) = parse_config_env()

# Timestep (same as create_velocities.jl)
Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds

# FTS timing (same as create_velocities.jl)
prescribed_Δt = 1month
fts_times = ((1:12) .- 0.5) * prescribed_Δt
stop_time = 12 * prescribed_Δt

@info "Configuration:"
@info "  PARENT_MODEL     = $parentmodel"
@info "  VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "  Δt               = $(Δt / seconds) s"
@info "  FTS times        = $(fts_times ./ days) days"
flush(stdout); flush(stderr)

################################################################################
# Load grid
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, arch)

################################################################################
# Load u, v, η FTS (InMemory for speed)
################################################################################

backend = InMemory()
time_indexing = Cyclical(stop_time)

if VELOCITY_SOURCE == "cgridtransports"
    u_file = joinpath(monthly_dir, "u_from_mass_transport_monthly.jld2")
    v_file = joinpath(monthly_dir, "v_from_mass_transport_monthly.jld2")
elseif VELOCITY_SOURCE == "bgridvelocities"
    u_file = joinpath(monthly_dir, "u_interpolated_monthly.jld2")
    v_file = joinpath(monthly_dir, "v_interpolated_monthly.jld2")
end
η_file = joinpath(monthly_dir, "eta_monthly.jld2")

@info "Loading u FTS from $u_file"
flush(stdout); flush(stderr)
u_fts = FieldTimeSeries(u_file, "u"; architecture = arch, grid, backend, time_indexing)

@info "Loading v FTS from $v_file"
flush(stdout); flush(stderr)
v_fts = FieldTimeSeries(v_file, "v"; architecture = arch, grid, backend, time_indexing)

@info "Loading η FTS from $η_file"
flush(stdout); flush(stderr)
η_fts = FieldTimeSeries(η_file, "η"; architecture = arch, grid, backend, time_indexing)

################################################################################
# Build model with DiagnosticVerticalVelocity
################################################################################

@info "Building model with DiagnosticVerticalVelocity"
flush(stdout); flush(stderr)

velocities = PrescribedVelocityFields(
    u = u_fts, v = v_fts,
    formulation = DiagnosticVerticalVelocity(),
)
free_surface = PrescribedFreeSurface(displacement = η_fts)

model = HydrostaticFreeSurfaceModel(
    grid;
    velocities,
    free_surface,
    tracers = (:dummy,),  # need at least one tracer for the model to work
    closure = nothing,
    buoyancy = nothing,
    coriolis = nothing,
)

@info "Model built"
flush(stdout); flush(stderr)

################################################################################
# Set up simulation with w output at FTS times
################################################################################

w_output_file = joinpath(monthly_dir, "w_diagnosed_monthly")
@info "Output file: $(w_output_file).jld2"
flush(stdout); flush(stderr)

simulation = Simulation(model; Δt, stop_time)

# Save w at half-monthly intervals (24 snapshots per year).
# More snapshots than the 12 monthly FTS gives better interpolation.
half_month = prescribed_Δt / 2
simulation.output_writers[:w] = JLD2Writer(
    model, Dict("w" => model.velocities.w);
    schedule = TimeInterval(half_month),
    filename = w_output_file,
    overwrite_existing = true,
    with_halos = true,
    including = [],
)

# Progress callback
wall_time = Ref(time_ns())
function progress_message(sim)
    elapsed = (time_ns() - wall_time[]) * 1.0e-9 / 60
    @info @sprintf(
        "  diagnose_w iter: %04d, time: %.3f yr, Δt: %.2e yr, wall: %.3f minutes",
        iteration(sim), time(sim) / year, sim.Δt / year, elapsed,
    )
    flush(stdout)
    return flush(stderr)
end
add_callback!(simulation, progress_message, IterationInterval(100))

@info "Running w diagnosis simulation ($(stop_time / year) yr, Δt = $(Δt / seconds) s)"
flush(stdout); flush(stderr)

run!(simulation)

@info "W diagnosis complete"
@info "Output saved to $(w_output_file).jld2"
flush(stdout); flush(stderr)

# Verify saved snapshots
jldopen("$(w_output_file).jld2", "r") do f
    iters = filter(k -> k != "serialized", keys(f["timeseries/w"]))
    @info "Saved $(length(iters)) w snapshots"
    for iter in sort(iters; by = k -> parse(Int, k))
        t = f["timeseries/t/$iter"]
        @info @sprintf("  iter %s: t = %.3f days = %.6f yr", iter, t / days, t / year)
    end
end
flush(stdout); flush(stderr)

@info "diagnose_w.jl complete"
flush(stdout); flush(stderr)
