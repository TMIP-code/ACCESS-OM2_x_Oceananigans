"""
Shared model setup for ACCESS-OM2 offline age simulations.

This file is `include()`d by:
  - `run_1year.jl`              — standalone 1-year simulation
  - `solve_periodic_newton.jl`  — Newton-GMRES periodic solver
  - `solve_periodic_anderson.jl` — Anderson/SpeedMapping periodic solver

It sets up: architecture, configuration, grid, velocities, closures, and model.
It does NOT create a Simulation or set initial conditions — each downstream
script does that according to its own needs.

Environment variables:
  PARENT_MODEL    – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION   – wdiagnosed | wprescribed  (default: wdiagnosed)
"""

@info "Loading packages and functions"
flush(stdout)

using Oceananigans

# GPU / CPU detection
if contains(ENV["HOSTNAME"], "gpu")
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
    @show CUDA.versioninfo()
    arch = GPU()
    arch_str = "GPU"
else
    arch = CPU()
    arch_str = "CPU"
end
@info "Using $arch architecture"
flush(stdout)

using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode
using Oceananigans.OutputReaders: Cyclical, InMemory
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
month = months = year / 12

using Adapt: adapt
using Statistics
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using TOML
using JLD2
using Printf

################################################################################
# Configuration
################################################################################

# Determine which model profile to use. Priority:
# 1. ARGS[1] passed to julia
# 2. ENV["PARENT_MODEL"]
# 3. `defaults.parentmodel` in LocalPreferences.toml
# 4. fallback to ACCESS-OM2-1
cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
    Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds
else
    outputdir = profile["outputdir"]
    Δt = profile["dt_seconds"] * second
end

VELOCITY_SOURCE = get(ENV, "VELOCITY_SOURCE", "cgridtransports")
W_FORMULATION = get(ENV, "W_FORMULATION", "wdiagnosed")
(VELOCITY_SOURCE ∈ ("bgridvelocities", "cgridtransports")) || println("VELOCITY_SOURCE must be one of: bgridvelocities, cgridtransports")
(W_FORMULATION ∈ ("wdiagnosed", "wprescribed")) || println("W_FORMULATION must be one of: wdiagnosed, wprescribed")

run_mode_tag = "$(VELOCITY_SOURCE)_$(W_FORMULATION)"
run_suffix = run_mode_tag

@info "Run configuration"
@info "- PARENT_MODEL    = $parentmodel"
@info "- VELOCITY_SOURCE = $VELOCITY_SOURCE"
@info "- W_FORMULATION   = $W_FORMULATION"
@info "- Architecture    = $arch_str"

@show outputdir
mkpath(outputdir)

include("tripolargrid_reader.jl")

################################################################################
# Load grid from JLD2
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))

@info "Reconstructing grid (loading data from JLD2)"
flush(stdout)
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout)

################################################################################
# Load velocities from disk
################################################################################

@info "Loading velocities from disk"
flush(stdout)

if VELOCITY_SOURCE == "cgridtransports"
    flush(stdout)
    u_file = joinpath(preprocessed_inputs_dir, "u_from_mass_transport_periodic.jld2")
    v_file = joinpath(preprocessed_inputs_dir, "v_from_mass_transport_periodic.jld2")
    w_file = joinpath(preprocessed_inputs_dir, "w_from_mass_transport_periodic.jld2")
    @info """Loading velocities from MOM mass transport outputs files:
    - $(u_file)
    - $(v_file)
    - $(w_file)
    """
elseif VELOCITY_SOURCE == "bgridvelocities"
    flush(stdout)
    u_file = joinpath(preprocessed_inputs_dir, "u_interpolated_periodic.jld2")
    v_file = joinpath(preprocessed_inputs_dir, "v_interpolated_periodic.jld2")
    w_file = joinpath(preprocessed_inputs_dir, "w_periodic.jld2")
    @info """Loading velocities from MOM velocity outputs files:
    - $(u_file)
    - $(v_file)
    - $(w_file)
    """
end

N_in_mem = 4  # Keep 4 timesteps in memory (monthly data)

backend = InMemory(N_in_mem)
time_indexing = Cyclical(1year)

u_ts = FieldTimeSeries(u_file, "u"; architecture = arch, grid, backend, time_indexing)
v_ts = FieldTimeSeries(v_file, "v"; architecture = arch, grid, backend, time_indexing)
@info "Loading sea surface height from MOM output"
flush(stdout)
η_file = joinpath(preprocessed_inputs_dir, "eta_periodic.jld2")
η_ts = FieldTimeSeries(η_file, "η"; architecture = arch, grid, backend, time_indexing)

prescribed_Δt = u_ts.times[2] - u_ts.times[1]  # Infer from time spacing
fts_times = u_ts.times

@info "Velocities loaded (InMemory backend with $N_in_mem timesteps in memory)"
flush(stdout)

if W_FORMULATION == "wprescribed"
    @info "Using prescribed w field from: $(w_file)"
    isfile(w_file) || println("W_FORMULATION=wprescribed requires file: $(w_file)")
    w_ts = FieldTimeSeries(w_file, "w"; architecture = arch, grid, backend, time_indexing)

    @info "Prescribing u, v, and w"
    flush(stdout)
    velocities = PrescribedVelocityFields(u = u_ts, v = v_ts, w = w_ts)
elseif W_FORMULATION == "wdiagnosed"
    @info "Prescribing u and v; diagnosing w via continuity"
    flush(stdout)
    velocities = PrescribedVelocityFields(u = u_ts, v = v_ts, formulation = DiagnosticVerticalVelocity())
end

@info "Using prescribed sea surface height from MOM output"
flush(stdout)
free_surface = PrescribedFreeSurface(displacement = η_ts)

################################################################################
# Closures
################################################################################

@info "Creating closures"
flush(stdout)

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

# Vertical diffusivity parameters
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# Load MLD to add strong vertical diffusion in the mixed layer
mld_ds = open_dataset(joinpath(inputdir, "mld.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

implicit_vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization();
    κ = κVField
)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)

closure = (
    horizontal_diffusion,
    implicit_vertical_diffusion,
)

@info "Closures created"
flush(stdout)

################################################################################
# Model
################################################################################

@info "Building model"
flush(stdout)

age_parameters = (;
    relaxation_timescale = 3Δt, # Relaxation timescale for removing age at surface
    source_rate = 1.0,          # Source for the age (1 second / second)
)

@inline age_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate)

age_dynamics = Forcing(
    age_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)

forcing = (
    age = age_dynamics,
)

model = HydrostaticFreeSurfaceModel(
    grid;
    tracer_advection = Centered(order = 2),
    velocities = velocities,
    free_surface = free_surface,
    tracers = (; age = CenterField(grid)),
    closure = closure,
    forcing = forcing,
)

stop_time = 12 * prescribed_Δt

@info "Model built (stop_time = $(stop_time / year) years)"
flush(stdout)

@info "setup_model.jl complete"
flush(stdout)
