"""
Shared model setup for ACCESS-OM2 offline age simulations.

This file is `include()`d by:
  - `run_1year.jl`              — standalone 1-year simulation
  - `solve_periodic_NK.jl`      — Newton-GMRES periodic solver
  - `solve_periodic_AA.jl`      — Anderson/SpeedMapping periodic solver

It sets up: architecture, configuration, grid, velocities, closures, and model.
It does NOT create a Simulation or set initial conditions — each downstream
script does that according to its own needs.

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

@info "Loading packages and functions"
flush(stdout); flush(stderr)

using Oceananigans

include("select_architecture.jl")

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

include("shared_functions.jl")

(; parentmodel, outputdir, Δt_seconds) = load_project_config()
Δt = Δt_seconds * second

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

@info "Run configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- VELOCITY_SOURCE   = $VELOCITY_SOURCE"
@info "- W_FORMULATION     = $W_FORMULATION"
@info "- ADVECTION_SCHEME  = $ADVECTION_SCHEME"
@info "- TIMESTEPPER       = $TIMESTEPPER"
@info "- Architecture      = $arch_str"

@show outputdir
mkpath(outputdir)

################################################################################
# Load grid from JLD2
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))

@info "Reconstructing grid (loading data from JLD2)"
flush(stdout); flush(stderr)
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout); flush(stderr)

# Build a non-distributed CPU grid for loading global data from JLD2 files
if arch isa Distributed
    @info "Building CPU grid for distributed FTS loading"
    flush(stdout); flush(stderr)
    cpu_grid = load_tripolar_grid(grid_file, CPU())
end

################################################################################
# Load velocities from disk
################################################################################

@info "Loading velocities from disk"
flush(stdout); flush(stderr)

if VELOCITY_SOURCE == "cgridtransports"
    flush(stdout); flush(stderr)
    u_file = joinpath(preprocessed_inputs_dir, "u_from_mass_transport_periodic.jld2")
    v_file = joinpath(preprocessed_inputs_dir, "v_from_mass_transport_periodic.jld2")
    w_file = joinpath(preprocessed_inputs_dir, "w_from_mass_transport_periodic.jld2")
    @info """Loading velocities from MOM mass transport outputs files:
    - $(u_file)
    - $(v_file)
    - $(w_file)
    """
elseif VELOCITY_SOURCE == "bgridvelocities"
    flush(stdout); flush(stderr)
    u_file = joinpath(preprocessed_inputs_dir, "u_interpolated_periodic.jld2")
    v_file = joinpath(preprocessed_inputs_dir, "v_interpolated_periodic.jld2")
    w_file = joinpath(preprocessed_inputs_dir, "w_periodic.jld2")
    @info """Loading velocities from MOM velocity outputs files:
    - $(u_file)
    - $(v_file)
    - $(w_file)
    """
end

backend = InMemory()
time_indexing = Cyclical(1year)
fts_kw = arch isa Distributed ? (; cpu_grid) : (;)

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
u_ts = load_fts(arch, u_file, "u", grid; backend, time_indexing, fts_kw...)
@show u_ts
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
v_ts = load_fts(arch, v_file, "v", grid; backend, time_indexing, fts_kw...)
@show v_ts
@info "Loading sea surface height from MOM output"
flush(stdout); flush(stderr)
η_file = joinpath(preprocessed_inputs_dir, "eta_periodic.jld2")
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
η_ts = load_fts(arch, η_file, "η", grid; backend, time_indexing, fts_kw...)
@show η_ts

prescribed_Δt = u_ts.times[2] - u_ts.times[1]  # Infer from time spacing
fts_times = u_ts.times

@info "Velocities loaded (InMemory backend)"
flush(stdout); flush(stderr)

if W_FORMULATION == "wprescribed"
    @info "Using prescribed w field from: $(w_file)"
    isfile(w_file) || println("W_FORMULATION=wprescribed requires file: $(w_file)")
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    w_ts = load_fts(arch, w_file, "w", grid; backend, time_indexing, fts_kw...)
    @show w_ts

    @info "Prescribing u, v, and w"
    flush(stdout); flush(stderr)
    velocities = PrescribedVelocityFields(u = u_ts, v = v_ts, w = w_ts)
elseif W_FORMULATION == "wdiagnosed"
    @info "Prescribing u and v; diagnosing w via continuity"
    flush(stdout); flush(stderr)
    velocities = PrescribedVelocityFields(u = u_ts, v = v_ts, formulation = DiagnosticVerticalVelocity())
end

@info "Using prescribed sea surface height from MOM output"
flush(stdout); flush(stderr)
free_surface = PrescribedFreeSurface(displacement = η_ts)

################################################################################
# Closures
################################################################################

@info "Creating closures"
flush(stdout); flush(stderr)

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

# Vertical diffusivity parameters
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# Load MLD to add strong vertical diffusion in the mixed layer
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
mld_file = joinpath(inputdir, "mld.nc")
κVField = load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)

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
flush(stdout); flush(stderr)

################################################################################
# Model
################################################################################

@info "Building model"
flush(stdout); flush(stderr)

# Mutable 1-element array for source_rate, living on the compute architecture (GPU/CPU).
# Toggled between 1.0 (age forward map Φ!) and 0.0 (linear JVP) between simulation runs.
source_rate_arr = on_architecture(arch, [1.0])

age_parameters = (;
    relaxation_timescale = 3Δt, # Relaxation timescale for removing age at surface
    source_rate = source_rate_arr,
)

@inline age_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate[1])

age_dynamics = Forcing(
    age_source_sink,
    parameters = age_parameters,
    discrete_form = true,
)

forcing = (
    age = age_dynamics,
)

tracer_advection = advection_from_scheme(ADVECTION_SCHEME)
@info "Tracer advection scheme: $tracer_advection"

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
model = HydrostaticFreeSurfaceModel(
    grid;
    timestepper = timestepper_from_string(TIMESTEPPER),
    tracer_advection,
    velocities = velocities,
    free_surface = free_surface,
    tracers = (; age = CenterField(grid)),
    closure = closure,
    forcing = forcing,
)

stop_time = 12 * prescribed_Δt

@info "Model built (stop_time = $(stop_time / year) years)"
flush(stdout); flush(stderr)

@info "setup_model.jl complete"
flush(stdout); flush(stderr)
