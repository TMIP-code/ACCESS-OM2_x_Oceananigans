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
  GM_REDI          – yes | no  (default: no)  — enable GM-Redi isopycnal diffusion with prescribed T/S
  MONTHLY_KAPPAV   – yes | no  (default: no)  — use monthly time-varying vertical diffusivity from MLD
"""

@info "Loading packages and functions"
flush(stdout); flush(stderr)

using Oceananigans

include("select_architecture.jl")

using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode
using Oceananigans.OutputReaders: Cyclical, InMemory, Time
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

(; parentmodel, experiment, time_window, experiment_dir, monthly_dir, yearly_dir, outputdir, Δt_seconds) = load_project_config()
Δt = Δt_seconds * second

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
GM_REDI = lowercase(get(ENV, "GM_REDI", "no")) == "yes"
MONTHLY_KAPPAV = lowercase(get(ENV, "MONTHLY_KAPPAV", "no")) == "yes"
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"
if GM_REDI
    model_config = "$(model_config)_GMREDI"
end
if MONTHLY_KAPPAV
    model_config = "$(model_config)_mkappaV"
end

@info "Run configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- VELOCITY_SOURCE   = $VELOCITY_SOURCE"
@info "- W_FORMULATION     = $W_FORMULATION"
@info "- ADVECTION_SCHEME  = $ADVECTION_SCHEME"
@info "- TIMESTEPPER       = $TIMESTEPPER"
@info "- GM_REDI           = $GM_REDI"
@info "- MONTHLY_KAPPAV    = $MONTHLY_KAPPAV"
@info "- Architecture      = $arch_str"

@show outputdir
mkpath(outputdir)

################################################################################
# Load grid from JLD2
################################################################################

@info "Reconstructing grid (loading data from JLD2)"
flush(stdout); flush(stderr)
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
grid_file = joinpath(experiment_dir, "grid.jld2")
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
    u_file = joinpath(monthly_dir, "u_from_mass_transport_monthly.jld2")
    v_file = joinpath(monthly_dir, "v_from_mass_transport_monthly.jld2")
    w_file = joinpath(monthly_dir, "w_from_mass_transport_monthly.jld2")
    @info """Loading velocities from MOM mass transport outputs files:
    - $(u_file)
    - $(v_file)
    - $(w_file)
    """
elseif VELOCITY_SOURCE == "bgridvelocities"
    flush(stdout); flush(stderr)
    u_file = joinpath(monthly_dir, "u_interpolated_monthly.jld2")
    v_file = joinpath(monthly_dir, "v_interpolated_monthly.jld2")
    w_file = joinpath(monthly_dir, "w_monthly.jld2")
    @info """Loading velocities from MOM velocity outputs files:
    - $(u_file)
    - $(v_file)
    - $(w_file)
    """
end

backend = InMemory()
time_indexing = Cyclical(1year)

# Check for pre-partitioned FTS files (written by partition_data.jl)
partition_dir = nothing
if arch isa Distributed
    px = parse(Int, get(ENV, "PARTITION_X", "1"))
    py = parse(Int, get(ENV, "PARTITION_Y", "1"))
    pd = joinpath(experiment_dir, time_window, "partitions", "$(px)x$(py)")
    if isdir(pd) && !isempty(readdir(pd))
        partition_dir = pd
        @info "Using pre-partitioned FTS data from $partition_dir"
    else
        @info "No pre-partitioned data at $pd — using runtime fold_set! partitioning"
    end
end
fts_kw = arch isa Distributed ? (; cpu_grid, partition_dir) : (;)

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
u_ts = load_fts(arch, u_file, "u", grid; backend, time_indexing, fts_kw...)
@show u_ts
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
v_ts = load_fts(arch, v_file, "v", grid; backend, time_indexing, fts_kw...)
@show v_ts
@info "Loading sea surface height from MOM output"
flush(stdout); flush(stderr)
η_file = joinpath(monthly_dir, "eta_monthly.jld2")
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
η_ts = load_fts(arch, η_file, "η", grid; backend, time_indexing, fts_kw...)
@show η_ts

prescribed_Δt = u_ts.times[2] - u_ts.times[1]  # Infer from time spacing
fts_times = u_ts.times

@info "Velocities loaded (InMemory backend)"
flush(stdout); flush(stderr)

# Load T and S FieldTimeSeries for GM-Redi buoyancy
if GM_REDI
    @info "Loading T and S FieldTimeSeries for GM-Redi buoyancy"
    flush(stdout); flush(stderr)
    T_file = joinpath(monthly_dir, "temp_monthly.jld2")
    S_file = joinpath(monthly_dir, "salt_monthly.jld2")
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    T_ts = load_fts(arch, T_file, "T", grid; backend, time_indexing, fts_kw...)
    @show T_ts
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    S_ts = load_fts(arch, S_file, "S", grid; backend, time_indexing, fts_kw...)
    @show S_ts
    @info "T and S FieldTimeSeries loaded"
    flush(stdout); flush(stderr)
end

if W_FORMULATION == "wprescribed"
    # Select w source: "diagnosed" (from Oceananigans continuity) or "parent" (from MOM output)
    PRESCRIBED_W_SOURCE = get(ENV, "PRESCRIBED_W_SOURCE", "parent")
    if PRESCRIBED_W_SOURCE == "diagnosed"
        w_file = joinpath(monthly_dir, "w_diagnosed_monthly.jld2")
    end
    # else: w_file already set from VELOCITY_SOURCE (mass_transport or interpolated)
    @info "Using prescribed w (source=$PRESCRIBED_W_SOURCE) from: $(w_file)"
    isfile(w_file) || error("W_FORMULATION=wprescribed requires file: $(w_file)")
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

# Vertical diffusivity parameters
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# Load MLD-based vertical diffusivity (static yearly average)
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
mld_file = joinpath(yearly_dir, "mld_yearly.nc")
κVField = load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)

# Optionally load monthly κV FTS for time-varying vertical diffusivity
if MONTHLY_KAPPAV
    @info "Loading monthly κV FTS (time-varying vertical diffusivity)"
    flush(stdout); flush(stderr)
    κV_file = joinpath(monthly_dir, "kappa_v_monthly.jld2")
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    κV_ts = load_fts(arch, κV_file, "κV", grid; backend, time_indexing, fts_kw...)
    @show κV_ts
    # Initialize κVField from first month
    set!(κVField, κV_ts[1])
    @info "κVField initialized from first month of κV FTS"
    flush(stdout); flush(stderr)
end

if GM_REDI
    # Per-tracer diffusivities: zero for ghost tracers T/S, real values for age
    horizontal_diffusion = HorizontalScalarDiffusivity(κ = (; T = 0.0, S = 0.0, age = 300.0))
    implicit_vertical_diffusion = VerticalScalarDiffusivity(
        VerticallyImplicitTimeDiscretization();
        κ = (; T = 0.0, S = 0.0, age = κVField),
    )
    gm_redi = IsopycnalSkewSymmetricDiffusivity(
        κ_skew = (; T = 0.0, S = 0.0, age = 300.0),
        κ_symmetric = (; T = 0.0, S = 0.0, age = 300.0),
    )
    closure = (horizontal_diffusion, implicit_vertical_diffusion, gm_redi)
    @info "Closures: horizontal + vertical + GM-Redi (IsopycnalSkewSymmetricDiffusivity)"
else
    implicit_vertical_diffusion = VerticalScalarDiffusivity(
        VerticallyImplicitTimeDiscretization();
        κ = κVField,
    )
    horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
    closure = (horizontal_diffusion, implicit_vertical_diffusion)
end

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

if GM_REDI
    age_advection = advection_from_scheme(ADVECTION_SCHEME)
    tracer_advection = (; T = nothing, S = nothing, age = age_advection)
    @info "Tracer advection: T=nothing, S=nothing, age=$age_advection"
    tracers = (; T = CenterField(grid), S = CenterField(grid), age = CenterField(grid))
    buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState())
    @info "Buoyancy: SeawaterBuoyancy with LinearEquationOfState"
else
    tracer_advection = advection_from_scheme(ADVECTION_SCHEME)
    @info "Tracer advection scheme: $tracer_advection"
    tracers = (; age = CenterField(grid))
end

# Build optional kwargs — only pass buoyancy when GM_REDI is enabled
# (buoyancy defaults to nothing in Oceananigans — preserve legacy code path)
buoyancy_kw = GM_REDI ? (; buoyancy) : (;)

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
model = HydrostaticFreeSurfaceModel(
    grid;
    timestepper = timestepper_from_string(TIMESTEPPER),
    tracer_advection,
    velocities = velocities,
    free_surface = free_surface,
    tracers = tracers,
    closure = closure,
    forcing = forcing,
    buoyancy_kw...,
)

# Prescribe T and S from FTS at each iteration (only called if GM_REDI is enabled)
function prescribe_TS!(sim)
    t = Time(time(sim))
    set!(sim.model.tracers.T, T_ts[t])
    return set!(sim.model.tracers.S, S_ts[t])
end

# Update κV from monthly FTS (only called if MONTHLY_KAPPAV is enabled)
function update_κV!(sim)
    t = Time(time(sim))
    return set!(κVField, κV_ts[t])
end

stop_time = 12 * prescribed_Δt

@info "Model built (stop_time = $(stop_time / year) years)"
flush(stdout); flush(stderr)

@info "setup_model.jl complete"
flush(stdout); flush(stderr)
