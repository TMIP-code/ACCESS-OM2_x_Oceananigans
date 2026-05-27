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
  VELOCITY_SOURCE  – cgridtransports | totaltransport (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  GM_REDI          – no | diff | adv  (default: no)  — enable GM-Redi isopycnal diffusion with prescribed T/S
  MONTHLY_KAPPAV   – yes | no  (default: yes) — use monthly time-varying κV derived from the 2D
                                                monthly MLD FTS each iteration via update_κV_from_mld!
  IMPLICIT_KAPPAV  – yes | no  (default: yes) — when "no", drop the implicit vertical-diffusion closure
                                                (Probe B for the GPU seam tracer bug; outputs tagged _noKV)
"""

@info "Loading packages and functions"
flush(stdout); flush(stderr)

using Oceananigans

include("select_architecture.jl")

using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy, LinearEquationOfState
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: AdvectiveFormulation, DiffusiveFormulation
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

(; parentmodel, experiment, time_window, mld_time_window, experiment_dir, monthly_dir, yearly_dir, mld_monthly_dir, mld_yearly_dir, outputdir, Δt_seconds) = load_project_config()
Δt = Δt_seconds * second

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
GM_REDI_STR = lowercase(require_env("GM_REDI"))
GM_REDI_STR == "yes" && (GM_REDI_STR = "diff")  # backward compat
GM_REDI = GM_REDI_STR in ("diff", "adv")
GM_ADVECTIVE = GM_REDI_STR == "adv"
MONTHLY_KAPPAV = lowercase(require_env("MONTHLY_KAPPAV")) == "yes"
IMPLICIT_KAPPAV_STR = lowercase(require_env("IMPLICIT_KAPPAV"))
IMPLICIT_KAPPAV_STR ∈ ("yes", "no") || error("IMPLICIT_KAPPAV must be yes or no (got: $IMPLICIT_KAPPAV_STR)")
IMPLICIT_KAPPAV = IMPLICIT_KAPPAV_STR == "yes"
TRAF = lowercase(require_env("TRAF")) == "yes"
if TRAF
    @info "TRAF=yes — reversing every monthly FTS in time; flipping sign of u, v FTS"
    W_FORMULATION == "wprescribed" && error(
        "TRAF=yes is only supported with W_FORMULATION=wdiagnosed in the first cut " *
            "(w is recomputed from continuity using sign-flipped/time-reversed u, v, η). " *
            "Support for TRAF + wprescribed is a follow-up.",
    )
end
model_config = require_env("MODEL_CONFIG")

@info "Run configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- VELOCITY_SOURCE   = $VELOCITY_SOURCE"
@info "- W_FORMULATION     = $W_FORMULATION"
@info "- ADVECTION_SCHEME  = $ADVECTION_SCHEME"
@info "- TIMESTEPPER       = $TIMESTEPPER"
@info "- GM_REDI           = $GM_REDI"
@info "- MONTHLY_KAPPAV    = $MONTHLY_KAPPAV"
@info "- IMPLICIT_KAPPAV   = $IMPLICIT_KAPPAV"
@info "- TRAF              = $TRAF"
@info "- OMEGA             = $(require_env("OMEGA"))"
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
@show grid
ug = grid.underlying_grid

# Guard: the halo dimensions baked into grid.jld2 are authoritative at runtime
# (env GRID_HX/HY/HZ only affect create_grid.jl). Error out if a stale grid.jld2
# would silently override the submit-time GRID_H{X,Y,Z}.
let
    env_Hx = parse(Int, get(ENV, "GRID_HX", string(ug.Hx)))
    env_Hy = parse(Int, get(ENV, "GRID_HY", string(ug.Hy)))
    env_Hz = parse(Int, get(ENV, "GRID_HZ", string(ug.Hz)))
    if (env_Hx, env_Hy, env_Hz) ≠ (ug.Hx, ug.Hy, ug.Hz)
        error(
            "Loaded grid.jld2 halo ($(ug.Hx), $(ug.Hy), $(ug.Hz)) does not match " *
                "GRID_HX/HY/HZ=($env_Hx, $env_Hy, $env_Hz). Rebuild the grid " *
                "(JOB_CHAIN=grid) with the desired halos, or unset GRID_HX/HY/HZ to " *
                "accept the saved values. File: $grid_file"
        )
    end
end
@show typeof(ug.λᶜᶜᵃ) typeof(ug.φᶜᶜᵃ)
@show typeof(ug.Δxᶜᶜᵃ) typeof(ug.Δyᶜᶜᵃ) typeof(ug.Azᶜᶜᵃ)
@show typeof(ug.z)
@show typeof(grid.immersed_boundary.bottom_height)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout); flush(stderr)

# Helper: log per-rank GPU memory at a labelled checkpoint
function gpu_mem_log(tag::AbstractString)
    try
        rank = arch isa Distributed ? arch.local_rank : 0
        @info "[rank $rank] GPU mem $tag:"
        CUDA.pool_status()
    catch e
        @warn "GPU mem probe failed at $tag: $e"
    end
    return flush(stdout)
end
gpu_mem_log("after grid load")

################################################################################
# Load velocities from disk
################################################################################

@info "Loading velocities from disk"
flush(stdout); flush(stderr)

flush(stdout); flush(stderr)
vs_prefix = VELOCITY_SOURCE == "totaltransport" ? "total_transport" : "mass_transport"
u_file = joinpath(monthly_dir, "u_from_$(vs_prefix)_monthly.jld2")
v_file = joinpath(monthly_dir, "v_from_$(vs_prefix)_monthly.jld2")
w_file = joinpath(monthly_dir, "w_from_$(vs_prefix)_monthly.jld2")
@info """Loading velocities from MOM $(vs_prefix) outputs files:
- $(u_file)
- $(v_file)
- $(w_file)
"""

backend = InMemory()
time_indexing = Cyclical(1year)

# Distributed runs require pre-partitioned FTS files (written by partition_data.jl)
fts_kw = (;)
if arch isa Distributed
    px = parse(Int, require_env("PARTITION_X"))
    py = parse(Int, require_env("PARTITION_Y"))
    ptag = "$(px)x$(py)$(LB_TAG)"
    partition_dir = joinpath(experiment_dir, time_window, "partitions", ptag)
    (isdir(partition_dir) && !isempty(readdir(partition_dir))) || error(
        "Pre-partitioned FTS directory missing or empty: $partition_dir. " *
            "Run the `partition` step (e.g., JOB_CHAIN=partition-run1yrfast) first."
    )
    @info "Using pre-partitioned FTS data from $partition_dir"
    fts_kw = (; partition_dir)
end

arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
u_ts = load_fts(arch, u_file, "u", grid; backend, time_indexing, fts_kw...)
TRAF && reverse_fts_time!(u_ts; flip_sign = true)
@show u_ts
gpu_mem_log("after u FTS load")
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
v_ts = load_fts(arch, v_file, "v", grid; backend, time_indexing, fts_kw...)
TRAF && reverse_fts_time!(v_ts; flip_sign = true)
@show v_ts
gpu_mem_log("after v FTS load")
@info "Loading sea surface height from MOM output"
flush(stdout); flush(stderr)
η_file = joinpath(monthly_dir, "eta_monthly.jld2")
arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
η_ts = load_fts(arch, η_file, "η", grid; backend, time_indexing, fts_kw...)
TRAF && reverse_fts_time!(η_ts; flip_sign = false)
@show η_ts
gpu_mem_log("after η FTS load")

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
    TRAF && reverse_fts_time!(T_ts; flip_sign = false)
    @show T_ts
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    S_ts = load_fts(arch, S_file, "S", grid; backend, time_indexing, fts_kw...)
    TRAF && reverse_fts_time!(S_ts; flip_sign = false)
    @show S_ts
    @info "T and S FieldTimeSeries loaded"
    flush(stdout); flush(stderr)
end

if W_FORMULATION == "wprescribed"
    # Select w source: "diagnosed" (from Oceananigans continuity) or "parent" (from MOM output)
    PRESCRIBED_W_SOURCE = require_env("PRESCRIBED_W_SOURCE")
    if PRESCRIBED_W_SOURCE == "diagnosed"
        w_diag_suffix = VELOCITY_SOURCE == "totaltransport" ? "w_diagnosed_totaltransport_monthly" : "w_diagnosed_monthly"
        w_file = joinpath(monthly_dir, "$(w_diag_suffix).jld2")
    end
    # else: w_file already set from VELOCITY_SOURCE (mass_transport or interpolated)
    @info "Using prescribed w (source=$PRESCRIBED_W_SOURCE) from: $(w_file)"
    isfile(w_file) || error("W_FORMULATION=wprescribed requires file: $(w_file)")
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    w_ts = load_fts(arch, w_file, "w", grid; backend, time_indexing, fts_kw...)
    @show w_ts
    gpu_mem_log("after w FTS load")

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
mld_file = joinpath(mld_yearly_dir, "mld_yearly.nc")
@info "Loading MLD (yearly) from: $mld_file"
flush(stdout); flush(stderr)
κVField = load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)
@show κVField
gpu_mem_log("after yearly MLD-based κVField")

# Optionally load monthly 2D MLD FTS for time-varying vertical diffusivity.
# κV is derived on the fly each iteration via update_κV_from_mld! — ~Nz×
# smaller in memory than loading a precomputed 3D κV FTS.
if MONTHLY_KAPPAV
    arch isa Distributed && MPI.Barrier(MPI.COMM_WORLD)
    mld_monthly_file = joinpath(mld_monthly_dir, "mld_monthly.jld2")
    @info "Loading monthly MLD FTS from: $mld_monthly_file"
    flush(stdout); flush(stderr)
    mld_ts = load_fts(arch, mld_monthly_file, "MLD", grid; backend, time_indexing, fts_kw...)
    # MLD on disk is positive-downward (see prep_closures.jl). Negate every
    # snapshot in place so the in-memory FTS values are in z-coordinate sign
    # convention (negative in the ocean), matching update_κV_from_mld! and
    # the yearly load_mld_diffusivity path.
    for n in 1:length(mld_ts.times)
        parent(mld_ts[n]) .*= -1
    end
    TRAF && reverse_fts_time!(mld_ts; flip_sign = false)
    @show mld_ts
    mld_scratch = Field{Center, Center, Nothing}(grid)
    z_center_3d = make_z_center_3d(arch, grid)
    set!(mld_scratch, mld_ts[1])
    update_κV_from_mld!(κVField, mld_scratch, z_center_3d, κVML, κVBG)
    @info "κVField initialized from first month of MLD FTS (negated to z-coord)"
    flush(stdout); flush(stderr)
    gpu_mem_log("after monthly MLD FTS load + κV init")
end

gpu_mem_log("after all preprocessed inputs loaded")

if GM_REDI
    # No HorizontalScalarDiffusivity — isopycnal κ_symmetric in GM-Redi handles horizontal mixing
    implicit_vertical_diffusion = VerticalScalarDiffusivity(
        VerticallyImplicitTimeDiscretization();
        κ = (; T = 0.0, S = 0.0, age = κVField),
    )
    gm_formulation = GM_ADVECTIVE ? AdvectiveFormulation() : DiffusiveFormulation()
    # AdvectiveFormulation requires scalar κ_skew (Oceananigans limitation);
    # this is fine since T/S have tracer_advection=nothing and are prescribed each step.
    gm_κ_skew = GM_ADVECTIVE ? 300.0 : (; T = 0.0, S = 0.0, age = 300.0)
    gm_κ_symmetric = GM_ADVECTIVE ? 300.0 : (; T = 0.0, S = 0.0, age = 300.0)
    gm_redi = IsopycnalSkewSymmetricDiffusivity(
        skew_flux_formulation = gm_formulation,
        κ_skew = gm_κ_skew,
        κ_symmetric = gm_κ_symmetric,
    )
    closure = IMPLICIT_KAPPAV ? (implicit_vertical_diffusion, gm_redi) : (gm_redi,)
    @info "Closures: $(IMPLICIT_KAPPAV ? "vertical + " : "")GM-Redi ($gm_formulation) — no horizontal scalar diffusion"
else
    implicit_vertical_diffusion = VerticalScalarDiffusivity(
        VerticallyImplicitTimeDiscretization();
        κ = κVField,
    )
    horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
    closure = IMPLICIT_KAPPAV ? (horizontal_diffusion, implicit_vertical_diffusion) : (horizontal_diffusion,)
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

# OMEGA mask: restrict where the age source is applied (sink at k=Nz is untouched).
omega = parse_omega()
k_mask = build_omega_k_mask(grid, omega; arch)
@info "OMEGA = $(omega.tag) (suffix='$(omega.suffix)') — source wet k-levels: $(count(>(0), Array(k_mask)))/$(size(grid, 3))"

age_parameters = (;
    relaxation_timescale = 3Δt, # Relaxation timescale for removing age at surface
    source_rate = source_rate_arr,
    k_mask = k_mask,
)

@inline age_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.age[i, j, k] / params.relaxation_timescale, params.source_rate[1] * params.k_mask[k])

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
    @show tracers.T tracers.S tracers.age
    buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState())
    @info "Buoyancy: SeawaterBuoyancy with LinearEquationOfState"
else
    tracer_advection = advection_from_scheme(ADVECTION_SCHEME)
    @info "Tracer advection scheme: $tracer_advection"
    tracers = (; age = CenterField(grid))
    @show tracers.age
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

# Prescribe T and S from FTS at each iteration (only called if GM_REDI is enabled).
# Uses interp_fts! (data_loading.jl) to avoid per-step Field allocation that
# `set!(field, fts[Time(t)])` triggers under the hood.
function prescribe_TS!(sim)
    t = time(sim)
    interp_fts!(sim.model.tracers.T, T_ts, t)
    interp_fts!(sim.model.tracers.S, S_ts, t)
    return nothing
end

# Update κV each iteration (only called if MONTHLY_KAPPAV is enabled).
if MONTHLY_KAPPAV
    function update_κV!(sim)
        interp_fts!(mld_scratch, mld_ts, time(sim))
        update_κV_from_mld!(κVField, mld_scratch, z_center_3d, κVML, κVBG)
        return nothing
    end
end

n_months = parse(Int, get(ENV, "N_MONTHS", "12"))
stop_time = n_months * prescribed_Δt

@info "Model built (stop_time = $(stop_time / year) years, N_MONTHS=$n_months)"
flush(stdout); flush(stderr)
gpu_mem_log("after model build")

@info "setup_model.jl complete"
flush(stdout); flush(stderr)
