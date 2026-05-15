"""
Plot age diagnostic figures and animations from saved standard-run simulation output.

This is a standalone CPU script that loads the saved age field and generates:
  - Static zonal-average and horizontal-slice plots (PNGs)
  - Animated zonal-average age over time (MP4s)
  - Animated surface maps of input fields: u, v, w, η, MLD, T, S (MP4s, with halos)

It is designed to be submitted as a CPU PBS job after the GPU simulation completes.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
DURATION=1year julia --project src/plot_standardrun_age.jl
```

Alternatively, pass the JLD2 output filepath as ARGS[1].

Environment variables:
  DURATION         – 1year | 10years | 100years  (default: 1year)
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | totaltransport (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  GM_REDI          – yes | no  (default: no)
  MONTHLY_KAPPAV   – yes | no  (default: no)
"""

@info "Loading packages for age plotting"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: Cyclical, InMemory, Time
using Oceananigans.Units: day, days, second, seconds
year = years = 365.25days

using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using Statistics
using TOML
using JLD2
using Printf

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

DURATION = get(ENV, "DURATION", "1year")

duration_configs = Dict(
    "1year" => (; colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1),
    "10years" => (; colorrange = (0, 10), levels = 0:1:10),
    "100years" => (; colorrange = (0, 100), levels = 0:10:100),
)

haskey(duration_configs, DURATION) || error("Unknown DURATION=$DURATION; must be one of: $(join(keys(duration_configs), ", "))")
(; colorrange, levels) = duration_configs[DURATION]

(; parentmodel, experiment_dir, monthly_dir, mld_monthly_dir, outputdir) = load_project_config(; parentmodel_arg_index = 2)

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
GM_REDI = lowercase(get(ENV, "GM_REDI", "no")) == "yes"
MONTHLY_KAPPAV = lowercase(get(ENV, "MONTHLY_KAPPAV", "no")) == "yes"
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

age_output_dir = joinpath(outputdir, "standardrun", model_config)

if !isempty(ARGS)
    output_filepath = ARGS[1]
else
    output_filepath = joinpath(age_output_dir, "age_$(DURATION).jld2")
end

@info "Age plot configuration"
@info "- DURATION         = $DURATION"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- GM_REDI          = $GM_REDI"
@info "- MONTHLY_KAPPAV   = $MONTHLY_KAPPAV"
@info "- model_config     = $model_config"
@info "- Output file      = $output_filepath"
flush(stdout); flush(stderr)

isfile(output_filepath) || error("Output file not found: $output_filepath")

################################################################################
# Load grid
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

Nx, Ny, Nz = size(grid)

################################################################################
# Load age field (last saved timestep)
################################################################################

@info "Loading age field from $output_filepath"
flush(stdout); flush(stderr)

age_fts = FieldTimeSeries(output_filepath, "age"; grid, backend = InMemory())
n_times = length(age_fts.times)
@info "Found $n_times output timesteps; using last one"
flush(stdout); flush(stderr)

age_data = interior(age_fts[n_times])

################################################################################
# Compute masks and volume
################################################################################

@info "Computing wet mask and cell volumes"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)

vol = compute_volume(grid)
vol_3D = Array(interior(vol))

################################################################################
# Generate static diagnostic plots (PNGs)
################################################################################

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
age_base = TRAF ? "age_traf" : "age"
tracer_title = TRAF ? "TRAF age (time to re-emergence)" : "age"

age_years_3D = age_data ./ year
label = "$(age_base)_$(DURATION)_$(ADVECTION_SCHEME)"

@info "Generating static age diagnostic plots"
flush(stdout); flush(stderr)

plot_age_diagnostics(
    age_years_3D, grid, wet3D, vol_3D, age_output_dir, label;
    colorrange, levels,
)

################################################################################
# Animate age zonal averages over time (MP4s)
################################################################################

@info "Generating age zonal average animations"
flush(stdout); flush(stderr)

animate_zonal_averages(
    age_fts, grid, wet3D, vol_3D, age_output_dir, label;
    colorrange, levels, tracer_title,
)

################################################################################
# Animate surface maps of input fields (MP4s, with halos)
################################################################################

@info "Loading input FieldTimeSeries for surface animations"
flush(stdout); flush(stderr)

time_indexing = Cyclical(1year)

# Velocity fields
vs_prefix = VELOCITY_SOURCE == "totaltransport" ? "total_transport" : "mass_transport"
u_file = joinpath(monthly_dir, "u_from_$(vs_prefix)_monthly.jld2")
v_file = joinpath(monthly_dir, "v_from_$(vs_prefix)_monthly.jld2")
w_file = joinpath(monthly_dir, "w_from_$(vs_prefix)_monthly.jld2")
η_file = joinpath(monthly_dir, "eta_monthly.jld2")

u_ts = FieldTimeSeries(u_file, "u"; grid, backend = InMemory(), time_indexing)
v_ts = FieldTimeSeries(v_file, "v"; grid, backend = InMemory(), time_indexing)
w_ts = FieldTimeSeries(w_file, "w"; grid, backend = InMemory(), time_indexing)
η_ts = FieldTimeSeries(η_file, "η"; grid, backend = InMemory(), time_indexing)

# Surface k-index: top level for 3D fields (with halos, offset by halo size)
Hz = grid.Hx  # halo size (same in all directions for this grid)
k_surface_ccc = Nz + Hz  # top Center level in parent array (with halos)
k_surface_ccf = Nz + Hz + 1  # top Face level in parent array (w is at faces)

field_specs = Tuple{String, Any, Union{Nothing, Int}}[
    ("u", u_ts, k_surface_ccc),
    ("v", v_ts, k_surface_ccc),
    ("w", w_ts, k_surface_ccf),
    ("eta", η_ts, nothing),
]

# Optionally add T, S surface animations — opt-in via PLOT_TS=yes.
# T/S monthly FTS files may have been preprocessed with smaller halos than the
# current grid; loading them with a mismatched halo throws DimensionMismatch.
# Most runs don't need T/S surface plots, so we gate the load behind an
# explicit flag and skip-with-hint otherwise.
T_file = joinpath(monthly_dir, "temp_monthly.jld2")
S_file = joinpath(monthly_dir, "salt_monthly.jld2")
PLOT_TS = lowercase(get(ENV, "PLOT_TS", "no")) == "yes"
if isfile(T_file) && isfile(S_file)
    if PLOT_TS
        @info "Loading T and S FTS for surface animations (PLOT_TS=yes)"
        T_ts = FieldTimeSeries(T_file, "T"; grid, backend = InMemory(), time_indexing)
        S_ts = FieldTimeSeries(S_file, "S"; grid, backend = InMemory(), time_indexing)
        push!(field_specs, ("T", T_ts, k_surface_ccc))
        push!(field_specs, ("S", S_ts, k_surface_ccc))
    else
        @info "Skipping T/S surface animations — set PLOT_TS=yes to enable (T/S monthly FTS present at $T_file)"
    end
end

# Optional diagnostic FTS loaders. Wrap in try/catch so a stale-halo file
# (e.g. preprocessed before GRID_H{X,Y,Z} changed) skips that one diagnostic
# with a hint instead of aborting the whole plot. MLD has a (1454,1094)
# halo-mismatch against the current grid's (1466,1106) on OM2-025; the κV
# file also has Hz=2 baked in vs the current GRID_HZ=7 on OM2-1.

function _try_load_diag_fts(file, varname; grid, backend, time_indexing)
    try
        return FieldTimeSeries(file, varname; grid, backend, time_indexing)
    catch err
        @warn "Skipping $varname plot — could not load $file (likely halo mismatch; re-preprocess at current GRID_H{X,Y,Z})" exception = (err, catch_backtrace())
        return nothing
    end
end

# Optionally add MLD if monthly FTS exists
mld_file = joinpath(mld_monthly_dir, "mld_monthly.jld2")
if isfile(mld_file)
    @info "Loading MLD FTS from: $mld_file"
    mld_ts = _try_load_diag_fts(mld_file, "MLD"; grid, backend = InMemory(), time_indexing)
    if mld_ts !== nothing
        push!(field_specs, ("MLD", mld_ts, nothing))
        push!(field_specs, ("MLK", mld_ts, nothing))
    end
end

# Optionally add κV at ~200m depth — only when MONTHLY_KAPPAV=yes, mirroring
# setup_model.jl's gate.
κV_file = joinpath(mld_monthly_dir, "kappa_v_monthly.jld2")
MONTHLY_KAPPAV = lowercase(get(ENV, "MONTHLY_KAPPAV", "no")) == "yes"
if MONTHLY_KAPPAV && isfile(κV_file)
    @info "Loading κV FTS from: $κV_file"
    κV_ts = _try_load_diag_fts(κV_file, "κV"; grid, backend = InMemory(), time_indexing)
    if κV_ts !== nothing
        k_200m = find_nearest_depth_index(grid, 200)
        k_200m_halos = k_200m + Hz  # offset for halo in parent array
        push!(field_specs, ("kappaV_200m", κV_ts, k_200m_halos))
    end
elseif isfile(κV_file)
    @info "Skipping κV plot — set MONTHLY_KAPPAV=yes to enable (κV file present at $κV_file)"
end

@info "Generating surface field animations ($(length(field_specs)) fields)"
flush(stdout); flush(stderr)

animate_surface_fields(field_specs, age_output_dir, label; show_halos = true)

@info "plot_standardrun_age.jl complete (DURATION=$DURATION)"
flush(stdout); flush(stderr)
