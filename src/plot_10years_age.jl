"""
Plot age diagnostic figures from saved 10-year simulation output.

This is a standalone CPU script that loads the saved age field and generates
zonal-average and horizontal-slice plots. It is designed to be submitted as a
CPU PBS job after the GPU simulation completes.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/plot_10years_age.jl")
```

Alternatively, pass the JLD2 output filepath as ARGS[1].

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

@info "Loading packages for age plotting"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
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

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if length(ARGS) >= 2
    ARGS[2]
elseif haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
else
    outputdir = profile["outputdir"]
end

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

age_output_dir = joinpath(outputdir, "age", model_config)

if !isempty(ARGS)
    output_filepath = ARGS[1]
else
    output_filepath = joinpath(age_output_dir, "age_10years.jld2")
end

@info "Age plot configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- Output file      = $output_filepath"
flush(stdout); flush(stderr)

isfile(output_filepath) || error("Output file not found: $output_filepath")

################################################################################
# Load grid
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

################################################################################
# Load age field (last saved timestep)
################################################################################

@info "Loading age field from $output_filepath"
flush(stdout); flush(stderr)

age_fts = FieldTimeSeries(output_filepath, "age")
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
# Generate diagnostic plots
################################################################################

age_years_3D = age_data ./ year
label = "age_10years_$(ADVECTION_SCHEME)"

@info "Generating age diagnostic plots"
flush(stdout); flush(stderr)

plot_age_diagnostics(
    age_years_3D, grid, wet3D, vol_3D, age_output_dir, label;
    colorrange = (0, 10), levels = 0:1:10
)

@info "plot_10years_age.jl complete"
flush(stdout); flush(stderr)
