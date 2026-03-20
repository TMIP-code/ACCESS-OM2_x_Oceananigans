"""
Plot age diagnostic figures from the 1-year periodic simulation output.

Generates:
  - Periodicity check (vol-weighted RMS of snapshot 25 − snapshot 1)
  - 10 static PNGs from the time-averaged age (4 zonal averages + 6 depth slices)
  - 10 smooth MP4 animations (4 zonal averages + 6 depth slices, 144 frames @ 24 fps)

This is a standalone CPU script designed to be submitted as a CPU PBS job
after the GPU simulation completes.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/plot_periodic_1year_age.jl")
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  LINEAR_SOLVER    – Pardiso | ParU | UMFPACK  (default: Pardiso)
  LUMP_AND_SPRAY   – yes | no  (default: no)
"""

@info "Loading packages for periodic age plotting"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Units: day, days, second, seconds, Time
year = years = 365.25days

using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using LinearAlgebra: dot, Diagonal
using Statistics
using TOML
using JLD2
using Printf

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, outputdir) = load_project_config()

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
lumpspray_tag = LUMP_AND_SPRAY ? "LSprec" : "prec"
solver_tag = "$(LINEAR_SOLVER)_$(lumpspray_tag)"

periodic_1year_dir = joinpath(outputdir, "periodic", model_config, "1year", solver_tag)
output_filepath = joinpath(periodic_1year_dir, "age_periodic_1year.jld2")

@info "Periodic age plot configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- LINEAR_SOLVER    = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY   = $LUMP_AND_SPRAY (tag: $solver_tag)"
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

################################################################################
# Load age FieldTimeSeries
################################################################################

@info "Loading age FieldTimeSeries from $output_filepath"
flush(stdout); flush(stderr)

age_fts = FieldTimeSeries(output_filepath, "age")
n_times = length(age_fts.times)
@info "Found $n_times output timesteps"
flush(stdout); flush(stderr)

################################################################################
# Compute masks, volume, and grid-derived quantities (once)
################################################################################

@info "Computing wet mask, cell volumes, and grid coordinates"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)

vol_3D = Array(interior(compute_volume(grid)))

age_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)    # age in years, NaN-masked

################################################################################
# Plot settings
################################################################################

plot_dir = joinpath(periodic_1year_dir, "plots")
mkpath(plot_dir)

colorrange = (0, 1600)
levels = 0:100:1600
colormap = cgrad(:viridis, length(levels) - 1, categorical = true)

@info "Colorrange for all plots: $colorrange years"
flush(stdout); flush(stderr)

################################################################################
# Periodicity check: snapshot 1 ≈ snapshot n_times
################################################################################

@info "Running periodicity check (snapshot 1 vs snapshot $n_times)"
flush(stdout); flush(stderr)

begin
    v1D = vol_3D[idx]
    inv_sumv = 1 / sum(v1D)
    age_first_1D = interior(age_fts[1])[idx]
    age_last_1D = interior(age_fts[n_times])[idx]
    diff_1D = age_last_1D .- age_first_1D
    vol_rms_drift = sqrt(dot(diff_1D, Diagonal(v1D), diff_1D) * inv_sumv) / year
    max_drift = maximum(abs, diff_1D) / year
    mean_drift = mean(abs, diff_1D) / year
    @info "Periodicity check results" vol_rms_drift_years = vol_rms_drift max_drift_years = max_drift mean_drift_years = mean_drift
    flush(stdout); flush(stderr)
end

################################################################################
# Averaged diagnostics (10 static PNGs)
################################################################################

@info "Computing time-averaged age over first $(n_times - 1) snapshots"
flush(stdout); flush(stderr)

n_avg = n_times - 1
fill!(age_buf, 0.0)
for n in 1:n_avg
    age_raw = interior(age_fts[n])
    @. age_buf += ifelse(wet3D, age_raw / year, 0.0)
end
@. age_buf = ifelse(wet3D, age_buf / n_avg, NaN)

label = "age_periodic_mean_$(ADVECTION_SCHEME)"

@info "Generating averaged diagnostic plots"
flush(stdout); flush(stderr)

plot_age_diagnostics(age_buf, grid, wet3D, vol_3D, plot_dir, label; colorrange, levels)

################################################################################
# Smooth animations (10 MP4 files)
################################################################################

@info "Generating smooth animations (144 frames @ 24 fps, 6s per video)"
flush(stdout); flush(stderr)

anim_prefix = "age_periodic_$(ADVECTION_SCHEME)"
animate_zonal_averages(age_fts, grid, wet3D, vol_3D, plot_dir, anim_prefix; colorrange, levels, colormap)
animate_depth_slices(age_fts, grid, wet3D, plot_dir, anim_prefix; colorrange, levels, colormap)

@info "plot_periodic_1year_age.jl complete — 10 averaged PNGs + 10 animations saved to $plot_dir"
flush(stdout); flush(stderr)
