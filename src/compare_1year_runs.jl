"""
Compare 1-year age output between serial (1 GPU) and distributed (multi-GPU) runs.

Loads the final t=1year snapshot from both architectures and generates diagnostic
plots of the difference. This verifies that distributed runs produce the same
results as serial runs.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
GPU_TAG=2x2 julia --project src/compare_1year_runs.jl
```

Environment variables:
  GPU_TAG          – partition tag to compare against serial (e.g., "2x2")
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION    – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
"""

@info "Loading packages for architecture comparison"
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

GPU_TAG = get(ENV, "GPU_TAG", "")
isempty(GPU_TAG) && error("GPU_TAG must be set (e.g., GPU_TAG=2x2)")

(; parentmodel, outputdir) = load_project_config(; parentmodel_arg_index = 2)

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

serial_dir = joinpath(outputdir, "standardrun", model_config)
distributed_dir = joinpath(outputdir, "standardrun", model_config, GPU_TAG)

serial_file = joinpath(serial_dir, "age_1year.jld2")
distributed_file = joinpath(distributed_dir, "age_1year.jld2")

@info "Architecture comparison configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- GPU_TAG          = $GPU_TAG"
@info "- model_config     = $model_config"
@info "- Serial output    = $serial_file"
@info "- Distributed out  = $distributed_file"
flush(stdout); flush(stderr)

isfile(serial_file) || error("Serial output not found: $serial_file")
isfile(distributed_file) || error("Distributed output not found: $distributed_file")

################################################################################
# Load grid
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

################################################################################
# Load age fields (last saved timestep from each run)
################################################################################

@info "Loading serial age field from $serial_file"
flush(stdout); flush(stderr)
serial_fts = FieldTimeSeries(serial_file, "age")
n_serial = length(serial_fts.times)
@info "Serial run: $n_serial timesteps, using last one (t = $(serial_fts.times[end] / year) years)"
age_serial = interior(serial_fts[n_serial])

@info "Loading distributed age field from $distributed_file"
flush(stdout); flush(stderr)
dist_fts = FieldTimeSeries(distributed_file, "age")
n_dist = length(dist_fts.times)
@info "Distributed run: $n_dist timesteps, using last one (t = $(dist_fts.times[end] / year) years)"
age_dist = interior(dist_fts[n_dist])

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
# Compute differences
################################################################################

@info "Computing differences"
flush(stdout); flush(stderr)

age_serial_yr = age_serial ./ year
age_dist_yr = age_dist ./ year
age_diff_yr = age_dist_yr .- age_serial_yr

# Relative difference (avoid division by zero in dry cells / near-zero regions)
age_reldiff = similar(age_diff_yr)
for i in eachindex(age_diff_yr)
    if wet3D[i] && abs(age_serial_yr[i]) > 1.0e-10
        age_reldiff[i] = age_diff_yr[i] / age_serial_yr[i]
    else
        age_reldiff[i] = NaN
    end
end

# Summary statistics (wet cells only)
wet_diff = age_diff_yr[wet3D]
wet_reldiff = filter(!isnan, age_reldiff[wet3D])

@info "Difference statistics (wet cells only):"
@info @sprintf("  max|diff|    = %.2e years", maximum(abs, wet_diff))
@info @sprintf("  mean|diff|   = %.2e years", mean(abs, wet_diff))
@info @sprintf("  RMS diff     = %.2e years", sqrt(mean(wet_diff .^ 2)))
@info @sprintf("  max|reldiff| = %.2e", maximum(abs, wet_reldiff))
@info @sprintf("  mean|reldiff|= %.2e", mean(abs, wet_reldiff))
flush(stdout); flush(stderr)

################################################################################
# Generate diagnostic plots
################################################################################

plot_output_dir = joinpath(serial_dir, "plots", "compare_$(GPU_TAG)")
mkpath(plot_output_dir)
@info "Saving comparison plots to $plot_output_dir"
flush(stdout); flush(stderr)

# Plot serial age (reference)
@info "Plotting serial age (reference)"
plot_age_diagnostics(
    age_serial_yr, grid, wet3D, vol_3D, plot_output_dir, "serial_1year_$(ADVECTION_SCHEME)";
    colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
)

# Plot distributed age
@info "Plotting distributed age ($(GPU_TAG))"
plot_age_diagnostics(
    age_dist_yr, grid, wet3D, vol_3D, plot_output_dir, "distributed_$(GPU_TAG)_1year_$(ADVECTION_SCHEME)";
    colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
)

# Plot absolute difference
@info "Plotting absolute difference"
max_abs_diff = maximum(abs, wet_diff)
diff_range = max_abs_diff > 0 ? (-max_abs_diff, max_abs_diff) : (-1.0e-10, 1.0e-10)
n_levels = 11
diff_step = (diff_range[2] - diff_range[1]) / (n_levels - 1)
diff_levels = range(diff_range[1], diff_range[2]; length = n_levels)
plot_age_diagnostics(
    age_diff_yr, grid, wet3D, vol_3D, plot_output_dir, "diff_$(GPU_TAG)_1year_$(ADVECTION_SCHEME)";
    colorrange = diff_range, levels = diff_levels, colormap = :balance,
)

# Plot relative difference
@info "Plotting relative difference"
max_abs_reldiff = maximum(abs, wet_reldiff)
reldiff_range = max_abs_reldiff > 0 ? (-max_abs_reldiff, max_abs_reldiff) : (-1.0e-10, 1.0e-10)
reldiff_levels = range(reldiff_range[1], reldiff_range[2]; length = n_levels)
plot_age_diagnostics(
    age_reldiff, grid, wet3D, vol_3D, plot_output_dir, "reldiff_$(GPU_TAG)_1year_$(ADVECTION_SCHEME)";
    colorrange = reldiff_range, levels = reldiff_levels, colormap = :balance,
)

@info "compare_1year_runs.jl complete (GPU_TAG=$GPU_TAG)"
flush(stdout); flush(stderr)
