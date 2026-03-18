"""
Compare age output between serial (1 GPU) and distributed (multi-GPU) runs.

Loads each snapshot (part file) one at a time, computes volume-weighted
RMS norm of the difference, then plots diagnostics of the final snapshot.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
GPU_TAG=2x2 DURATION_TAG=1year julia --project src/compare_runs_across_architectures.jl
```

Environment variables:
  GPU_TAG          – partition tag to compare against serial (e.g., "2x2")
  DURATION_TAG     – output tag (default: "1year", can be "diag" for 10-step diagnostic)
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
DURATION_TAG = get(ENV, "DURATION_TAG", "1year")

# Parse partition from GPU_TAG (e.g., "2x2" → px=2, py=2)
gpu_tag_parts = split(GPU_TAG, "x")
length(gpu_tag_parts) == 2 || error("GPU_TAG must be in format PxQ (e.g., 2x2), got: $GPU_TAG")
px = parse(Int, gpu_tag_parts[1])
py = parse(Int, gpu_tag_parts[2])

(; parentmodel, outputdir) = load_project_config(; parentmodel_arg_index = 2)

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

serial_dir = joinpath(outputdir, "standardrun", model_config)
distributed_dir = joinpath(outputdir, "standardrun", model_config, GPU_TAG)

@info "Architecture comparison configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- GPU_TAG          = $GPU_TAG ($(px)x$(py) partition)"
@info "- DURATION_TAG     = $DURATION_TAG"
@info "- model_config     = $model_config"
@info "- Serial dir       = $serial_dir"
@info "- Distributed dir  = $distributed_dir"
flush(stdout); flush(stderr)

# Verify part files exist
serial_part1 = joinpath(serial_dir, "age_$(DURATION_TAG)_part1.jld2")
dist_part1 = joinpath(distributed_dir, "age_$(DURATION_TAG)_rank0_part1.jld2")
isfile(serial_part1) || error("Serial part file not found: $serial_part1")
isfile(dist_part1) || error("Distributed part file not found: $dist_part1")

# Auto-detect number of parts from serial directory
NPARTS = length(filter(f -> startswith(f, "age_$(DURATION_TAG)_part") && endswith(f, ".jld2"), readdir(serial_dir)))
@info "Detected $NPARTS part files for DURATION_TAG=$DURATION_TAG"

################################################################################
# Load grid, masks, volumes
################################################################################

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")

@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

@info "Computing wet mask and cell volumes"
flush(stdout); flush(stderr)

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx, Ny, Nz = size(wet3D)

vol = compute_volume(grid)
vol_3D = Array(interior(vol))

# Volume-weighted RMS norm (in years)
v1D = vol_3D[idx]
vol_norm = make_vol_norm(v1D, year)

################################################################################
# Per-snapshot comparison (load one part at a time)
################################################################################

@info "Comparing $NPARTS snapshots"
@info @sprintf("  %5s  %10s  %14s  %14s", "part", "time(yr)", "vol_norm(yr)", "max|diff|(yr)")
flush(stdout); flush(stderr)

# These will hold the last-loaded data for final diagnostic plots
age_serial_yr = nothing
age_dist_yr = nothing
age_diff_yr = nothing

for part in 1:NPARTS
    age_serial_full, t_serial = load_serial_part(serial_dir, "age", DURATION_TAG, part)
    age_dist_raw, t_dist = load_distributed_part(distributed_dir, "age", DURATION_TAG, part, px, py, Nx, Ny, Nz)

    # Trim serial data to interior size (fold row may be included in serial output)
    age_serial_raw = @view age_serial_full[1:Nx, 1:Ny, 1:Nz]

    diff_raw = age_dist_raw .- age_serial_raw
    diff_1D = diff_raw[idx]

    vn = vol_norm(diff_1D)
    maxdiff = maximum(abs, diff_1D) / year
    t_yr = t_serial / year

    @info @sprintf("  %5d  %10.5f  %14.2e  %14.2e", part, t_yr, vn, maxdiff)

    # Keep last snapshot for plotting
    if part == NPARTS
        global age_serial_yr = Array(age_serial_raw) ./ year
        global age_dist_yr = age_dist_raw ./ year
        global age_diff_yr = age_dist_yr .- age_serial_yr
    end
end
flush(stdout); flush(stderr)

################################################################################
# Summary statistics for final snapshot (wet cells only)
################################################################################

wet_diff = age_diff_yr[wet3D]

# Relative difference (avoid division by zero)
age_reldiff = similar(age_diff_yr)
for i in eachindex(age_diff_yr)
    if wet3D[i] && abs(age_serial_yr[i]) > 1.0e-10
        age_reldiff[i] = age_diff_yr[i] / age_serial_yr[i]
    else
        age_reldiff[i] = NaN
    end
end
wet_reldiff = filter(!isnan, age_reldiff[wet3D])

@info "Final snapshot statistics (wet cells only):"
@info @sprintf("  max|diff|    = %.2e years", maximum(abs, wet_diff))
@info @sprintf("  mean|diff|   = %.2e years", mean(abs, wet_diff))
@info @sprintf("  RMS diff     = %.2e years", sqrt(mean(wet_diff .^ 2)))
if !isempty(wet_reldiff)
    @info @sprintf("  max|reldiff| = %.2e", maximum(abs, wet_reldiff))
    @info @sprintf("  mean|reldiff|= %.2e", mean(abs, wet_reldiff))
else
    @info "  reldiff: not computed (all values near zero)"
end
flush(stdout); flush(stderr)

################################################################################
# Generate diagnostic plots for final snapshot
################################################################################

plot_output_dir = joinpath(serial_dir, "plots", "compare_$(GPU_TAG)_$(DURATION_TAG)")
mkpath(plot_output_dir)
@info "Saving comparison plots to $plot_output_dir"
flush(stdout); flush(stderr)

# Plot serial age (reference)
@info "Plotting serial age (reference)"
plot_age_diagnostics(
    age_serial_yr, grid, wet3D, vol_3D, plot_output_dir, "serial_$(DURATION_TAG)_$(ADVECTION_SCHEME)";
    colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
)

# Plot distributed age
@info "Plotting distributed age ($(GPU_TAG))"
plot_age_diagnostics(
    age_dist_yr, grid, wet3D, vol_3D, plot_output_dir, "distributed_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)";
    colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
)

# Plot absolute difference
@info "Plotting absolute difference"
max_abs_diff = maximum(abs, wet_diff)
diff_range = max_abs_diff > 0 ? (-max_abs_diff, max_abs_diff) : (-1.0e-10, 1.0e-10)
n_levels = 11
diff_levels = range(diff_range[1], diff_range[2]; length = n_levels)
plot_age_diagnostics(
    age_diff_yr, grid, wet3D, vol_3D, plot_output_dir, "diff_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)";
    colorrange = diff_range, levels = diff_levels, colormap = :balance,
)

# Plot relative difference (skip if all values near zero)
if !isempty(wet_reldiff)
    @info "Plotting relative difference"
    max_abs_reldiff = maximum(abs, wet_reldiff)
    reldiff_range = max_abs_reldiff > 0 ? (-max_abs_reldiff, max_abs_reldiff) : (-1.0e-10, 1.0e-10)
    reldiff_levels = range(reldiff_range[1], reldiff_range[2]; length = n_levels)
    plot_age_diagnostics(
        age_reldiff, grid, wet3D, vol_3D, plot_output_dir, "reldiff_$(GPU_TAG)_$(DURATION_TAG)_$(ADVECTION_SCHEME)";
        colorrange = reldiff_range, levels = reldiff_levels, colormap = :balance,
    )
end

@info "compare_runs_across_architectures.jl complete (GPU_TAG=$GPU_TAG, DURATION_TAG=$DURATION_TAG)"
flush(stdout); flush(stderr)
