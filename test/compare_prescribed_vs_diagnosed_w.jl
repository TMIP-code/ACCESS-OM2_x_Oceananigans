"""
Compare age output between 3 w-formulation configurations (1-year runs):
  1. wdiagnosed   — w computed from continuity at each step
  2. wparent      — w prescribed from parent model output
  3. wprediag     — w prescribed from a prior diagnosed-w run

Loads the final snapshot from each `age_1year.jld2`, computes pairwise
differences (diagnosed vs each prescribed variant), and generates diagnostic
plots using the shared `plot_age_diagnostics()` function.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project test/compare_prescribed_vs_diagnosed_w.jl
```

Environment variables:
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE  – cgridtransports | bgridvelocities  (default: cgridtransports)
  ADVECTION_SCHEME – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER      – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  GM_REDI          – yes | no  (default: no)
  MONTHLY_KAPPAV   – yes | no  (default: no)
"""

@info "Loading packages for w-formulation comparison"
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

include("../src/shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, outputdir) = load_project_config(; parentmodel_arg_index = 2)

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
GM_REDI = lowercase(get(ENV, "GM_REDI", "no")) == "yes"
MONTHLY_KAPPAV = lowercase(get(ENV, "MONTHLY_KAPPAV", "no")) == "yes"

# Build suffix for GM_REDI / MONTHLY_KAPPAV (shared across all 3 configs)
suffix = ""
GM_REDI && (suffix = "$(suffix)_GMREDI")
MONTHLY_KAPPAV && (suffix = "$(suffix)_mkappaV")

DURATION_TAG = "1year"

# The 3 configurations to compare
configs = [
    (label = "wdiagnosed", tag = "$(VELOCITY_SOURCE)_wdiagnosed_$(ADVECTION_SCHEME)_$(TIMESTEPPER)$(suffix)"),
    (label = "wparent", tag = "$(VELOCITY_SOURCE)_wparent_$(ADVECTION_SCHEME)_$(TIMESTEPPER)$(suffix)"),
    (label = "wprediag", tag = "$(VELOCITY_SOURCE)_wprediag_$(ADVECTION_SCHEME)_$(TIMESTEPPER)$(suffix)"),
]

@info "W-formulation comparison configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- GM_REDI          = $GM_REDI"
@info "- MONTHLY_KAPPAV   = $MONTHLY_KAPPAV"
for c in configs
    @info "- $(c.label) → $(c.tag)"
end
flush(stdout); flush(stderr)

# Verify all output files exist
for c in configs
    age_file = joinpath(outputdir, "standardrun", c.tag, "age_$(DURATION_TAG).jld2")
    isfile(age_file) || error("Output file not found for $(c.label): $age_file")
end

################################################################################
# Load grid, masks, volumes
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")

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
# Load final age snapshot from each run
################################################################################

age_fields = Dict{String, Array{Float64, 3}}()

for c in configs
    run_dir = joinpath(outputdir, "standardrun", c.tag)
    iter_keys = list_iterations(run_dir, "age", DURATION_TAG)
    last_iter = iter_keys[end]

    @info "Loading $(c.label): iter $last_iter from $run_dir"
    flush(stdout); flush(stderr)

    age_full, t = load_serial_snapshot(run_dir, "age", DURATION_TAG, last_iter)

    # Strip halos to get interior
    Hx = (size(age_full, 1) - Nx) ÷ 2
    Hy = (size(age_full, 2) - Ny) ÷ 2
    Hz = (size(age_full, 3) - Nz) ÷ 2
    age_interior = age_full[(Hx + 1):(Hx + Nx), (Hy + 1):(Hy + Ny), (Hz + 1):(Hz + Nz)]

    age_fields[c.label] = age_interior ./ year
    @info "  t = $(@sprintf("%.5f", t / year)) yr, shape = $(size(age_interior))"
end
flush(stdout); flush(stderr)

################################################################################
# Pairwise comparisons
################################################################################

pairs = [
    ("wdiagnosed", "wparent"),
    ("wdiagnosed", "wprediag"),
]

@info "Pairwise comparison statistics (final snapshot, wet cells)"
@info @sprintf("  %-30s  %14s  %14s", "pair", "vol_norm(yr)", "max|diff|(yr)")
flush(stdout); flush(stderr)

diffs = Dict{String, Array{Float64, 3}}()

for (a, b) in pairs
    diff = age_fields[b] .- age_fields[a]
    diff_1D = diff[idx]

    vn = vol_norm(diff_1D .* year) # vol_norm expects seconds, convert back
    maxdiff = maximum(abs, diff[wet3D])
    pair_label = "$(a) vs $(b)"

    @info @sprintf("  %-30s  %14.2e  %14.2e", pair_label, vn, maxdiff)

    diffs[pair_label] = diff
end
flush(stdout); flush(stderr)

################################################################################
# Generate diagnostic plots
################################################################################

plot_output_dir = joinpath(outputdir, "standardrun", "plots", "compare_w_formulation")
mkpath(plot_output_dir)
@info "Saving comparison plots to $plot_output_dir"
flush(stdout); flush(stderr)

n_levels = 11

# Reference age plots for each configuration
for c in configs
    @info "Plotting reference age: $(c.label)"
    plot_age_diagnostics(
        age_fields[c.label], grid, wet3D, vol_3D, plot_output_dir,
        "age_$(DURATION_TAG)_$(c.label)";
        colorrange = (-0.1, 1.1), levels = -0.1:0.1:1.1,
    )
end

# Difference plots for each pair
for (a, b) in pairs
    pair_label = "$(a) vs $(b)"
    diff = diffs[pair_label]
    file_tag = "$(a)_vs_$(b)"

    @info "Plotting difference: $pair_label"

    # Absolute difference — colorscale based on mean|diff|
    wet_diff = diff[wet3D]
    mean_abs_diff = mean(abs, wet_diff)
    diff_scale = mean_abs_diff > 0 ? 3 * mean_abs_diff : 1.0e-10
    diff_range = (-diff_scale, diff_scale)
    diff_levels = range(diff_range[1], diff_range[2]; length = n_levels)
    plot_age_diagnostics(
        diff, grid, wet3D, vol_3D, plot_output_dir,
        "diff_$(DURATION_TAG)_$(file_tag)";
        colorrange = diff_range, levels = diff_levels,
        colormap = cgrad(:balance, n_levels - 1, categorical = true),
        lowclip = :blue, highclip = :red,
    )

    flush(stdout); flush(stderr)
end

@info "compare_prescribed_vs_diagnosed_w.jl complete"
flush(stdout); flush(stderr)
