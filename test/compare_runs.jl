"""
Compare two age fields from any supported source type.

Loads age data from SOURCE_A and SOURCE_B, computes differences (absolute and
relative), prints summary statistics, and generates diagnostic plots.

Source specification format (colon-delimited):
  "serial:MODEL_CONFIG:DURATION_TAG"   — final snapshot from standardrun
  "NK:MODEL_CONFIG:SOLVER_TAG"         — periodic steady-state from NK solver

Environment variables:
  SOURCE_A       – source spec for reference field (required)
  SOURCE_B       – source spec for comparison field (required)
  COMPARE_LABEL  – output directory label (default: auto-generated from specs)
  PARENT_MODEL   – model resolution tag  (default: ACCESS-OM2-1)

Examples:
  # Serial 1-year: wdiagnosed vs wparent
  SOURCE_A="serial:cgridtransports_wdiagnosed_centered2_AB2:1year" \\
  SOURCE_B="serial:cgridtransports_wparent_centered2_AB2:1year" \\
  julia --project test/compare_runs.jl

  # NK periodic: wdiagnosed vs wparent
  SOURCE_A="NK:cgridtransports_wdiagnosed_centered2_AB2:Pardiso_LSprec" \\
  SOURCE_B="NK:cgridtransports_wparent_centered2_AB2:Pardiso_LSprec" \\
  julia --project test/compare_runs.jl
"""

@info "Loading packages for run comparison"
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

SOURCE_A = get(ENV, "SOURCE_A", "")
SOURCE_B = get(ENV, "SOURCE_B", "")
isempty(SOURCE_A) && error("SOURCE_A env var required")
isempty(SOURCE_B) && error("SOURCE_B env var required")

spec_a = parse_source_spec(SOURCE_A)
spec_b = parse_source_spec(SOURCE_B)

function spec_short_label(spec)
    if spec.type == :serial
        return "serial_$(spec.model_config)_$(spec.duration_tag)"
    elseif spec.type == :NK
        return "NK_$(spec.model_config)_$(spec.solver_tag)"
    end
end

label_a = spec_short_label(spec_a)
label_b = spec_short_label(spec_b)

COMPARE_LABEL = get(ENV, "COMPARE_LABEL") do
    "$(label_a)__vs__$(label_b)"
end

@info "Run comparison configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- SOURCE_A      = $SOURCE_A"
@info "- SOURCE_B      = $SOURCE_B"
@info "- COMPARE_LABEL = $COMPARE_LABEL"
flush(stdout); flush(stderr)

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

v1D = vol_3D[idx]
vol_norm = make_vol_norm(v1D, year)

################################################################################
# Load age data
################################################################################

@info "Loading age field A: $SOURCE_A"
flush(stdout); flush(stderr)
age_a_sec = load_final_age_interior(spec_a, outputdir, Nx, Ny, Nz)

@info "Loading age field B: $SOURCE_B"
flush(stdout); flush(stderr)
age_b_sec = load_final_age_interior(spec_b, outputdir, Nx, Ny, Nz)

age_a = age_a_sec ./ year
age_b = age_b_sec ./ year

################################################################################
# Comparison statistics
################################################################################

diff = age_b .- age_a
diff_1D = diff[idx]

vn = vol_norm(diff_1D .* year)
maxdiff = maximum(abs, diff[wet3D])

# Relative difference (skip where reference age is near zero)
reldiff = similar(diff)
for i in eachindex(diff)
    if wet3D[i] && abs(age_a[i]) > 1.0e-10
        reldiff[i] = diff[i] / age_a[i]
    else
        reldiff[i] = NaN
    end
end
wet_reldiff = filter(!isnan, reldiff[wet3D])
mean_abs_reldiff = isempty(wet_reldiff) ? NaN : mean(abs, wet_reldiff)

@info "Comparison statistics (wet cells)"
@info @sprintf("  vol_norm     = %.2e yr", vn)
@info @sprintf("  max|diff|    = %.2e yr", maxdiff)
@info @sprintf("  mean|diff|   = %.2e yr", mean(abs, diff[wet3D]))
@info @sprintf("  mean|reldiff|= %.2e", mean_abs_reldiff)
flush(stdout); flush(stderr)

################################################################################
# Diagnostic plots
################################################################################

plot_output_dir = joinpath(outputdir, "plots", "compare", COMPARE_LABEL)
mkpath(plot_output_dir)
@info "Saving comparison plots to $plot_output_dir"
flush(stdout); flush(stderr)

n_levels = 11

# Adaptive colorrange for reference plots
function adaptive_colorrange(age, wet3D; n_levels = 11)
    wet_vals = filter(isfinite, age[wet3D])
    lo = quantile(wet_vals, 0.01)
    hi = quantile(wet_vals, 0.99)
    # Round to nice numbers
    lo = floor(lo; digits = 1)
    hi = ceil(hi; digits = 1)
    if lo ≈ hi
        lo -= 0.1
        hi += 0.1
    end
    levels = range(lo, hi; length = n_levels)
    return (; colorrange = (lo, hi), levels)
end

# Short tags for filenames and titles
tag_a = spec_a.model_config
tag_b = spec_b.model_config
type_a = String(spec_a.type)
type_b = String(spec_b.type)

# Reference age: A
@info "Plotting reference age: $tag_a ($type_a)"
(; colorrange, levels) = adaptive_colorrange(age_a, wet3D; n_levels)
plot_age_diagnostics(
    age_a, grid, wet3D, vol_3D, plot_output_dir,
    "age_$(tag_a)";
    colorrange, levels,
    title_prefix = "age ($type_a) $tag_a",
)

# Reference age: B
@info "Plotting reference age: $tag_b ($type_b)"
(; colorrange, levels) = adaptive_colorrange(age_b, wet3D; n_levels)
plot_age_diagnostics(
    age_b, grid, wet3D, vol_3D, plot_output_dir,
    "age_$(tag_b)";
    colorrange, levels,
    title_prefix = "age ($type_b) $tag_b",
)

# Absolute difference
@info "Plotting absolute difference: $tag_b minus $tag_a"
wet_diff = diff[wet3D]
mean_abs_diff = mean(abs, wet_diff)
diff_scale = mean_abs_diff > 0 ? 3 * mean_abs_diff : 1.0e-10
diff_range = (-diff_scale, diff_scale)
diff_levels = range(diff_range[1], diff_range[2]; length = n_levels)
plot_age_diagnostics(
    diff, grid, wet3D, vol_3D, plot_output_dir,
    "diff_$(tag_b)_minus_$(tag_a)";
    colorrange = diff_range, levels = diff_levels,
    colormap = cgrad(:balance, n_levels - 1, categorical = true),
    lowclip = :blue, highclip = :red,
    colorbar_label = "Δage (years)",
    title_prefix = "Δage: $tag_b − $tag_a",
)

# Relative difference
if !isempty(wet_reldiff)
    @info "Plotting relative difference (mean|reldiff| = $(@sprintf("%.2e", mean_abs_reldiff)))"
    reldiff_scale = mean_abs_reldiff > 0 ? 3 * mean_abs_reldiff : 1.0e-10
    reldiff_range = (-reldiff_scale, reldiff_scale)
    reldiff_levels = range(reldiff_range[1], reldiff_range[2]; length = n_levels)
    plot_age_diagnostics(
        reldiff, grid, wet3D, vol_3D, plot_output_dir,
        "reldiff_$(tag_b)_minus_$(tag_a)";
        colorrange = reldiff_range, levels = reldiff_levels,
        colormap = cgrad(:balance, n_levels - 1, categorical = true),
        lowclip = :blue, highclip = :red,
        colorbar_label = "Δage / age",
        title_prefix = "Δage/age: $tag_b − $tag_a",
    )
end

flush(stdout); flush(stderr)
@info "compare_runs.jl complete — plots in $plot_output_dir"
flush(stdout); flush(stderr)
