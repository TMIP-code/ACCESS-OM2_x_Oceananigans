"""
Post-hoc comparison of `TIMESTEP_MULT` sweep results.

For the current `{PM, EXP, TW, MC}` tuple, discovers every `{MC}` and
`{MC}_DTx{M}/` directory containing the target age field, then per `M`
reports:

  - volume-weighted mean / max / min age
  - n_wet (sanity)
  - RMS Δ vs M=1 (whole-domain + surface layer)
  - max|Δ| and its (i, j, k) location

`COMPARE_TARGET` selects which age field to compare (default `standardrun`
for back-compat with the stability sweep):

  - `standardrun`   → `outputs/.../standardrun/{MC}_DTx{M}/age_1year.jld2`
                      (1-year forward map, last snapshot, `timeseries/age/{iter}`)
  - `nk_steady`     → `outputs/.../periodic/{MC}_DTx{M}/NK/age_{LINEAR_SOLVER}_{precond_tag}.jld2`
                      (final NK fixed point, bare `age` 3D array)
  - `nk_periodic`   → `outputs/.../periodic/{MC}_DTx{M}/1year/{LINEAR_SOLVER}_{precond_tag}/age_periodic_1year.jld2`
                      (1-year integration from NK solution, year-mean over the
                      first `n_times − 1` half-monthly snapshots)

The summary is printed to stdout *and* written as a TSV to a
target-dependent path under the discovery root so the three modes don't
overwrite each other:

  - standardrun     → `standardrun/timestep_multiplier_summary.tsv`
  - nk_steady       → `periodic/timestep_multiplier_summary_nk_steady.tsv`
  - nk_periodic     → `periodic/timestep_multiplier_summary_nk_periodic.tsv`

Diff PNGs land under `diff_vs_DTx1/` for standardrun and
`diff_vs_DTx1_periodic_{nk_steady,nk_periodic}/` for the two NK modes —
the per-target suffix is required because both modes share `periodic/`
as their discovery root, so a shared subdir would collide when the
two sweep jobs run in parallel (see docs/timestep_multiplier_NK.md).

CPU-only. Designed to be submitted via
`scripts/plotting/plot_timestep_multiplier_sweep.sh`.

Environment variables (inherited from env_defaults.sh):
  PARENT_MODEL, EXPERIMENT, TIME_WINDOW
  VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, GM_REDI, MONTHLY_KAPPAV
  COMPARE_TARGET     (standardrun | nk_steady | nk_periodic; default standardrun)
  LINEAR_SOLVER      (only used when COMPARE_TARGET startswith "nk_"; default Pardiso)
  LUMP_AND_SPRAY     (only used when COMPARE_TARGET startswith "nk_"; default no)
"""

@info "Loading packages for timestep-multiplier sweep comparison"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: InMemory
using JLD2
using Statistics
using Printf
using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian

include("shared_functions.jl")

# OCEANS const used by compute_ocean_basin_masks (mirrors plot_standardrun_age.jl).
const OCEANS = oceanpolygons()

# Opt-out: DIFF_PLOTS=no skips the plot generation (scalar metrics still run).
DIFF_PLOTS = lowercase(get(ENV, "DIFF_PLOTS", "yes")) == "yes"

year_seconds = 365.25 * 86400

(; parentmodel, experiment_dir, outputdir) = load_project_config()
mc_base = replace(require_env("MODEL_CONFIG"), r"_DTx\d+$" => "")

COMPARE_TARGET = lowercase(get(ENV, "COMPARE_TARGET", "standardrun"))
COMPARE_TARGET in ("standardrun", "nk_steady", "nk_periodic") ||
    error("COMPARE_TARGET must be standardrun | nk_steady | nk_periodic (got: \"$COMPARE_TARGET\")")

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
lumpspray_tag = LUMP_AND_SPRAY ? "LSprec" : "prec"
solver_tag = "$(LINEAR_SOLVER)_$(lumpspray_tag)"

# Target-dependent discovery root, per-`M` file resolver, diff-plot subdir,
# TSV filename, and load-format kind.
if COMPARE_TARGET == "standardrun"
    sweep_dir = joinpath(outputdir, "standardrun")
    age_file_for(d) = joinpath(sweep_dir, d, "age_1year.jld2")
    diff_subdir = "diff_vs_DTx1"
    tsv_path = joinpath(sweep_dir, "timestep_multiplier_summary.tsv")
    file_format = :timeseries_last  # JLD2OutputWriter w/ halos; take last iter
elseif COMPARE_TARGET == "nk_steady"
    sweep_dir = joinpath(outputdir, "periodic")
    age_file_for(d) = joinpath(sweep_dir, d, "NK", "age_$(solver_tag).jld2")
    diff_subdir = "diff_vs_DTx1_periodic_nk_steady"
    tsv_path = joinpath(sweep_dir, "timestep_multiplier_summary_nk_steady.tsv")
    file_format = :nk_steady       # bare `age` 3D array, no halos, seconds
else  # nk_periodic
    sweep_dir = joinpath(outputdir, "periodic")
    age_file_for(d) = joinpath(sweep_dir, d, "1year", solver_tag, "age_periodic_1year.jld2")
    diff_subdir = "diff_vs_DTx1_periodic_nk_periodic"
    tsv_path = joinpath(sweep_dir, "timestep_multiplier_summary_nk_periodic.tsv")
    file_format = :timeseries_mean # JLD2OutputWriter w/ halos; mean over first n-1 iters
end

isdir(sweep_dir) || error("Discovery root not found: $sweep_dir (COMPARE_TARGET=$COMPARE_TARGET)")

@info "Sweep configuration"
@info "- COMPARE_TARGET = $COMPARE_TARGET"
@info "- sweep_dir      = $sweep_dir"
@info "- mc_base        = $mc_base"
if COMPARE_TARGET ≠ "standardrun"
    @info "- solver_tag     = $solver_tag  (LINEAR_SOLVER=$LINEAR_SOLVER, LUMP_AND_SPRAY=$LUMP_AND_SPRAY)"
end
@info "- diff_subdir    = $diff_subdir"
@info "- tsv_path       = $tsv_path"

# Discover M values: scan for {MC_base}{suffix}/ where suffix is "" or "_DTxN"
# and the target age file actually exists under it.
function discover_runs(sweep_dir, mc_base, age_file_for)
    runs = Tuple{Int, String, String}[]  # (M, dirname, age_file)
    for d in readdir(sweep_dir)
        full = joinpath(sweep_dir, d)
        isdir(full) || continue
        age_file = age_file_for(d)
        isfile(age_file) || continue
        if d == mc_base
            push!(runs, (1, d, age_file))
        else
            m = match(Regex("^" * mc_base * raw"_DTx(\d+)$"), d)
            m === nothing && continue
            push!(runs, (parse(Int, m.captures[1]), d, age_file))
        end
    end
    return sort!(runs; by = r -> r[1])
end

runs = discover_runs(sweep_dir, mc_base, age_file_for)
isempty(runs) && error("No {MC}_DTx{M}/<age file> found under $sweep_dir for MC_base=$mc_base (COMPARE_TARGET=$COMPARE_TARGET)")
@info "Discovered $(length(runs)) runs to compare"
for (M, d, f) in runs
    @info "  M=$M → $d ($(basename(f)))"
end

################################################################################
# Grid + wet mask + volumes
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())
Nx, Ny, Nz = size(grid)
Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

(; wet3D) = compute_wet_mask(grid)
vol_field = compute_volume(grid)
vol_3D = Array(interior(vol_field))
total_vol = sum(vol_3D[wet3D])
@info "Grid: Nx=$Nx Ny=$Ny Nz=$Nz; n_wet=$(sum(wet3D)); total_vol=$total_vol m³"

################################################################################
# Load each run's final age snapshot (interior, in years)
################################################################################

_strip_halos(arr, Hx, Hy, Hz, Nx, Ny, Nz) =
    arr[(Hx + 1):(Hx + Nx), (Hy + 1):(Hy + Ny), (Hz + 1):(Hz + Nz)]

# JLD2OutputWriter layout: keys under "timeseries/age/" are stringified iter
# numbers; pick the largest (= last snapshot).
function _load_jld2_last_iter(age_file)
    return jldopen(age_file, "r") do f
        ages = f["timeseries/age"]
        iters = filter(k -> tryparse(Int, k) !== nothing, keys(ages))
        last_key = string(maximum(parse.(Int, iters)))
        return Float64.(ages[last_key])
    end
end

# JLD2OutputWriter layout, but mean over the first `n-1` snapshots (year-mean
# of the 25 half-monthly snapshots from run1yrNK — same convention as
# plot_periodic_1year_age.jl, which excludes snapshot 25 = repeat of t=0).
function _load_jld2_mean_first_n_minus_one(age_file)
    return jldopen(age_file, "r") do f
        ages = f["timeseries/age"]
        iters = sort!(parse.(Int, filter(k -> tryparse(Int, k) !== nothing, keys(ages))))
        n_avg = length(iters) - 1
        n_avg ≥ 1 || error("Expected ≥ 2 snapshots in $age_file (got $(length(iters)))")
        acc = Float64.(ages[string(iters[1])])
        for k in iters[2:n_avg]
            acc .+= Float64.(ages[string(k)])
        end
        acc ./= n_avg
        return acc
    end
end

function load_age_years(age_file, file_format, Hx, Hy, Hz, Nx, Ny, Nz, year_seconds)
    if file_format == :timeseries_last
        arr = _load_jld2_last_iter(age_file)
        return _strip_halos(arr, Hx, Hy, Hz, Nx, Ny, Nz) ./ year_seconds
    elseif file_format == :timeseries_mean
        arr = _load_jld2_mean_first_n_minus_one(age_file)
        return _strip_halos(arr, Hx, Hy, Hz, Nx, Ny, Nz) ./ year_seconds
    elseif file_format == :nk_steady
        # solve_periodic_NK.jl writes `age = age_steady_3D` (Nx, Ny, Nz, no halos)
        # in seconds, with dry cells = 0.
        arr = jldopen(age_file, "r") do f
            Float64.(f["age"])
        end
        size(arr) == (Nx, Ny, Nz) ||
            error("NK steady age in $age_file has size $(size(arr)), expected ($Nx, $Ny, $Nz)")
        return arr ./ year_seconds
    else
        error("Unknown file_format: $file_format")
    end
end

@info "Loading age fields (file_format=$file_format)"
flush(stdout); flush(stderr)
age_by_M = Dict{Int, Array{Float64, 3}}()
for (M, d, age_file) in runs
    @info "  M=$M loading $age_file"
    age_by_M[M] = load_age_years(age_file, file_format, Hx, Hy, Hz, Nx, Ny, Nz, year_seconds)
end

################################################################################
# Per-M scalar diagnostics
################################################################################

vol_wet = vol_3D[wet3D]
surf_idx = wet3D[:, :, Nz]
surf_vol = vol_3D[:, :, Nz][surf_idx]

per_M = Dict{Int, NamedTuple}()
for (M, _, _) in runs
    age = age_by_M[M]
    age_wet = age[wet3D]
    mean_age = sum(age_wet .* vol_wet) / total_vol
    max_age = maximum(age_wet)
    min_age = minimum(age_wet)
    per_M[M] = (; mean_age, max_age, min_age)
end

################################################################################
# Pairwise diff vs M=1
################################################################################

haskey(age_by_M, 1) || error("M=1 baseline not found; cannot compute RMS Δ")
age_ref = age_by_M[1]

diff_stats = Dict{Int, NamedTuple}()
for (M, _, _) in runs
    M == 1 && continue
    diff = age_by_M[M] .- age_ref
    diff_wet = diff[wet3D]
    rms_whole = sqrt(sum((diff_wet .^ 2) .* vol_wet) / total_vol)
    surf_diff_wet = diff[:, :, Nz][surf_idx]
    rms_surf = sqrt(sum((surf_diff_wet .^ 2) .* surf_vol) / sum(surf_vol))
    max_abs, max_idx = findmax(abs.(diff_wet))
    # recover (i, j, k) of max|Δ| in the full 3D index space
    flat_idx = findall(wet3D)[max_idx]
    diff_stats[M] = (; rms_whole, rms_surf, max_abs, max_loc = (flat_idx[1], flat_idx[2], flat_idx[3]))
end

################################################################################
# Print summary table
################################################################################

println()
println("="^88)
println("TIMESTEP_MULT sweep summary — $parentmodel / MC_base = $mc_base / target = $COMPARE_TARGET")
println("="^88)
@printf "%4s  %12s  %12s  %12s  %12s  %12s  %12s\n" "M" "mean (yr)" "max (yr)" "min (yr)" "RMS_all (yr)" "RMS_surf (yr)" "max|Δ| (yr)"
println("-"^88)
for (M, _, _) in runs
    s = per_M[M]
    if M == 1
        @printf "%4d  %12.6f  %12.6f  %12.6f  %12s  %12s  %12s\n" M s.mean_age s.max_age s.min_age "—" "—" "—"
    else
        d = diff_stats[M]
        @printf "%4d  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  (i,j,k)=(%d,%d,%d)\n" M s.mean_age s.max_age s.min_age d.rms_whole d.rms_surf d.max_abs d.max_loc[1] d.max_loc[2] d.max_loc[3]
    end
end
println("="^88)

################################################################################
# TSV dump
################################################################################

@info "Writing TSV summary to $tsv_path"
open(tsv_path, "w") do io
    println(io, "M\tmean_age_yr\tmax_age_yr\tmin_age_yr\trms_whole_yr\trms_surf_yr\tmax_abs_diff_yr\tmax_diff_i\tmax_diff_j\tmax_diff_k")
    for (M, _, _) in runs
        s = per_M[M]
        if M == 1
            @printf io "%d\t%.6f\t%.6f\t%.6f\t\t\t\t\t\t\n" M s.mean_age s.max_age s.min_age
        else
            d = diff_stats[M]
            @printf io "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\n" M s.mean_age s.max_age s.min_age d.rms_whole d.rms_surf d.max_abs d.max_loc[1] d.max_loc[2] d.max_loc[3]
        end
    end
end

################################################################################
# Diff plots (per M > 1)
################################################################################

if DIFF_PLOTS && length(runs) > 1
    @info "Generating diff plots for each M > 1"
    flush(stdout); flush(stderr)
    for (M, d, _) in runs
        M == 1 && continue
        diff_field = age_by_M[M] .- age_ref
        diff_wet = diff_field[wet3D]
        # Auto-scale colour range to the 99th percentile of |diff| so a single
        # outlier doesn't compress the colour scale. Use the same Δmax for all
        # zonal/horizontal plots of this M so they're directly comparable.
        Δmax = quantile(abs.(diff_wet), 0.99)
        Δmax > 0 || (Δmax = maximum(abs.(diff_wet)) + eps())
        levels = range(-Δmax, Δmax; length = 21)
        n_steps = length(levels) - 1
        # ColorSchemes.jl exposes :RdBu but not :RdBu_r; reverse explicitly so
        # negative-Δ (M ages less than M=1) lands on blue and positive on red.
        cmap = cgrad(:RdBu, n_steps; categorical = true, rev = true)
        diff_dir = joinpath(sweep_dir, d, diff_subdir)
        label = "DTx$(M)_vs_DTx1"
        title_prefix = @sprintf "M=%d − M=1 (Δmax≈%.3f yr at 99th pct)" M Δmax
        @info "  M=$M  Δmax=$Δmax yr → $diff_dir"
        plot_age_diagnostics(
            diff_field .* year_seconds, grid, wet3D, vol_3D, diff_dir, label;
            colorrange = (-Δmax, Δmax),
            levels,
            colormap = cmap,
            lowclip = cmap[1],
            highclip = cmap[end],
            colorbar_label = "Age diff (years)",
            title_prefix,
        )
    end
else
    @info "Skipping diff plots (DIFF_PLOTS=$(get(ENV, "DIFF_PLOTS", "yes")), $(length(runs)) runs)"
end

@info "plot_timestep_multiplier_sweep.jl complete"
flush(stdout); flush(stderr)
