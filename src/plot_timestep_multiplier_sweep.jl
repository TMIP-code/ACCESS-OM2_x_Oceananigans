"""
Post-hoc comparison of `TIMESTEP_MULT` sweep results.

For the current `{PM, EXP, TW, MC}` tuple, discovers every `{MC}` and
`{MC}_DTx{M}/` directory under `outputs/{PM}/{EXP}/{TW}/standardrun/` that
contains an `age_1year.jld2` file. For each `M` it loads the final age
snapshot and reports:

  - volume-weighted mean / max / min age
  - n_wet (sanity)
  - RMS Δ vs M=1 (whole-domain + surface layer)
  - max|Δ| and its (i, j, k) location

The summary is printed to stdout *and* written as a TSV to
`outputs/{PM}/{EXP}/{TW}/standardrun/timestep_multiplier_summary.tsv` so
the results can be pasted directly into `docs/timestep_multiplier.md`.

CPU-only. Designed to be submitted via
`scripts/plotting/plot_timestep_multiplier_sweep.sh`.

Environment variables (inherited from env_defaults.sh):
  PARENT_MODEL, EXPERIMENT, TIME_WINDOW
  VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, GM_REDI, MONTHLY_KAPPAV
"""

@info "Loading packages for timestep-multiplier sweep comparison"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.OutputReaders: InMemory
using JLD2
using Statistics
using Printf

include("shared_functions.jl")

year_seconds = 365.25 * 86400

(; parentmodel, experiment_dir, outputdir) = load_project_config()
(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
mc_base = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)
# Strip any _DTx suffix that build_model_config tagged on from the current
# TIMESTEP_MULT — we want the bare `{MC}` so we can scan all sibling _DTx dirs.
mc_base = replace(mc_base, r"_DTx\d+$" => "")

standardrun_dir = joinpath(outputdir, "standardrun")
isdir(standardrun_dir) || error("standardrun directory not found: $standardrun_dir")

# Discover M values: scan for {MC_base}{suffix}/age_1year.jld2 where suffix is "" or "_DTxN".
function discover_runs(standardrun_dir, mc_base)
    runs = Tuple{Int, String, String}[]  # (M, dirname, age_file)
    for d in readdir(standardrun_dir)
        full = joinpath(standardrun_dir, d)
        isdir(full) || continue
        age_file = joinpath(full, "age_1year.jld2")
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

runs = discover_runs(standardrun_dir, mc_base)
isempty(runs) && error("No {MC}_DTx{M}/age_1year.jld2 found under $standardrun_dir for MC_base=$mc_base")
@info "Discovered $(length(runs)) runs to compare"
for (M, d, _) in runs
    @info "  M=$M → $d"
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

function load_final_age(age_file, Hx, Hy, Hz, Nx, Ny, Nz, year_seconds)
    return jldopen(age_file, "r") do f
        ages = f["timeseries/age"]
        iters = filter(k -> tryparse(Int, k) !== nothing, keys(ages))
        last_key = string(maximum(parse.(Int, iters)))
        arr = Float64.(ages[last_key])
        interior_view = arr[(Hx + 1):(Hx + Nx), (Hy + 1):(Hy + Ny), (Hz + 1):(Hz + Nz)]
        return interior_view ./ year_seconds  # → years
    end
end

@info "Loading age fields"
flush(stdout); flush(stderr)
age_by_M = Dict{Int, Array{Float64, 3}}()
for (M, d, age_file) in runs
    @info "  M=$M loading $age_file"
    age_by_M[M] = load_final_age(age_file, Hx, Hy, Hz, Nx, Ny, Nz, year_seconds)
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
println("TIMESTEP_MULT sweep summary — $parentmodel / MC_base = $mc_base")
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

tsv_path = joinpath(standardrun_dir, "timestep_multiplier_summary.tsv")
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

@info "plot_timestep_multiplier_sweep.jl complete"
flush(stdout); flush(stderr)
