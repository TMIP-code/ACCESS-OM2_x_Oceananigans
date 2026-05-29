"""
Animate the surface ventilation diagnostic `calVdown` over the year, from the
half-monthly snapshots of the 1-year re-run of a converged periodic-NK age
solution (the same `age_periodic_1year.jld2` FieldTimeSeries that
`compute_ventilation_diagnostic.jl` reads).

For each output frame the surface (`k = Nz`) age field is interpolated in time,
mapped to `calVdown = V_surf · age_surf / (τ · Az_surf)`, normalised with
`1e16 / vtot`, and drawn as a per-cell quad mesh on the tripolar grid (the
`plotmap!` helper) with the same orange pseudo-log colour scale as the static
`plot_ventilation.jl`. The colour scale is fixed across frames.

Writes one MP4 per (PM, EXP, TW, leg) to the same `plots/` dir as the age
animations:
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/1year/{solver_tag}/plots/calVdown_{forward|adjoint}_movie.mp4

The forward leg's FTS gives the forward-age-based 𝒱↑; the `_traf` (adjoint) leg
gives the adjoint-age-based 𝒱↓ — matching `plot_ventilation.jl`.

This is a standalone CPU script (no GPU needed), submitted after the 1-year
re-run completes.

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/animate_ventilation.jl
```

Environment variables:
  PARENT_MODEL, EXPERIMENT (auto from PM if unset), TIME_WINDOW,
  VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER,
  LINEAR_SOLVER (default Pardiso), LUMP_AND_SPRAY, TRAF, OMEGA,
  PARTITION_X/Y, VENT_LEVELS_P, VENT_N_FRAMES (default 144), VENT_FRAMERATE (24).
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Units: Time
using CairoMakie
using GeoMakie
using GeometryBasics
using JLD2
using Printf
using Statistics

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")
include(joinpath(@__DIR__, "shared_utils", "plotting_functions.jl"))

(; parentmodel, experiment, time_window, experiment_dir, outputdir, Δt_seconds) =
    load_project_config()

model_config = require_env("MODEL_CONFIG")

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) ||
    error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

ls = parse_lump_and_spray()
lumpspray_tag = ls.tag

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
leg_tag = TRAF ? "adjoint" : "forward"
leg_label_long = TRAF ? "Adjoint 𝒱↓" : "Forward 𝒱↑"

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

# Search both the gpu_tag root and the bare gpu_tag-less root (some partitioned
# runs, e.g. OM2-025 1x2, wrote to `periodic/{MC}` without a `{gpu_tag}` part).
periodic_roots = unique(
    [
        isempty(gpu_tag) ?
            joinpath(outputdir, "periodic", model_config) :
            joinpath(outputdir, "periodic", model_config, gpu_tag),
        joinpath(outputdir, "periodic", model_config),
    ]
)

omega = parse_omega()
omega_suffix = omega.suffix
fts_basename = "age_periodic_1year$(omega_suffix).jld2"

fts_candidates = unique(
    [
        joinpath(root, "1year", "$(LINEAR_SOLVER)_$(tag)", fts_basename)
            for root in periodic_roots
            for tag in (lumpspray_tag, "LSprec", "prec")
    ]
)
fts_hit = findfirst(isfile, fts_candidates)
fts_hit === nothing && error(
    "No 1-year periodic age FieldTimeSeries found. Tried:\n" *
        join(["  " * f for f in fts_candidates], "\n") *
        "\nRun the 1-year re-run step (`run1yrNK` in driver.sh) first.",
)
fts_file = fts_candidates[fts_hit]
plot_dir = joinpath(dirname(fts_file), "plots")
mkpath(plot_dir)

n_frames = parse(Int, get(ENV, "VENT_N_FRAMES", "144"))
framerate = parse(Int, get(ENV, "VENT_FRAMERATE", "24"))

τ = 3 * Δt_seconds

@info "animate_ventilation.jl configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- TIME_WINDOW   = $time_window"
@info "- model_config  = $model_config"
@info "- leg           = $leg_tag"
@info "- FTS input     = $fts_file"
@info "- output dir    = $plot_dir"
@info "- frames        = $n_frames @ $framerate fps"
@info "- τ = 3·Δt      = $(τ) s"
flush(stdout); flush(stderr)

################################################################################
# Load grid, FTS, and surface metrics
################################################################################

@info "Loading grid"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())

@info "Loading age FieldTimeSeries from $fts_file"
flush(stdout); flush(stderr)
age_fts = FieldTimeSeries(fts_file, "age")
n_times = length(age_fts.times)
@info "Found $n_times snapshots; stop_time = $(age_fts.times[end]) s"

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)
k_surf = Nz′
wet_surf = wet3D[:, :, k_surf]

vol_3D = Array(interior(compute_volume(grid)))
V_surf = vol_3D[:, :, k_surf]
vtot = sum(vol_3D[idx])

Az_full = load(grid_file, "Azᶜᶜᵃ")
Hx = grid.underlying_grid.Hx
Hy = grid.underlying_grid.Hy
Az_surf = Az_full[Hx .+ (1:Nx′), Hy .+ (1:Ny′)]

norm_factor = 1.0e16 / vtot

# calVdown (normalised) for the surface age field `age_surf` (Nx′×Ny′, seconds).
function calV_from_age_surf(age_surf)
    cv = (V_surf .* age_surf) ./ (τ .* Az_surf) .* norm_factor
    cv[.!wet_surf] .= NaN
    return cv
end

################################################################################
# Grid metrics + coastlines (same as plot_ventilation.jl)
################################################################################

ug = grid.underlying_grid
Nx, Ny = Nx′, Ny′
lon2D = Array(ug.λᶜᶜᵃ[1:Nx, 1:Ny])
lat2D = Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])
λff_full = Array(ug.λᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)])
φff_full = Array(ug.φᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)])
lon_vertices = Array{eltype(λff_full)}(undef, 4, Nx, Ny)
lat_vertices = Array{eltype(φff_full)}(undef, 4, Nx, Ny)
@inbounds for j in 1:Ny, i in 1:Nx
    lon_vertices[1, i, j] = λff_full[i, j]
    lon_vertices[2, i, j] = λff_full[i + 1, j]
    lon_vertices[3, i, j] = λff_full[i + 1, j + 1]
    lon_vertices[4, i, j] = λff_full[i, j + 1]
    lat_vertices[1, i, j] = φff_full[i, j]
    lat_vertices[2, i, j] = φff_full[i + 1, j]
    lat_vertices[3, i, j] = φff_full[i + 1, j + 1]
    lat_vertices[4, i, j] = φff_full[i, j + 1]
end
gridmetrics = (; lon = lon2D, lat = lat2D, lon_vertices, lat_vertices)

lon_window_start = 20
function add_coastlines!(ax)
    coast = GeoMakie.coastlines()
    cl1 = lines!(ax, coast; color = :black, linewidth = 0.7)
    cl2 = lines!(ax, coast; color = :black, linewidth = 0.7)
    translate!(cl1, 0, 0, 50)
    translate!(cl2, 360, 0, 50)
    return nothing
end

################################################################################
# Fixed colour scale — picked from the max over snapshots 1..(n_times-1)
################################################################################

function pick_levels(maxv; user_p = nothing)
    p = if user_p !== nothing
        user_p
    else
        k = floor(Int, log10(maxv / 30))
        10.0^k
    end
    return Float32[0, p, 3p, 10p, 30p, 100p]
end

@info "Scanning snapshots for colour-scale max"
flush(stdout); flush(stderr)
maxv = 0.0
for n in 1:(n_times - 1)
    age_surf_n = Float64.(@view interior(age_fts[n])[:, :, k_surf])
    cv = calV_from_age_surf(age_surf_n)
    v = maximum(filter(isfinite, cv); init = 0.0)
    v > maxv && (maxv = v)
end
user_p = (haskey(ENV, "VENT_LEVELS_P") && !isempty(ENV["VENT_LEVELS_P"])) ?
    parse(Float64, ENV["VENT_LEVELS_P"]) : nothing
levels_mean = pick_levels(maxv; user_p)
@info "Movie levels = $levels_mean  (data max ≈ $maxv)"

cm_mean = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levels_mean); categorical = true)
highclip_mean = cm_mean[end]
cm_mean = cgrad(collect(cm_mean[1:(end - 1)]); categorical = true)
scale_mean = mk_piecewise_linear(levels_mean)

################################################################################
# Build figure and record
################################################################################

year_s = 365.25 * 86400
stop_time = age_fts.times[end]
frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

xticks_map = -0:90:1000
yticks_map = -90:30:90
mean_label = rich("% v", subscript("tot"), " / (10,000 km)", superscript("2"))

fig = Figure(; size = (1100, 620), fontsize = 14)
title_obs = Observable("")
ax = Axis(
    fig[1, 2];
    title = title_obs,
    backgroundcolor = :lightgray,
    xgridvisible = false, ygridvisible = false,
    xticks = (xticks_map, lonticklabel.(xticks_map)),
    yticks = (yticks_map, latticklabel.(yticks_map)),
)

age_surf_0 = Float64.(@view interior(age_fts[Time(frame_times[1])])[:, :, k_surf])
plt = plotmap!(
    ax, calV_from_age_surf(age_surf_0), gridmetrics;
    colorrange = extrema(levels_mean),
    colormap = cm_mean,
    highclip = highclip_mean,
    colorscale = scale_mean,
    lon_window_start,
)
add_coastlines!(ax)
ylims!(ax, (-90, 90))

N_mean = length(levels_mean) - 1
Colorbar(
    fig[1, 1];
    colormap = cm_mean,
    colorrange = (0, N_mean),
    highclip = highclip_mean,
    ticks = (0:N_mean, [isinteger(x) ? string(Int(x)) : string(x) for x in levels_mean]),
    label = mean_label,
    vertical = true, flipaxis = false,
    height = Relative(0.8),
)
colsize!(fig.layout, 1, Auto(0.1))

outputfile = joinpath(plot_dir, "calVdown_$(leg_tag)_movie$(omega_suffix).mp4")
@info "Recording $n_frames frames → $outputfile"
flush(stdout); flush(stderr)

record(fig, outputfile, 1:n_frames; framerate) do i
    age_surf_i = Float64.(@view interior(age_fts[Time(frame_times[i])])[:, :, k_surf])
    cv = calV_from_age_surf(age_surf_i)
    plt.color[] = vcat(fill.(vec(cv), 4)...)
    title_obs[] = @sprintf(
        "Surface ventilation %s — %s %s (t = %.1f months)",
        leg_label_long, parentmodel, time_window, frame_times[i] / (year_s / 12),
    )
end

@info "animate_ventilation.jl complete — saved $outputfile"
flush(stdout); flush(stderr)
