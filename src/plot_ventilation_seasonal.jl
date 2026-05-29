"""
Plot the **seasonal** surface ventilation diagnostic `calVdown` (DJF / MAM / JJA
/ SON) for a single (parent model, experiment, time window, leg).

Reads the seasonal `calVdown_raw_{DJF,MAM,JJA,SON}` arrays written by
`compute_ventilation_diagnostic.jl` into
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK[_QAxB]/ventilation.jld2
normalises with `1e16 / vtot`, and lays out a 2 × 2 panel of season maps that
share a single colour scale (the same orange pseudo-log ramp as the annual-mean
`plot_ventilation.jl`). Uses the per-cell quad mesh `plotmap!` helper on the
tripolar grid (no re-grid, no dateline artifacts) with `GeoMakie.coastlines()`.

Writes one PNG per (PM, EXP, TW, leg) to
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK[_QAxB]/plots/calVdown_seasonal_{forward|adjoint}.png

The forward leg's FTS gives the forward-age-based 𝒱↑; the `_traf` (adjoint) leg
gives the adjoint-age-based 𝒱↓ — same as `plot_ventilation.jl`.

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/plot_ventilation_seasonal.jl
```

Required env vars: PARENT_MODEL, EXPERIMENT (auto from PM if unset), TIME_WINDOW,
VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER.
Optional: LUMP_AND_SPRAY, TRAF, PARTITION_X, PARTITION_Y, OMEGA, VENT_LEVELS_P.
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
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

(; parentmodel, experiment, time_window, experiment_dir, outputdir) = load_project_config()

model_config = require_env("MODEL_CONFIG")

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
leg_tag = TRAF ? "adjoint" : "forward"
leg_label_long = TRAF ? "Adjoint 𝒱↓" : "Forward 𝒱↑"

ls = parse_lump_and_spray()

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

omega = parse_omega()
omega_suffix = omega.suffix
vent_basename = "ventilation$(omega_suffix).jld2"

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

nk_candidates = unique(
    [
        joinpath(root, sub)
            for root in periodic_roots
            for sub in ("NK$(ls.dir_suffix)", "NK")
    ]
)
nk_hit = findfirst(d -> isfile(joinpath(d, vent_basename)), nk_candidates)
nk_hit === nothing && error(
    "$vent_basename not found for TIME_WINDOW=$time_window. Tried:\n" *
        join(["  " * joinpath(d, vent_basename) for d in nk_candidates], "\n") *
        "\nRun compute_ventilation_diagnostic.jl for TIME_WINDOW=$time_window first.",
)
nk_dir = nk_candidates[nk_hit]
vent_file = joinpath(nk_dir, vent_basename)

plot_dir = joinpath(nk_dir, "plots")
mkpath(plot_dir)

@info "plot_ventilation_seasonal.jl configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- EXPERIMENT    = $experiment"
@info "- TIME_WINDOW   = $time_window"
@info "- model_config  = $model_config"
@info "- leg           = $leg_tag"
@info "- input         = $vent_file"
@info "- output dir    = $plot_dir"
flush(stdout); flush(stderr)

################################################################################
# Load data and grid
################################################################################

@info "Loading $vent_file"
flush(stdout); flush(stderr)
d = load(vent_file)

const SEASONS = ("DJF", "MAM", "JJA", "SON")
seasonal_keys = Dict(s => "calVdown_raw_$(s)" for s in SEASONS)
for s in SEASONS
    haskey(d, seasonal_keys[s]) || error(
        "$(seasonal_keys[s]) missing from $vent_file — re-run " *
            "compute_ventilation_diagnostic.jl (the seasonal arrays were added there).",
    )
end

vtot = d["vtot"]
Az_surf = d["Az_surf"]
norm_factor = 1.0e16 / vtot
calV = Dict(s => d[seasonal_keys[s]] .* norm_factor for s in SEASONS)
counts = haskey(d, "season_counts") ? d["season_counts"] : nothing

@info @sprintf("v_tot = %.3e m³;  1e16/v_tot = %.3e", vtot, norm_factor)
for s in SEASONS
    vals = filter(isfinite, calV[s])
    cnt = counts === nothing ? "?" : getfield(counts, Symbol(s))
    @info @sprintf(
        "%s (n=%s) [%% v_tot / (10,000 km)²]:  min=%+.3e mean=%+.3e max=%+.3e",
        s, string(cnt),
        isempty(vals) ? NaN : minimum(vals),
        isempty(vals) ? NaN : mean(vals),
        isempty(vals) ? NaN : maximum(vals),
    )
end

@info "Loading grid"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
ug = grid.underlying_grid
Nx, Ny = size(calV["DJF"])

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
# Colour scale — shared across the four seasons (same orange ramp as the
# annual-mean plot_ventilation.jl). Levels picked from the global seasonal max.
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

user_p = (haskey(ENV, "VENT_LEVELS_P") && !isempty(ENV["VENT_LEVELS_P"])) ?
    parse(Float64, ENV["VENT_LEVELS_P"]) : nothing
maxv = maximum(maximum(filter(isfinite, calV[s]); init = 0.0) for s in SEASONS)
levels_mean = pick_levels(maxv; user_p)
@info "Seasonal panel levels = $levels_mean  (data max ≈ $maxv)"

cm_mean = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levels_mean); categorical = true)
highclip_mean = cm_mean[end]
cm_mean = cgrad(collect(cm_mean[1:(end - 1)]); categorical = true)
scale_mean = mk_piecewise_linear(levels_mean)

################################################################################
# Build the 2 × 2 figure
################################################################################

@info "Building figure"
flush(stdout); flush(stderr)

fig = Figure(;
    size = (1500, 950), fontsize = 14,
    fonts = (; regular = "Arial", bold = "Arial Bold"),
)

xticks_map = -0:90:1000
yticks_map = -90:30:90
mean_label = rich("% v", subscript("tot"), " / (10,000 km)", superscript("2"))

# Panel positions: row 1 = DJF, MAM; row 2 = JJA, SON. Colorbar in col 1 spans
# both rows; maps in cols 2 (left) and 3 (right).
panel_pos = Dict("DJF" => (1, 2), "MAM" => (1, 3), "JJA" => (2, 2), "SON" => (2, 3))
panel_letter = Dict("DJF" => "a", "MAM" => "b", "JJA" => "c", "SON" => "d")

axes_by_season = Dict{String, Any}()
for s in SEASONS
    r, c = panel_pos[s]
    ax = Axis(
        fig[r, c];
        backgroundcolor = :lightgray,
        xgridvisible = false, ygridvisible = false,
        xticks = (xticks_map, lonticklabel.(xticks_map)),
        yticks = (yticks_map, latticklabel.(yticks_map)),
    )
    plotmap!(
        ax, calV[s], gridmetrics;
        colorrange = extrema(levels_mean),
        colormap = cm_mean,
        highclip = highclip_mean,
        colorscale = scale_mean,
        lon_window_start,
    )
    add_coastlines!(ax)
    ylims!(ax, (-90, 90))
    cnt = counts === nothing ? "" : " (n=$(getfield(counts, Symbol(s))))"
    text!(
        ax, 0, 1; text = "($(panel_letter[s])) $s$cnt",
        align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
    )
    # Hide redundant decorations on interior edges.
    r == 1 && hidexdecorations!(ax; ticks = false, grid = false, ticklabels = true, label = true)
    c == 3 && hideydecorations!(ax; ticks = false, grid = false, ticklabels = true, label = true)
    axes_by_season[s] = ax
end

# Shared vertical colorbar (col 1, both rows), level boundaries as ticks.
N_mean = length(levels_mean) - 1
cb = Colorbar(
    fig[1:2, 1];
    colormap = cm_mean,
    colorrange = (0, N_mean),
    highclip = highclip_mean,
    ticks = (0:N_mean, [isinteger(x) ? string(Int(x)) : string(x) for x in levels_mean]),
    label = mean_label,
    vertical = true, flipaxis = false,
)
cb.height = Relative(0.8)

Label(
    fig[0, 1:3];
    text = rich(
        "Seasonal surface ventilation ", leg_label_long, " — ",
        parentmodel, " ", time_window, " (", model_config, ")",
    ),
    fontsize = 18, tellwidth = false,
)

colsize!(fig.layout, 1, Auto(0.12))
resize_to_layout!(fig)

################################################################################
# Save
################################################################################

outputfile = joinpath(plot_dir, "calVdown_seasonal_$(leg_tag)$(omega_suffix).png")
@info "Saving $outputfile"
flush(stdout); flush(stderr)
save(outputfile, fig)

@info "plot_ventilation_seasonal.jl complete"
flush(stdout); flush(stderr)
