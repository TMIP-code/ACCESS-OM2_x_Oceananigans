"""
Plot the surface ventilation diagnostic `calVdown` produced by
`compute_ventilation_diagnostic.jl`, in the Pasquier *et al.* 2024 / 2025
plotting style: per-cell quad mesh on the tripolar grid (via the helpers in
[src/shared_utils/plotting_functions.jl](shared_utils/plotting_functions.jl))
on a pseudo-log colour scale with levels `[0, 10, 30, 100, 300, 1000]` in
units `% v_tot / (10,000 km²)`.

The script reads **both** time windows (1968-1977 and 1999-2008) for one
(PM, leg) pair and lays out a 2 × 2 panel:

  [1, 1] map of `calV` at 1968-1977 (orange ramp, pseudo-log scale)
  [1, 2] zonal-integral side panel — both windows overlaid as lines
  [2, 1] decadal-difference map (1999-2008 − 1968-1977) — diverging :tol_bu_rd
  [2, 2] zonal-integral of the difference (bicolour band, red / blue)

It loads
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK[_QAxB]/ventilation.jld2
for `TW ∈ {1968-1977, 1999-2008}` (mirrors the same dual-naming fallback as
`compute_ventilation_diagnostic.jl`), normalises with `1e12 / vtot`, and
writes one PNG per (PM, leg) to
  outputs/{PM}/{EXP}/plots/{MC}/calVdown_{forward|adjoint}.png

Handles both forward (IAF) and adjoint (TRAF) legs uniformly via the
`_traf` suffix that `build_model_config` appends to `model_config` when
`TRAF=yes`.

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/plot_ventilation.jl
```

Required env vars: PARENT_MODEL, EXPERIMENT (auto from PM if unset),
VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER.
Optional: LUMP_AND_SPRAY, TRAF, PARTITION_X, PARTITION_Y.
`TIME_WINDOW` is **not** read — both 1968-1977 and 1999-2008 are required
on disk.
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

(; parentmodel, experiment, experiment_dir, outputdir) = load_project_config()

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
leg_tag = TRAF ? "adjoint" : "forward"

ls = parse_lump_and_spray()

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

# Hard-coded time windows. The figure compares these two; if you want a
# different pair, edit here (and the labels below).
const TW1 = "1968-1977"
const TW2 = "1999-2008"

# Experiment-level outputs root (parent of the per-TW dirs). outputdir from
# load_project_config is per-TW (uses ENV["TIME_WINDOW"] which we ignore),
# so the experiment-level dir is one above it.
exp_outdir = dirname(outputdir)

function ventilation_path(tw)
    tw_root = joinpath(exp_outdir, tw)
    periodic_root = isempty(gpu_tag) ?
        joinpath(tw_root, "periodic", model_config) :
        joinpath(tw_root, "periodic", model_config, gpu_tag)
    candidate_dirs = unique(
        [
            joinpath(periodic_root, "NK$(ls.dir_suffix)"),
            joinpath(periodic_root, "NK"),
        ]
    )
    for d in candidate_dirs
        f = joinpath(d, "ventilation.jld2")
        isfile(f) && return f
    end
    error(
        "ventilation.jld2 not found for TIME_WINDOW=$tw. Tried:\n" *
            join(["  " * joinpath(d, "ventilation.jld2") for d in candidate_dirs], "\n") *
            "\nRun compute_ventilation_diagnostic.jl for TIME_WINDOW=$tw first " *
            "(or `bash scripts/driver.sh JOB_CHAIN=ventilation TIME_WINDOW=$tw …`).",
    )
end

vent_file_1 = ventilation_path(TW1)
vent_file_2 = ventilation_path(TW2)

plot_dir = joinpath(exp_outdir, "plots", model_config)
mkpath(plot_dir)

@info "plot_ventilation.jl configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- EXPERIMENT    = $experiment"
@info "- model_config  = $model_config"
@info "- leg           = $leg_tag"
@info "- $TW1 input    = $vent_file_1"
@info "- $TW2 input    = $vent_file_2"
@info "- output dir    = $plot_dir"
flush(stdout); flush(stderr)

################################################################################
# Load data and grid
################################################################################

@info "Loading $vent_file_1"
flush(stdout); flush(stderr)
d1 = load(vent_file_1)
@info "Loading $vent_file_2"
flush(stdout); flush(stderr)
d2 = load(vent_file_2)

calVdown_raw_1 = d1["calVdown_raw"]   # m³/m² (= m), NaN at dry cells
calVdown_raw_2 = d2["calVdown_raw"]
vtot = d1["vtot"]                     # m³ — use TW1's; assert TW2 matches
Az_surf = d1["Az_surf"]               # m²

size(calVdown_raw_1) == size(calVdown_raw_2) ||
    error("Shape mismatch between TW1 ($(size(calVdown_raw_1))) and TW2 ($(size(calVdown_raw_2)))")
let δ = abs(d2["vtot"] - vtot) / vtot
    δ < 1.0e-6 || @warn "v_tot differs between time windows" tw1_vtot = vtot tw2_vtot = d2["vtot"] reldiff = δ
end

# Normalise: % v_tot / (10,000 km²)
norm_factor = 1.0e12 / vtot
calV1 = calVdown_raw_1 .* norm_factor
calV2 = calVdown_raw_2 .* norm_factor
calV_diff = calV2 .- calV1

@info @sprintf("v_tot = %.3e m³;  1e12/v_tot = %.3e", vtot, norm_factor)
for (name, v) in (("calV1", calV1), ("calV2", calV2), ("calV_diff", calV_diff))
    vals = filter(isfinite, v)
    @info @sprintf(
        "%s [%% v_tot / 1e4 km²]:  min = %+.3e   mean = %+.3e   max = %+.3e",
        rpad(name, 9), minimum(vals), mean(vals), maximum(vals),
    )
end

################################################################################
# Build gridmetrics from the Oceananigans tripolar grid
################################################################################

@info "Loading grid"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
ug = grid.underlying_grid
Nx, Ny = size(calV1)

# Cell centres (interior). λᶜᶜᵃ / φᶜᶜᵃ are OffsetArrays — interior origin
# is at index 1, halos at indices 1-Hx..0. So [1:Nx, 1:Ny] gives the interior.
lon2D = Array(ug.λᶜᶜᵃ[1:Nx, 1:Ny])
lat2D = Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])

# Cell vertices (corners). Need (Nx+1) × (Ny+1) face-face points.
λff_full = Array(ug.λᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)])
φff_full = Array(ug.φᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)])

# Pack into the (4, Nx, Ny) NamedTuple expected by plotmap!. Vertex order:
# (i, j) → (i+1, j) → (i+1, j+1) → (i, j+1).
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

################################################################################
# Coastlines helper — GeoMakie.coastlines() shifted into the [20, 380] window
################################################################################

lon_window_start = 20  # matches plotmap!'s default

function add_coastlines!(ax)
    coast = GeoMakie.coastlines()   # Vector{LineString{2, Float32}}
    cl1 = lines!(ax, coast; color = :black, linewidth = 0.7)
    cl2 = lines!(ax, coast; color = :black, linewidth = 0.7)
    # Shift the second copy by +360° in lon so coastlines cover both halves
    # of the [20, 380] window. Bring both to front in z.
    translate!(cl1, 0, 0, 50)
    translate!(cl2, 360, 0, 50)
    return nothing
end

################################################################################
# Colour scales and palettes
################################################################################

# Pasquier 2024 spec is [0, 10, 30, 100, 300, 1000] %v_tot/(10,000 km²), tuned to
# sub-basin v_tot (deep ETP, etc.). Whole-ocean v_tot here is ~50× larger, so the
# data falls well below 10 — pick a log-decade ladder of the same shape sized to
# the actual data max. Override by setting VENT_LEVELS_P (the lowest non-zero
# level) explicitly, e.g. VENT_LEVELS_P=10 to recover the Pasquier set.
function pick_levels(maxv; user_p = nothing)
    p = if user_p !== nothing
        user_p
    else
        # Place maxv between the 4th (30p) and 5th (100p) levels.
        k = floor(Int, log10(maxv / 30))
        10.0^k
    end
    return Float32[0, p, 3p, 10p, 30p, 100p]
end

user_p = (haskey(ENV, "VENT_LEVELS_P") && !isempty(ENV["VENT_LEVELS_P"])) ?
    parse(Float64, ENV["VENT_LEVELS_P"]) : nothing
maxv_mean = max(maximum(filter(isfinite, calV1)), maximum(filter(isfinite, calV2)))
levels_mean = pick_levels(maxv_mean; user_p)
@info "Mean panel levels = $levels_mean  (data max ≈ $maxv_mean)"

# Mean panels: orange ramp with white at low end.
cm_mean = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levels_mean); categorical = true)
highclip_mean = cm_mean[end]
cm_mean = cgrad(collect(cm_mean[1:(end - 1)]); categorical = true)
scale_mean = mk_piecewise_linear(levels_mean)
plot_levels_mean = scale_mean.(levels_mean)

# Diff panels: same ladder, mirrored around 0.
maxv_diff = maximum(abs, filter(isfinite, calV_diff))
diff_pos = pick_levels(maxv_diff; user_p)
levels_diff = Float32[-reverse(diff_pos[2:end]); 0; diff_pos[2:end]]
@info "Diff panel levels = $levels_diff  (data |max| ≈ $maxv_diff)"
cm_diff_full = cgrad(
    cgrad(:tol_bu_rd, length(levels_diff), categorical = true)[[1:(end ÷ 2 + 1); (end ÷ 2 + 1):end]],
    categorical = true,
)
highclip_diff = cm_diff_full[end]
lowclip_diff = cm_diff_full[1]
cm_diff = cgrad(collect(cm_diff_full[2:(end - 1)]); categorical = true)
scale_diff = mk_piecewise_linear(levels_diff)
plot_levels_diff = scale_diff.(levels_diff)

# Distinct colour for each time window's line in the zonal-integral panel.
tw1_color = :black
tw2_color = cgrad(:seaborn_colorblind, categorical = true)[2]

################################################################################
# Zonal integrals — bin (calV * Az) into 1° latitude bands.
#
# calV is in % v_tot / (10,000 km²) = % v_tot per 1e10 m². So for each 1° band
#   ∫_lon  calV[i,j] · Az[i,j]  /  (1e10 m²)   →   % v_tot per 1° band
# which (with 1° wide bins) equals "% v_tot / °lat".
################################################################################

const DLAT = 1.0
const LAT_BIN_EDGES = collect((-90.0):DLAT:90.0)
const LAT_BIN_CENTRES = LAT_BIN_EDGES[1:(end - 1)] .+ DLAT / 2

function zonal_integral(calV, Az, lat2D)
    Nb = length(LAT_BIN_CENTRES)
    z = zeros(Float64, Nb)
    @inbounds for j in axes(calV, 2), i in axes(calV, 1)
        v = calV[i, j]
        isfinite(v) || continue
        b = Int(floor((lat2D[i, j] + 90) / DLAT)) + 1
        (b < 1 || b > Nb) && continue
        z[b] += v * Az[i, j] / 1.0e10
    end
    return z
end

zint_1 = zonal_integral(calV1, Az_surf, lat2D)
zint_2 = zonal_integral(calV2, Az_surf, lat2D)
zint_diff = zonal_integral(calV_diff, Az_surf, lat2D)

@info @sprintf(
    "zonal_integral_1  total ≈ %+.3e %% v_tot  (sum × DLAT)",
    sum(zint_1) * DLAT,
)
@info @sprintf(
    "zonal_integral_2  total ≈ %+.3e %% v_tot",
    sum(zint_2) * DLAT,
)
@info @sprintf(
    "zonal_integral_dif total ≈ %+.3e %% v_tot",
    sum(zint_diff) * DLAT,
)

################################################################################
# Build the 2 × 2 figure
#
# Layout (4 rows × 2 cols):
#   row 1: [1,1] map           [1,2] zonal-integral panel
#   row 2: cb1 (horizontal, below the map; col 1 only)
#   row 3: [2,1] diff map      [2,2] zonal-integral-of-diff
#   row 4: cb2 (horizontal, below the diff map; col 1 only)
################################################################################

@info "Building figure"
flush(stdout); flush(stderr)

leg_label_long = TRAF ? "Adjoint 𝒱↓" : "Forward 𝒱↓"

fig = Figure(;
    size = (1500, 950), fontsize = 14,
    fonts = (; regular = "Arial", bold = "Arial Bold")
)

xticks_map = -0:90:1000
yticks_map = -90:30:90

# ----- [1, 1] map at TW1 ----------------------------------------------------
ax11 = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    backgroundcolor = :lightgray,
    xgridvisible = false, ygridvisible = false,
    xticks = (xticks_map, lonticklabel.(xticks_map)),
    yticks = (yticks_map, latticklabel.(yticks_map)),
)
co11 = plotmap!(
    ax11, calV1, gridmetrics;
    colorrange = extrema(plot_levels_mean),
    colormap = cm_mean,
    highclip = highclip_mean,
    colorscale = scale_mean,
    lon_window_start,
)
add_coastlines!(ax11)
ylims!(ax11, (-90, 90))
text!(
    ax11, 0, 1; text = rich("(a) ", parentmodel, " — ", TW1),
    align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
)
hidexdecorations!(ax11; ticks = false, grid = false, ticklabels = true, label = true)

# ----- colour bar for the mean panels (below [1, 1]) ------------------------
cb1 = Colorbar(
    fig[2, 1], co11;
    vertical = false, flipaxis = false,
    ticks = (plot_levels_mean, [isinteger(x) ? string(Int(x)) : string(x) for x in levels_mean]),
    label = rich("% v", subscript("tot"), " / (10,000 km", superscript("2"), ")"),
)
cb1.width = Relative(2 / 3)

# ----- [1, 2] zonal-integral side panel -------------------------------------
ax12 = Axis(
    fig[1, 2];
    xgridvisible = false, ygridvisible = false,
    yticks = (yticks_map, latticklabel.(yticks_map)),
    xlabel = rich("% v", subscript("tot"), " / °lat"),
)
xlims!(ax12, (0, nothing))
ylims!(ax12, (-90, 90))
linkyaxes!(ax12, ax11)
hideydecorations!(ax12; ticks = false, grid = false)
lin1 = lines!(ax12, zint_1, LAT_BIN_CENTRES; color = tw1_color, linewidth = 2)
lin2 = lines!(ax12, zint_2, LAT_BIN_CENTRES; color = tw2_color, linewidth = 2)
axislegend(
    ax12, [lin1, lin2], [TW1, TW2];
    position = :rb, framevisible = false, padding = (3, 3, 3, 3),
    margin = (3, 3, 3, 3), rowgap = 0, patchsize = (10, 10),
)
text!(
    ax12, 0, 1; text = "(b) zonal integral",
    align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
)

# ----- [2, 1] diff map ------------------------------------------------------
ax21 = Axis(
    fig[3, 1];
    aspect = DataAspect(),
    backgroundcolor = :lightgray,
    xgridvisible = false, ygridvisible = false,
    xticks = (xticks_map, lonticklabel.(xticks_map)),
    yticks = (yticks_map, latticklabel.(yticks_map)),
)
co21 = plotmap!(
    ax21, calV_diff, gridmetrics;
    colorrange = extrema(plot_levels_diff),
    colormap = cm_diff,
    highclip = highclip_diff,
    lowclip = lowclip_diff,
    colorscale = scale_diff,
    lon_window_start,
)
add_coastlines!(ax21)
ylims!(ax21, (-90, 90))
text!(
    ax21, 0, 1; text = rich("(c) ", TW2, " − ", TW1),
    align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
)

# ----- colour bar for the diff panels (below [2, 1]) ------------------------
cb2 = Colorbar(
    fig[4, 1], co21;
    vertical = false, flipaxis = false,
    ticks = (plot_levels_diff, divergingcbarticklabelformat(levels_diff)),
    label = rich("Δ % v", subscript("tot"), " / (10,000 km", superscript("2"), ")"),
)
cb2.width = Relative(2 / 3)

# ----- [2, 2] zonal-integral-of-diff with bicolour band ---------------------
ax22 = Axis(
    fig[3, 2];
    xgridvisible = false, ygridvisible = false,
    yticks = (yticks_map, latticklabel.(yticks_map)),
    xlabel = rich("Δ % v", subscript("tot"), " / °lat"),
    xtickformat = divergingcbarticklabelformat,
)
ylims!(ax22, (-90, 90))
linkyaxes!(ax22, ax21)
hideydecorations!(ax22; ticks = false, grid = false)

zero_line = zeros(length(zint_diff))
# Two single-sign bands meeting at zero. On 1° lat bins the rounding error
# at zero-crossings is at most one bin and not visible at this y-axis scale,
# so we skip the `dataforbicolorband` zero-crossing interpolation.
band!(
    ax22,
    Point2f.(min.(zint_diff, 0), LAT_BIN_CENTRES),
    Point2f.(zero_line, LAT_BIN_CENTRES);
    color = cm_diff[2],
)
band!(
    ax22,
    Point2f.(zero_line, LAT_BIN_CENTRES),
    Point2f.(max.(zint_diff, 0), LAT_BIN_CENTRES);
    color = cm_diff[end - 1],
)
lines!(ax22, zint_diff, LAT_BIN_CENTRES; color = :black, linewidth = 1.5)
vlines!(ax22, 0; color = :black, linewidth = 0.5)
text!(
    ax22, 0, 1; text = "(d) zonal integral of (c)",
    align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
)

# ----- Figure title ---------------------------------------------------------
Label(
    fig[0, 1:2];
    text = rich(
        "Surface ventilation ", leg_label_long, " — ", parentmodel,
        " (", model_config, ")"
    ),
    fontsize = 18, tellwidth = false,
)

# ----- Spacing tweaks -------------------------------------------------------
rowgap!(fig.layout, 5)
colgap!(fig.layout, 10)
colsize!(fig.layout, 1, Aspect(1, 2.0))
colsize!(fig.layout, 2, Auto(0.3))
resize_to_layout!(fig)

################################################################################
# Save
################################################################################

outputfile = joinpath(plot_dir, "calVdown_$(leg_tag).png")
@info "Saving $outputfile"
flush(stdout); flush(stderr)
save(outputfile, fig)

@info "plot_ventilation.jl complete"
flush(stdout); flush(stderr)
