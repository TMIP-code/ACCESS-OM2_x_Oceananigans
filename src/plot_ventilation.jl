"""
Plot the surface ventilation diagnostic 𝒱 produced by
`compute_ventilation_diagnostic.jl`, in the Pasquier *et al.* 2024 / 2025
plotting style: per-cell quad mesh on the tripolar grid (via the helpers in
[src/shared_utils/plotting_functions.jl](shared_utils/plotting_functions.jl))
on a pseudo-log colour scale with levels `[0, 10, 30, 100, 300, 1000]` in
units `% v_tot / (10,000 km)²`.

The time windows come from the environment: `TW1` (default `TIME_WINDOW`) and,
optionally, `TW2`. With `TW2` set, it lays out a 2 × 2 panel:

  [1, 1] map of `calV` at TW1 (orange ramp, pseudo-log scale)
  [1, 2] zonal-integral side panel — both windows overlaid as lines
  [2, 1] decadal-difference map (TW2 − TW1) — diverging
  [2, 2] zonal-integral of the difference (bicolour band)

With `TW2` unset, only row 1 (TW1 map + zonal integral) is drawn.

It loads
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK[_QAxB]/ventilation.jld2
for each window (mirrors the same dual-naming fallback as
`compute_ventilation_diagnostic.jl`), normalises with `1e16 / vtot`, and
writes one PNG per (PM, leg) to
  outputs/{PM}/{EXP}/plots/{MC}/calVup_forward.png            (TW1 + TW2)
  outputs/{PM}/{EXP}/plots/{MC}_traf/calVdown_adjoint.png     (TW1 + TW2)
  … or `{calV}_{leg}_{TW1}.png` with TW2 unset (TW1 only)

Leg ↔ arrow (see `ventilation_leg` in shared_utils/config.jl): the forward
leg (age Γ↓) gives 𝒱↑ = `calVup` (volume re-exposed at the surface); the
`_traf` leg (adjoint age Γ↑) gives 𝒱↓ = `calVdown`, the ventilation diagnostic
of Pasquier *et al.* 2024. The `_traf` suffix is appended to `MODEL_CONFIG` by
`env_defaults.sh` when `TRAF=yes`.

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/plot_ventilation.jl
```

Required env vars: PARENT_MODEL, EXPERIMENT (auto from PM if unset),
VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER.
Optional: LUMP_AND_SPRAY, TRAF, PARTITION_X, PARTITION_Y, TW1 (default
TIME_WINDOW), TW2 (default unset → single-window figure). E.g. the original
cross-decade figure is `TW1=1968-1977 TW2=1999-2008`.
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

model_config = require_env("MODEL_CONFIG")

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
(; leg_tag, calV_tag, leg_label_long) = ventilation_leg(TRAF)

ls = parse_lump_and_spray()

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

# Time windows from ENV (were hard-coded 1968-1977 / 1999-2008 until after
# commit a93b4df). TW2 unset → single-window figure (no difference row).
const TW1 = get(ENV, "TW1", require_env("TIME_WINDOW"))
const TW2 = get(ENV, "TW2", "")
const two_windows = !isempty(TW2)

# Experiment-level outputs root (parent of the per-TW dirs). outputdir from
# load_project_config is per-TIME_WINDOW, so the experiment-level dir is one
# above it.
exp_outdir = dirname(outputdir)

omega = parse_omega()
omega_suffix = omega.suffix
vent_basename = "ventilation$(omega_suffix).jld2"

function ventilation_path(tw)
    tw_root = joinpath(exp_outdir, tw)
    # Search both the gpu_tag root and the bare gpu_tag-less root (some
    # partitioned runs wrote to `periodic/{MC}` without the `{gpu_tag}` part).
    periodic_roots = unique(
        [
            isempty(gpu_tag) ?
                joinpath(tw_root, "periodic", model_config) :
                joinpath(tw_root, "periodic", model_config, gpu_tag),
            joinpath(tw_root, "periodic", model_config),
        ]
    )
    candidate_dirs = unique(
        [
            joinpath(root, sub)
                for root in periodic_roots
                for sub in ("NK$(ls.dir_suffix)", "NK")
        ]
    )
    for d in candidate_dirs
        f = joinpath(d, vent_basename)
        isfile(f) && return f
    end
    error(
        "$vent_basename not found for TIME_WINDOW=$tw. Tried:\n" *
            join(["  " * joinpath(d, vent_basename) for d in candidate_dirs], "\n") *
            "\nRun compute_ventilation_diagnostic.jl for TIME_WINDOW=$tw first " *
            "(or `bash scripts/driver.sh JOB_CHAIN=ventilation TIME_WINDOW=$tw …`).",
    )
end

vent_file_1 = ventilation_path(TW1)
vent_file_2 = two_windows ? ventilation_path(TW2) : nothing

plot_dir = joinpath(exp_outdir, "plots", model_config)
mkpath(plot_dir)

@info "plot_ventilation.jl configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- EXPERIMENT    = $experiment"
@info "- model_config  = $model_config"
@info "- leg           = $leg_tag"
@info "- TW1 input     = $vent_file_1  ($TW1)"
two_windows && @info "- TW2 input     = $vent_file_2  ($TW2)"
@info "- output dir    = $plot_dir"
flush(stdout); flush(stderr)

################################################################################
# Load data and grid
################################################################################

@info "Loading $vent_file_1"
flush(stdout); flush(stderr)
d1 = load(vent_file_1)

calVdown_raw_1 = d1["calVdown_raw"]   # m³/m² (= m), NaN at dry cells
vtot = d1["vtot"]                     # m³ — use TW1's; assert TW2 matches
Az_surf = d1["Az_surf"]               # m²

# Normalise: % v_tot / (10,000 km)² — note the (10,000 km)² = 1e14 m², not 1e10.
# Prefactor: 100 × (1e14 m² / (10,000 km)²) / vtot[m³] = 1e16 / vtot.
norm_factor = 1.0e16 / vtot
calV1 = calVdown_raw_1 .* norm_factor
stats_fields = [("calV1", calV1)]

if two_windows
    @info "Loading $vent_file_2"
    flush(stdout); flush(stderr)
    d2 = load(vent_file_2)
    calVdown_raw_2 = d2["calVdown_raw"]
    size(calVdown_raw_1) == size(calVdown_raw_2) ||
        error("Shape mismatch between TW1 ($(size(calVdown_raw_1))) and TW2 ($(size(calVdown_raw_2)))")
    let δ = abs(d2["vtot"] - vtot) / vtot
        δ < 1.0e-6 || @warn "v_tot differs between time windows" tw1_vtot = vtot tw2_vtot = d2["vtot"] reldiff = δ
    end
    calV2 = calVdown_raw_2 .* norm_factor
    calV_diff = calV2 .- calV1
    push!(stats_fields, ("calV2", calV2), ("calV_diff", calV_diff))
end

@info @sprintf("v_tot = %.3e m³;  1e16/v_tot = %.3e", vtot, norm_factor)
for (name, v) in stats_fields
    vals = filter(isfinite, v)
    @info @sprintf(
        "%s [%% v_tot / (10,000 km)²]:  min = %+.3e   mean = %+.3e   max = %+.3e",
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
Nx, Ny = size(calV1)

# Build the (4, Nx, Ny) curvilinear gridmetrics expected by plotmap! and the
# GeoMakie coastline overlay via the shared helpers in plotting_functions.jl.
gridmetrics = gridmetrics_from_grid(grid, Nx, Ny)
lat2D = gridmetrics.lat   # interior cell-centre latitudes, used by zonal_integral

lon_window_start = 20  # matches plotmap!'s default

################################################################################
# Colour scales and palettes
################################################################################

# Pasquier 2024 spec is [0, 10, 30, 100, 300, 1000] %v_tot/(10,000 km)². Pick a
# log-decade ladder of the same shape sized to the actual data max, so the same
# script works across resolutions / sub-basins where v_tot can change by 1-2
# orders of magnitude. Override the lowest-non-zero level via VENT_LEVELS_P,
# e.g. VENT_LEVELS_P=10 to force the literal Pasquier set.
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
maxv_mean = maximum(filter(isfinite, calV1))
two_windows && (maxv_mean = max(maxv_mean, maximum(filter(isfinite, calV2))))
levels_mean = pick_levels(maxv_mean; user_p)
@info "Mean panel levels = $levels_mean  (data max ≈ $maxv_mean)"

# Mean panels: orange ramp with white at low end.
cm_mean = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levels_mean); categorical = true)
highclip_mean = cm_mean[end]
cm_mean = cgrad(collect(cm_mean[1:(end - 1)]); categorical = true)
scale_mean = mk_piecewise_linear(levels_mean)
plot_levels_mean = scale_mean.(levels_mean)

# Diff panels: same ladder, mirrored around 0. No explicit 0 in levels — the
# middle interval [-p, p] is rendered as one white band centered on zero
# (Pasquier 2024 convention).
if two_windows
    maxv_diff = maximum(abs, filter(isfinite, calV_diff))
    diff_pos = pick_levels(maxv_diff; user_p)
    levels_diff = Float32[-reverse(diff_pos[2:end]); diff_pos[2:end]]
    @info "Diff panel levels = $levels_diff  (data |max| ≈ $maxv_diff)"
    # PRGn with white center, Pasquier 2024 convention.
    cm_diff_full = cgrad(withwhitecenter(Makie.ColorSchemes.PRGn), length(levels_diff) + 1; categorical = true)
    lowclip_diff = cm_diff_full[1]
    highclip_diff = cm_diff_full[end]
    cm_diff = cgrad(collect(cm_diff_full[2:(end - 1)]); categorical = true)
    scale_diff = mk_piecewise_linear(levels_diff)
    plot_levels_diff = scale_diff.(levels_diff)
end

# Distinct colour for each time window's line in the zonal-integral panel.
tw1_color = :black
tw2_color = cgrad(:seaborn_colorblind, categorical = true)[2]

################################################################################
# Zonal integrals — bin (calV * Az) into 1° latitude bands.
#
# calV is in % v_tot / (10,000 km)² = % v_tot per 1e14 m². So for each 1° band
#   ∫_lon  calV[i,j] · Az[i,j]  /  (1e14 m²)   →   % v_tot per 1° band
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
        z[b] += v * Az[i, j] / 1.0e14
    end
    return z
end

zint_1 = zonal_integral(calV1, Az_surf, lat2D)
@info @sprintf(
    "zonal_integral_1  total ≈ %+.3e %% v_tot  (sum × DLAT)",
    sum(zint_1) * DLAT,
)
if two_windows
    zint_2 = zonal_integral(calV2, Az_surf, lat2D)
    zint_diff = zonal_integral(calV_diff, Az_surf, lat2D)
    @info @sprintf(
        "zonal_integral_2  total ≈ %+.3e %% v_tot",
        sum(zint_2) * DLAT,
    )
    @info @sprintf(
        "zonal_integral_dif total ≈ %+.3e %% v_tot",
        sum(zint_diff) * DLAT,
    )
end

################################################################################
# Build the 2 × 2 figure
#
# Layout (3 rows × 3 cols, plus a title row 0):
#   row 0: title spanning cols 1-3
#   row 1: cb1 (vertical, ticks left) | (a) map        | (b) zonal-integral
#   row 2: cb2 (vertical, ticks left) | (c) diff map   | (d) zonal-integral of diff
#
# Colorbars are built manually (not inherited from the plot) so each colour
# bin occupies equal visual width, with ticks at the level boundaries
# (0, 10, 30, 100, 300, 1000 for the mean panel; the data-aware ladder when
# the values differ from the Pasquier 2024 spec). Map colors come from the
# same `cm_mean`/`cm_diff` cgrad, so position k in the colorbar shows
# exactly the colour applied to the k-th data bin in the map.
################################################################################

@info "Building figure"
flush(stdout); flush(stderr)


fig = Figure(;
    size = (1500, 1000), fontsize = 14,
    fonts = (; regular = "Arial", bold = "Arial Bold")
)

xticks_map = -0:90:1000
yticks_map = -90:30:90

mean_label = rich("% v", subscript("tot"), " / (10,000 km)", superscript("2"))
diff_label = rich("Δ % v", subscript("tot"), " / (10,000 km)", superscript("2"))

# ----- col 1, row 1: vertical colorbar for the mean panels ------------------
N_mean = length(levels_mean) - 1   # 5 colour bins
cb1 = Colorbar(
    fig[1, 1];
    colormap = cm_mean,
    colorrange = (0, N_mean),
    highclip = highclip_mean,
    ticks = (0:N_mean, [isinteger(x) ? string(Int(x)) : string(x) for x in levels_mean]),
    label = mean_label,
    vertical = true, flipaxis = false,
)
cb1.height = Relative(0.8)

# ----- col 2, row 1: (a) map at TW1 -----------------------------------------
ax11 = Axis(
    fig[1, 2];
    # aspect = DataAspect(),
    backgroundcolor = :lightgray,
    xgridvisible = false, ygridvisible = false,
    xticks = (xticks_map, lonticklabel.(xticks_map)),
    yticks = (yticks_map, latticklabel.(yticks_map)),
)
co11 = plotmap!(
    ax11, calV1, gridmetrics;
    colorrange = extrema(levels_mean),     # data space; Makie applies colorscale itself
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
# Hide the top map's x tick labels only when the diff row sits below it.
two_windows && hidexdecorations!(ax11; ticks = false, grid = false, ticklabels = true, label = true)

# ----- col 3, row 1: (b) zonal-integral side panel --------------------------
ax12 = Axis(
    fig[1, 3];
    xgridvisible = false, ygridvisible = false,
    yticks = (yticks_map, latticklabel.(yticks_map)),
    xlabel = rich("% v", subscript("tot"), " / °lat"),
)
xlims!(ax12, (0, nothing))
ylims!(ax12, (-90, 90))
linkyaxes!(ax12, ax11)
hideydecorations!(ax12; ticks = false, grid = false)
lin1 = lines!(ax12, zint_1, LAT_BIN_CENTRES; color = tw1_color, linewidth = 2)
if two_windows
    lin2 = lines!(ax12, zint_2, LAT_BIN_CENTRES; color = tw2_color, linewidth = 2)
    axislegend(
        ax12, [lin1, lin2], [TW1, TW2];
        position = :rb, framevisible = false, padding = (3, 3, 3, 3),
        margin = (3, 3, 3, 3), rowgap = 0, patchsize = (10, 10),
    )
end
text!(
    ax12, 0, 1; text = "(b) zonal integral",
    align = (:left, :top), space = :relative, offset = (5, -5), font = :bold,
)

if two_windows

    # ----- col 1, row 2: vertical colorbar for the diff panels ------------------
    N_diff = length(levels_diff) - 1
    cb2 = Colorbar(
        fig[2, 1];
        colormap = cm_diff,
        colorrange = (0, N_diff),
        lowclip = lowclip_diff,
        highclip = highclip_diff,
        ticks = (0:N_diff, divergingcbarticklabelformat(levels_diff)),
        label = diff_label,
        vertical = true, flipaxis = false,
    )
    cb2.height = Relative(0.8)

    # ----- col 2, row 2: (c) diff map -------------------------------------------
    ax21 = Axis(
        fig[2, 2];
        # aspect = DataAspect(),
        backgroundcolor = :lightgray,
        xgridvisible = false, ygridvisible = false,
        xticks = (xticks_map, lonticklabel.(xticks_map)),
        yticks = (yticks_map, latticklabel.(yticks_map)),
    )
    co21 = plotmap!(
        ax21, calV_diff, gridmetrics;
        colorrange = extrema(levels_diff),     # data space; Makie applies colorscale itself
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

    # ----- col 3, row 2: (d) zonal-integral of diff -----------------------------
    ax22 = Axis(
        fig[2, 3];
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

end # two_windows

# ----- Figure title ---------------------------------------------------------
Label(
    fig[0, 1:3];
    text = rich(
        "Surface ventilation ", leg_label_long, " — ", parentmodel,
        two_windows ? "" : " $(experiment)",
        " (", model_config, ")"
    ),
    fontsize = 18, tellwidth = false,
)

# ----- Spacing tweaks -------------------------------------------------------
# Col 1 = narrow vertical colorbar; col 2 = map; col 3 = zonal-integral panel.
# Pasquier 2024 template uses Auto(0.28) on the zonal-integral column → matched
# on col 3.
# Close the col 2 / col 3 gap (zonal shares the map y-axis via linkyaxes!).
# rowgap!(fig.layout, 5)
# colgap!(fig.layout, 1, 5)        # between cb (col 1) and map (col 2)
# colgap!(fig.layout, 2, 0)        # between map (col 2) and zonal (col 3)
colsize!(fig.layout, 3, Auto(0.28))
resize_to_layout!(fig)

################################################################################
# Save
################################################################################

tw_suffix = two_windows ? "" : "_$(TW1)"
outputfile = joinpath(plot_dir, "$(calV_tag)_$(leg_tag)$(tw_suffix)$(omega_suffix).png")
@info "Saving $outputfile"
flush(stdout); flush(stderr)
save(outputfile, fig)

@info "plot_ventilation.jl complete"
flush(stdout); flush(stderr)
