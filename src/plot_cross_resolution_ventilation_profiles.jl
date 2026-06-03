"""
Cross-resolution + cross-decade comparison of the **zonal-integral ventilation
profile** (% v_tot / °lat vs latitude) — the side-panel curve of
plot_ventilation.jl, gathered into a single corner-plot-style figure.

The zonal integral bins `calVdown · A_surf / 1e14` into 1° latitude bands on a
fixed −90…90 axis, so OM2-1 and OM2-025 land on the *same* latitude grid and are
directly comparable (no regridding needed for the profiles).

Layout (the main panel occupies a 2×2 top-left block; 2 difference panels below,
2 on the right — like a corner plot of the 2×2 [resolution × decade] case grid):

    ┌───────────────────────┬───────────────┐
    │  MAIN: 4 curves        │ Δ res @ TW1   │   (across resolution, col diff)
    │  (OM2-1/025 × TW1/2)   ├───────────────┤
    │                        │ Δ res @ TW2   │
    ├───────────┬───────────┼───────────────┤
    │ Δ dec OM2-1│Δ dec OM2-025│  (legend)    │
    └───────────┴───────────┴───────────────┘
       (across decade, row diff)

All panels share the latitude x-axis. Diff panels draw a bicolour band
(red positive, blue negative) about zero.

Reads `ventilation.jld2` (calVdown_raw, vtot, Az_surf) for both resolutions and
both time windows. Standalone CPU script; env vars mirror
plot_cross_resolution_ventilation.jl.

Usage:
```
qsub scripts/plotting/plot_cross_resolution_ventilation_profiles.sh
```
Writes outputs/cross_resolution/ventilation/calVdown_profiles_{forward,adjoint}.png
"""

@info "Loading packages for cross-resolution ventilation-profile plot"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using CairoMakie
using JLD2
using Printf
using Statistics

include("shared_functions.jl")
include(joinpath(@__DIR__, "shared_utils", "plotting_functions.jl"))

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
config_suffix = TRAF ? "_traf" : ""
leg_long = TRAF ? "Adjoint 𝒱↓" : "Forward 𝒱↓"
leg_tag = TRAF ? "adjoint" : "forward"

mc_om21 = get(ENV, "MODEL_CONFIG_OM21", "totaltransport_wparent_centered2_AB2_kH300_kVML1e-1_kVBG3e-5_mkappaV_DTx4") * config_suffix
mc_om2025 = get(ENV, "MODEL_CONFIG_OM2025", "totaltransport_wparent_centered2_AB2_kH75_kVML5e-2_kVBG15e-6_mkappaV_LBS_DTx2") * config_suffix
vent_subdir = get(ENV, "VENT_SUBDIR", "NK_Q2x2")

TW1 = get(ENV, "TW1", "1968-1977")
TW2 = get(ENV, "TW2", "1999-2008")

models = [
    ("OM2-1", "ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6", mc_om21),
    ("OM2-025", "ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6", mc_om2025),
]

repo_root = normpath(joinpath(@__DIR__, ".."))
grid_path(pm, exp) = joinpath(repo_root, "preprocessed_inputs", pm, exp, "grid.jld2")

function vent_path(pm, exp, mc, tw)
    base = joinpath(repo_root, "outputs", pm, exp, tw, "periodic", mc)
    for sub in unique([vent_subdir, "NK_Q2x2", "NK"])
        f = joinpath(base, sub, "ventilation.jld2")
        isfile(f) && return f
    end
    error("ventilation.jld2 not found for $pm $tw under $base/{$(vent_subdir),NK_Q2x2,NK}")
end

@info "plot_cross_resolution_ventilation_profiles.jl configuration"
@info "- leg = $leg_tag; OM2-1 = $mc_om21; OM2-025 = $mc_om2025; TWs = $TW1,$TW2"
flush(stdout); flush(stderr)

################################################################################
# Zonal integral (mirrors plot_ventilation.jl)
################################################################################

const DLAT = 1.0
const LAT_EDGES = collect((-90.0):DLAT:90.0)
const LAT_CENTRES = LAT_EDGES[1:(end - 1)] .+ DLAT / 2

function zonal_integral(calV, Az, lat2D)
    z = zeros(Float64, length(LAT_CENTRES))
    @inbounds for j in axes(calV, 2), i in axes(calV, 1)
        v = calV[i, j]
        isfinite(v) || continue
        b = Int(floor((lat2D[i, j] + 90) / DLAT)) + 1
        (b < 1 || b > length(z)) && continue
        z[b] += v * Az[i, j] / 1.0e14
    end
    return z
end

function lat2D_of(grid, Nx, Ny)
    ug = grid isa Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid ? grid.underlying_grid : grid
    return Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])
end

function load_zint(pm, exp, mc, tw, grid)
    d = load(vent_path(pm, exp, mc, tw))
    calV = d["calVdown_raw"] .* (1.0e16 / d["vtot"])
    Az = d["Az_surf"]
    Nx, Ny = size(calV)
    return zonal_integral(calV, Az, lat2D_of(grid, Nx, Ny))
end

################################################################################
# Load + integrate
################################################################################

(tag1, pm1, exp1, mc1) = models[1]
(tag2, pm2, exp2, mc2) = models[2]

@info "Loading grids + integrating"
flush(stdout); flush(stderr)
grid1 = load_tripolar_grid(grid_path(pm1, exp1), CPU())
grid2 = load_tripolar_grid(grid_path(pm2, exp2), CPU())

z1_tw1 = load_zint(pm1, exp1, mc1, TW1, grid1)
z1_tw2 = load_zint(pm1, exp1, mc1, TW2, grid1)
z2_tw1 = load_zint(pm2, exp2, mc2, TW1, grid2)
z2_tw2 = load_zint(pm2, exp2, mc2, TW2, grid2)

# Differences on the common latitude axis.
dres_tw1 = z2_tw1 .- z1_tw1     # OM2-025 − OM2-1, TW1
dres_tw2 = z2_tw2 .- z1_tw2     # OM2-025 − OM2-1, TW2
ddec_om1 = z1_tw2 .- z1_tw1     # TW2 − TW1, OM2-1
ddec_om2 = z2_tw2 .- z2_tw1     # TW2 − TW1, OM2-025

for (nm, z) in (
        ("OM2-1 $TW1", z1_tw1), ("OM2-1 $TW2", z1_tw2),
        ("OM2-025 $TW1", z2_tw1), ("OM2-025 $TW2", z2_tw2),
    )
    @info @sprintf("  ∫ %s = %.3f %% v_tot", nm, sum(z) * DLAT)
end
flush(stdout); flush(stderr)

################################################################################
# Figure
################################################################################

@info "Building figure"
flush(stdout); flush(stderr)

col_om1 = :black
col_om025 = cgrad(:seaborn_colorblind, categorical = true)[2]
band_pos = cgrad(:balance, categorical = true)[end]   # red-ish
band_neg = cgrad(:balance, categorical = true)[1]     # blue-ish

yticks_lat = -90:30:90
lat_axis_kwargs = (
    yticks = (collect(yticks_lat), latticklabel.(yticks_lat)),
    ygridvisible = true, xgridvisible = true, limits = (nothing, nothing, -90, 90),
)

fig = Figure(; size = (1200, 1000), fontsize = 15)
g = fig[1, 1] = GridLayout()

Label(
    g[0, 1:3]; text = "Surface ventilation $leg_long — zonal-integral profiles (% v_tot / °lat)",
    fontsize = 19, font = :bold, tellwidth = false,
)

# ── Main panel: 4 overlaid curves (value on x, latitude on y) ──
ax_main = Axis(
    g[1:2, 1:2]; xlabel = rich("% v", subscript("tot"), " / °lat"), ylabel = "Latitude",
    lat_axis_kwargs...,
)
l1 = lines!(ax_main, z1_tw1, LAT_CENTRES; color = col_om1, linewidth = 2)
l2 = lines!(ax_main, z1_tw2, LAT_CENTRES; color = col_om1, linewidth = 2, linestyle = :dash)
l3 = lines!(ax_main, z2_tw1, LAT_CENTRES; color = col_om025, linewidth = 2)
l4 = lines!(ax_main, z2_tw2, LAT_CENTRES; color = col_om025, linewidth = 2, linestyle = :dash)
axislegend(
    ax_main,
    [l1, l2, l3, l4],
    ["$tag1 $TW1", "$tag1 $TW2", "$tag2 $TW1", "$tag2 $TW2"];
    position = :rb, framevisible = true, labelsize = 11,
)

# Bicolour-band diff panel helper (latitude on y, Δ value on x).
function diffpanel!(pos, dz, title; ylabel = "")
    ax = Axis(pos; xlabel = rich("Δ % v", subscript("tot"), " / °lat"), ylabel = ylabel, lat_axis_kwargs...)
    zero_line = zeros(length(dz))
    band!(ax, Point2f.(min.(dz, 0), LAT_CENTRES), Point2f.(zero_line, LAT_CENTRES); color = band_neg)
    band!(ax, Point2f.(zero_line, LAT_CENTRES), Point2f.(max.(dz, 0), LAT_CENTRES); color = band_pos)
    lines!(ax, dz, LAT_CENTRES; color = :black, linewidth = 1.2)
    vlines!(ax, 0; color = :black, linewidth = 0.5)
    text!(
        ax, 0, 1; text = title, align = (:left, :top), space = :relative,
        offset = (4, -4), font = :bold, fontsize = 12,
    )
    return ax
end

# ── Right column: Δ resolution (across columns of the case grid) ──
ax_r1 = diffpanel!(g[1, 3], dres_tw1, "Δ res, $TW1")
ax_r2 = diffpanel!(g[2, 3], dres_tw2, "Δ res, $TW2")

# ── Bottom row: Δ decade (across rows of the case grid) ──
ax_b1 = diffpanel!(g[3, 1], ddec_om1, "Δ decade, $tag1"; ylabel = "Latitude")
ax_b2 = diffpanel!(g[3, 2], ddec_om2, "Δ decade, $tag2")

# Corner cell: short caption.
Label(
    g[3, 3];
    text = "rows: Δ resolution\n(025 − 1)\ncols: Δ decade\n($TW2 − $TW1)",
    fontsize = 11, tellwidth = false, tellheight = false, justification = :left,
)

linkyaxes!(ax_main, ax_r1, ax_r2, ax_b1, ax_b2)
hideydecorations!(ax_r1; ticks = false, grid = false)
hideydecorations!(ax_r2; ticks = false, grid = false)
hideydecorations!(ax_b2; ticks = false, grid = false)

################################################################################
# Save
################################################################################

outdir = joinpath(repo_root, "outputs", "cross_resolution", "ventilation")
mkpath(outdir)
outfile = joinpath(outdir, "calVdown_profiles_$(leg_tag).png")
@info "Saving $outfile"
flush(stdout); flush(stderr)
save(outfile, fig; px_per_unit = 2)

@info "plot_cross_resolution_ventilation_profiles.jl complete — saved $outfile"
flush(stdout); flush(stderr)
