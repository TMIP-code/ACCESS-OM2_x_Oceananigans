"""
Cross-resolution + cross-decade comparison of the surface ventilation map
`calVdown` (Pasquier *et al.* 2024) — a 3×3 panel mirroring
plot_cross_resolution_age_slice.jl but for the ventilation maps, **without** the
zonal-integral side panels.

Layout (3 rows × 3 cols of maps; the [3,3] corner is a GridLayout holding the
two shared colorbars):

                 OM2-1        OM2-025      Δ resolution (025 − 1)
  1968–1977     [1,1] 𝒱      [1,2] 𝒱      [1,3] diff
  1999–2008     [2,1] 𝒱      [2,2] 𝒱      [2,3] diff
  Δ decade      [3,1] diff   [3,2] diff   [colorbars]

- 𝒱 panels (cols 1–2, rows 1–2): calVdown normalised to % v_tot / (10,000 km)²
  on the Pasquier-2024 orange pseudo-log scale (white at the low end).
- Δ-resolution panels (col 3): OM2-025 − OM2-1, with the OM2-1 surface field
  conservatively regridded onto the OM2-025 grid via ConservativeRegridding.jl
  (masked-conservative; see plot_cross_resolution_age_slice.jl).
- Δ-decade panels (row 3): (1999–2008) − (1968–1977) on each native grid.
- Both diff families share a white-centred PRGn diverging pseudo-log scale.

Reads `ventilation.jld2` (calVdown_raw, vtot, Az_surf) for both resolutions and
both time windows. This is a standalone CPU script; env vars mirror
plot_cross_resolution_age_slice.jl (MODEL_CONFIG_OM21/OM2025/TW1/TW2/TRAF) plus
VENT_SUBDIR (default NK_Q2x2) and VENT_LEVELS_P (override the lowest non-zero
ventilation level).

Usage:
```
qsub scripts/plotting/plot_cross_resolution_ventilation.sh
```
Writes outputs/cross_resolution/ventilation/calVdown_{forward,adjoint}_3x3.png
"""

@info "Loading packages for cross-resolution ventilation-map plot"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Fields: CenterField
using CairoMakie
using GeoMakie
using GeometryBasics
using ConservativeRegridding
using JLD2
using Printf
using Statistics

const CR = ConservativeRegridding

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

user_p = (haskey(ENV, "VENT_LEVELS_P") && !isempty(ENV["VENT_LEVELS_P"])) ?
    parse(Float64, ENV["VENT_LEVELS_P"]) : nothing

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

@info "plot_cross_resolution_ventilation.jl configuration"
@info "- leg            = $leg_tag"
@info "- OM2-1 config   = $mc_om21"
@info "- OM2-025 config = $mc_om2025"
@info "- vent subdir    = $vent_subdir"
@info "- time windows   = $TW1, $TW2"
flush(stdout); flush(stderr)

################################################################################
# Helpers (shared with plot_cross_resolution_age_slice.jl)
################################################################################

function gridmetrics_from(grid, Nx, Ny)
    ug = grid isa Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lon2D = Array(ug.λᶜᶜᵃ[1:Nx, 1:Ny])
    lat2D = Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])
    λff = Array(ug.λᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)])
    φff = Array(ug.φᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)])
    lon_vertices = Array{eltype(λff)}(undef, 4, Nx, Ny)
    lat_vertices = Array{eltype(φff)}(undef, 4, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        lon_vertices[1, i, j] = λff[i, j]
        lon_vertices[2, i, j] = λff[i + 1, j]
        lon_vertices[3, i, j] = λff[i + 1, j + 1]
        lon_vertices[4, i, j] = λff[i, j + 1]
        lat_vertices[1, i, j] = φff[i, j]
        lat_vertices[2, i, j] = φff[i + 1, j]
        lat_vertices[3, i, j] = φff[i + 1, j + 1]
        lat_vertices[4, i, j] = φff[i, j + 1]
    end
    return (; lon = lon2D, lat = lat2D, lon_vertices, lat_vertices)
end

"""Masked-conservative regrid of 2-D `src` (NaN at dry) onto `R`'s dst grid."""
function regrid_field(R, src, dst_wet)
    nd = length(R.dst_areas)
    f = vec([isfinite(src[i]) ? Float64(src[i]) : 0.0 for i in eachindex(src)])
    w = vec([isfinite(src[i]) ? 1.0 : 0.0 for i in eachindex(src)])
    num = zeros(Float64, nd)
    den = zeros(Float64, nd)
    CR.regrid!(num, R, f)
    CR.regrid!(den, R, w)
    Nx, Ny = size(dst_wet)
    out = reshape([den[i] > 1.0e-8 ? num[i] / den[i] : NaN for i in eachindex(num)], Nx, Ny)
    @inbounds for i in eachindex(out)
        dst_wet[i] || (out[i] = NaN)
    end
    return out
end

# Pasquier-2024 level ladder (cf. plot_ventilation.jl)
function pick_levels(maxv; user_p = nothing)
    p = user_p !== nothing ? user_p : 10.0^floor(Int, log10(maxv / 30))
    return Float32[0, p, 3p, 10p, 30p, 100p]
end

logmag(name, a) = (
    v = filter(isfinite, vec(a)); isempty(v) ?
        (@info "  $name: all-NaN") :
        (@info @sprintf("  %s: min=%+.3e mean=%+.3e max=%+.3e", name, minimum(v), mean(v), maximum(v)))
)

################################################################################
# Load data + grids
################################################################################

(tag1, pm1, exp1, mc1) = models[1]
(tag2, pm2, exp2, mc2) = models[2]

@info "Loading grids"
flush(stdout); flush(stderr)
grid1 = load_tripolar_grid(grid_path(pm1, exp1), CPU())
grid2 = load_tripolar_grid(grid_path(pm2, exp2), CPU())

# calVdown normalised to % v_tot / (10,000 km)² (= 1e16 / vtot prefactor)
function load_calV(pm, exp, mc, tw)
    d = load(vent_path(pm, exp, mc, tw))
    return d["calVdown_raw"] .* (1.0e16 / d["vtot"])
end

@info "Loading ventilation fields"
flush(stdout); flush(stderr)
v1_tw1 = load_calV(pm1, exp1, mc1, TW1)
v1_tw2 = load_calV(pm1, exp1, mc1, TW2)
v2_tw1 = load_calV(pm2, exp2, mc2, TW1)
v2_tw2 = load_calV(pm2, exp2, mc2, TW2)

Nx1, Ny1 = size(v1_tw1)
Nx2, Ny2 = size(v2_tw1)
wet1 = isfinite.(v1_tw1)
wet2 = isfinite.(v2_tw1)

################################################################################
# Regrid OM2-1 → OM2-025 and build differences
################################################################################

@info "Building conservative regridder OM2-1 → OM2-025"
flush(stdout); flush(stderr)
R = CR.Regridder(CenterField(grid2.underlying_grid), CenterField(grid1.underlying_grid); progress = true)

v1_tw1_on2 = regrid_field(R, v1_tw1, wet2)
v1_tw2_on2 = regrid_field(R, v1_tw2, wet2)

rdiff_tw1 = v2_tw1 .- v1_tw1_on2
rdiff_tw2 = v2_tw2 .- v1_tw2_on2
ddiff_om1 = v1_tw2 .- v1_tw1
ddiff_om2 = v2_tw2 .- v2_tw1

@info "Magnitude summary (% v_tot / (10,000 km)²)"
logmag("OM2-1   $TW1", v1_tw1); logmag("OM2-1   $TW2", v1_tw2)
logmag("OM2-025 $TW1", v2_tw1); logmag("OM2-025 $TW2", v2_tw2)
logmag("Δres    $TW1", rdiff_tw1); logmag("Δres    $TW2", rdiff_tw2)
logmag("Δdecade OM2-1", ddiff_om1); logmag("Δdecade OM2-025", ddiff_om2)
flush(stdout); flush(stderr)

################################################################################
# Grid metrics + colour scales
################################################################################

gm1 = gridmetrics_from(grid1, Nx1, Ny1)
gm2 = gridmetrics_from(grid2, Nx2, Ny2)

# Absolute (𝒱) panels: orange ramp, white low end.
maxv = maximum(
    maximum(filter(isfinite, x)) for x in (v1_tw1, v1_tw2, v2_tw1, v2_tw2)
)
levels_mean = pick_levels(maxv; user_p)
cm_mean = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levels_mean); categorical = true)
highclip_mean = cm_mean[end]
cm_mean = cgrad(collect(cm_mean[1:(end - 1)]); categorical = true)
scale_mean = mk_piecewise_linear(levels_mean)
N_mean = length(levels_mean) - 1

# Diff panels: PRGn, white centre, mirrored zero-excluding ladder.
maxd = maximum(
    maximum(abs, filter(isfinite, x)) for x in (rdiff_tw1, rdiff_tw2, ddiff_om1, ddiff_om2)
)
diff_pos = pick_levels(maxd; user_p)
levels_diff = Float32[-reverse(diff_pos[2:end]); diff_pos[2:end]]
cm_diff_full = cgrad(withwhitecenter(Makie.ColorSchemes.PRGn), length(levels_diff) + 1; categorical = true)
lowclip_diff = cm_diff_full[1]
highclip_diff = cm_diff_full[end]
cm_diff = cgrad(collect(cm_diff_full[2:(end - 1)]); categorical = true)
scale_diff = mk_piecewise_linear(levels_diff)
N_diff = length(levels_diff) - 1

@info "Levels" mean = levels_mean diff = levels_diff

lon_window_start = 20

function add_coastlines!(ax)
    coast = GeoMakie.coastlines()
    cl1 = lines!(ax, coast; color = :black, linewidth = 0.6)
    cl2 = lines!(ax, coast; color = :black, linewidth = 0.6)
    translate!(cl1, 0, 0, 50)
    translate!(cl2, 360, 0, 50)
    return nothing
end

xticks_map = 90:90:360
yticks_map = -60:30:60

mapaxis(pos) = Axis(
    pos;
    backgroundcolor = :lightgray, xgridvisible = false, ygridvisible = false,
    xticks = (collect(xticks_map), lonticklabel.(xticks_map)),
    yticks = (collect(yticks_map), latticklabel.(yticks_map)),
)

function drawmap!(ax, field, gm; isdiff)
    if isdiff
        plotmap!(
            ax, field, gm; colorrange = extrema(levels_diff), colormap = cm_diff,
            highclip = highclip_diff, lowclip = lowclip_diff, colorscale = scale_diff, lon_window_start,
        )
    else
        plotmap!(
            ax, field, gm; colorrange = extrema(levels_mean), colormap = cm_mean,
            highclip = highclip_mean, colorscale = scale_mean, lon_window_start,
        )
    end
    add_coastlines!(ax)
    xlims!(ax, (lon_window_start, lon_window_start + 360))
    ylims!(ax, (-90, 90))
    return nothing
end

################################################################################
# Build the 3×3 figure
################################################################################

@info "Building figure"
flush(stdout); flush(stderr)

fig = Figure(; size = (1488, 744), fontsize = 15)
g = fig[1, 1] = GridLayout()

Label(g[1, 2], tag1; font = :bold, tellwidth = false)
Label(g[1, 3], tag2; font = :bold, tellwidth = false)
Label(g[1, 4], "Δ resolution\n($tag2 − $tag1)"; font = :bold, tellwidth = false)
Label(g[2, 1], TW1; font = :bold, rotation = pi / 2, tellheight = false)
Label(g[3, 1], TW2; font = :bold, rotation = pi / 2, tellheight = false)
Label(g[4, 1], "Δ decade\n($TW2 − $TW1)"; font = :bold, rotation = pi / 2, tellheight = false)

ax_11 = mapaxis(g[2, 2]); drawmap!(ax_11, v1_tw1, gm1; isdiff = false)
ax_12 = mapaxis(g[2, 3]); drawmap!(ax_12, v2_tw1, gm2; isdiff = false)
ax_13 = mapaxis(g[2, 4]); drawmap!(ax_13, rdiff_tw1, gm2; isdiff = true)
ax_21 = mapaxis(g[3, 2]); drawmap!(ax_21, v1_tw2, gm1; isdiff = false)
ax_22 = mapaxis(g[3, 3]); drawmap!(ax_22, v2_tw2, gm2; isdiff = false)
ax_23 = mapaxis(g[3, 4]); drawmap!(ax_23, rdiff_tw2, gm2; isdiff = true)
ax_31 = mapaxis(g[4, 2]); drawmap!(ax_31, ddiff_om1, gm1; isdiff = true)
ax_32 = mapaxis(g[4, 3]); drawmap!(ax_32, ddiff_om2, gm2; isdiff = true)

for ax in (ax_11, ax_12, ax_13, ax_21, ax_22, ax_23)
    hidexdecorations!(ax; ticks = false, grid = false)
end
for ax in (ax_12, ax_13, ax_22, ax_23, ax_32)
    hideydecorations!(ax; ticks = false, grid = false)
end

# Colorbars in the empty [4,4] corner.
cbs = g[4, 4] = GridLayout()
mean_label = rich("% v", subscript("tot"), " / (10,000 km)", superscript("2"))
diff_label = rich("Δ % v", subscript("tot"), " / (10,000 km)", superscript("2"))
Colorbar(
    cbs[1, 1]; colormap = cm_mean, colorrange = (0, N_mean), highclip = highclip_mean,
    ticks = (0:N_mean, [isinteger(x) ? string(Int(x)) : string(x) for x in levels_mean]),
    label = mean_label, vertical = false, flipaxis = false, tellwidth = false, tellheight = false,
    width = Relative(0.85),
)
Colorbar(
    cbs[2, 1]; colormap = cm_diff, colorrange = (0, N_diff),
    lowclip = lowclip_diff, highclip = highclip_diff,
    ticks = (0:N_diff, divergingcbarticklabelformat(levels_diff)),
    label = diff_label, vertical = false, flipaxis = false, tellwidth = false, tellheight = false,
    width = Relative(0.85),
)
rowgap!(cbs, 2)

Label(
    g[0, 1:4]; text = "Surface ventilation $leg_long, cross-resolution & cross-decade comparison",
    fontsize = 19, font = :bold, tellwidth = false,
)

################################################################################
# Save
################################################################################

outdir = joinpath(repo_root, "outputs", "cross_resolution", "ventilation")
mkpath(outdir)
outfile = joinpath(outdir, "calVdown_$(leg_tag)_3x3.png")
@info "Saving $outfile"
flush(stdout); flush(stderr)
save(outfile, fig; px_per_unit = 2)

@info "plot_cross_resolution_ventilation.jl complete — saved $outfile"
flush(stdout); flush(stderr)
