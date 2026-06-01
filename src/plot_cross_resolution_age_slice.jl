"""
Cross-resolution + cross-decade depth-slice comparison of the forward periodic
age (combines the four panels of "Figure 1a" of the cross-resolution
ventilation paper into a single 3×3 figure).

Layout (3 rows × 3 cols of maps; the [3,3] corner is a GridLayout holding the
two shared colorbars):

                 OM2-1        OM2-025      Δ resolution (025 − 1)
  1968–1977     [1,1] age    [1,2] age    [1,3] diff
  1999–2008     [2,1] age    [2,2] age    [2,3] diff
  Δ decade      [3,1] diff   [3,2] diff   [colorbars]

- Age panels (cols 1–2, rows 1–2): viridis depth-slice of the time-mean age.
- Δ-resolution panels (col 3): OM2-025 minus OM2-1, with the OM2-1 slice
  conservatively regridded onto the OM2-025 grid via ConservativeRegridding.jl
  (masked-conservative: regrid age·wetmask and wetmask separately, then divide,
  so coastline mismatches between resolutions don't bias the regridded values).
- Δ-decade panels (row 3): (1999–2008) − (1968–1977) on each native grid (no
  regridding needed — same grid within a resolution).

Each map is drawn as a per-cell quad mesh on the native tripolar grid via the
`plotmap!` helper in shared_utils/plotting_functions.jl (no re-grid for display,
no dateline artefacts), with GeoMakie coastlines.

This is a standalone CPU script. It reads the forward `age_periodic_1year.jld2`
1-year FieldTimeSeries for both resolutions and both time windows; the per-model
`model_config` tags differ (OM2-025 carries the `LBS` preconditioner tag and a
different timestep multiplier), so they are configured separately below and can
be overridden via environment variables.

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=96GB -q normal -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/plot_cross_resolution_age_slice.jl
```

Environment variables (all optional; defaults match the paper config):
  MODEL_CONFIG_OM21   – OM2-1 model_config   (default totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12)
  MODEL_CONFIG_OM2025 – OM2-025 model_config (default totaltransport_wdiagnosed_centered2_SRK3_mkappaV_LBS_DTx9)
  SOLVER_TAG          – 1-year solver tag dir (default Pardiso_LSprec)
  TW1, TW2            – the two time windows  (default 1968-1977, 1999-2008)
  DEPTH               – slice depth in metres (default 2000)
  TRAF                – yes ⇒ adjoint age (age_traf, `_traf` model_config suffix)
  AGE_CMIN/AGE_CMAX/AGE_DLEVEL    – age colour scale     (default 0 / 2000 / 100)
  DIFF_CMIN/DIFF_CMAX/DIFF_DLEVEL – diff colour scale     (default -1000 / 1000 / 100;
                                    symmetric, zero excluded → single white band ∓DLEVEL)
"""

@info "Loading packages for cross-resolution age-slice plot"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
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

const YEAR = 365.25 * 86400.0  # seconds

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
age_var = "age"                                  # FTS variable name (forward or traf)
config_suffix = TRAF ? "_traf" : ""
leg_long = TRAF ? "Adjoint age Γ↑" : "Forward age Γ↓"

mc_om21 = get(ENV, "MODEL_CONFIG_OM21", "totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12") * config_suffix
mc_om2025 = get(ENV, "MODEL_CONFIG_OM2025", "totaltransport_wdiagnosed_centered2_SRK3_mkappaV_LBS_DTx9") * config_suffix
solver_tag = get(ENV, "SOLVER_TAG", "Pardiso_LSprec")

TW1 = get(ENV, "TW1", "1968-1977")
TW2 = get(ENV, "TW2", "1999-2008")

depth_m = parse(Float64, get(ENV, "DEPTH", "2000"))

age_cmin = parse(Float64, get(ENV, "AGE_CMIN", "0"))
age_cmax = parse(Float64, get(ENV, "AGE_CMAX", "2000"))
age_dlevel = parse(Float64, get(ENV, "AGE_DLEVEL", "100"))
diff_cmin = parse(Float64, get(ENV, "DIFF_CMIN", "-1000"))
diff_cmax = parse(Float64, get(ENV, "DIFF_CMAX", "1000"))
diff_dlevel = parse(Float64, get(ENV, "DIFF_DLEVEL", "100"))

# (short tag, parent model, experiment, model_config)
models = [
    ("OM2-1", "ACCESS-OM2-1", "1deg_jra55_iaf_omip2_cycle6", mc_om21),
    ("OM2-025", "ACCESS-OM2-025", "025deg_jra55_iaf_omip2_cycle6", mc_om2025),
]

repo_root = normpath(joinpath(@__DIR__, ".."))

age_fts_path(pm, exp, mc, tw) = joinpath(
    repo_root, "outputs", pm, exp, tw, "periodic", mc, "1year", solver_tag,
    "age_periodic_1year.jld2",
)
grid_path(pm, exp) = joinpath(repo_root, "preprocessed_inputs", pm, exp, "grid.jld2")

@info "plot_cross_resolution_age_slice.jl configuration"
@info "- leg            = $(TRAF ? "adjoint (TRAF)" : "forward")"
@info "- OM2-1 config   = $mc_om21"
@info "- OM2-025 config = $mc_om2025"
@info "- solver tag     = $solver_tag"
@info "- time windows   = $TW1, $TW2"
@info "- depth          = $depth_m m"
@info "- age scale      = ($age_cmin, $age_cmax) step $age_dlevel"
@info "- diff scale     = ($diff_cmin, $diff_cmax) step $diff_dlevel"
flush(stdout); flush(stderr)

################################################################################
# Helpers
################################################################################

"""
    gridmetrics_from(grid, Nx, Ny) -> NamedTuple

Build the `(; lon, lat, lon_vertices, lat_vertices)` NamedTuple that `plotmap!`
expects, from an Oceananigans tripolar grid, on the interior `(Nx, Ny)` cells.
Mirrors the construction in plot_ventilation.jl.
"""
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

"""
    mean_age_slice(fts_path, grid, k) -> Matrix (Nx, Ny), years, NaN at dry

Time-mean of the depth-`k` slice over snapshots 1..(n-1) (the last snapshot is
the periodicity-check duplicate of n=1). Streams the FTS from disk so the big
OM2-025 file never lives fully in memory.
"""
function mean_age_slice(fts_path, grid, k)
    isfile(fts_path) || error("Age FTS not found: $fts_path")
    fts = FieldTimeSeries(fts_path, age_var; backend = OnDisk())
    n_times = length(fts.times)
    n_avg = n_times - 1
    (; wet3D) = compute_wet_mask(grid)
    wet_k = wet3D[:, :, k]
    Nx, Ny = size(wet_k)
    accum = zeros(Float64, Nx, Ny)
    for n in 1:n_avg
        slice_n = @view interior(fts[n])[:, :, k]
        @. accum += ifelse(wet_k, Float64(slice_n), 0.0)
    end
    out = similar(accum)
    @. out = ifelse(wet_k, accum / (n_avg * YEAR), NaN)   # → years
    check_age_field(
        reshape(out, Nx, Ny, 1), reshape(wet_k, Nx, Ny, 1), grid;
        kind = "age 2000 m slice", min_yr = 0.0, max_yr = 1.0e4, label = fts_path
    )
    return out, wet_k
end

"""
    regrid_slice(R, src_slice, src_wet, dst_wet) -> Matrix on dst grid

Masked-conservative regrid of a 2-D `src_slice` (NaN at dry) onto the
destination grid of regridder `R`. Regrids `age·wet` and `wet` separately and
divides, so only wet source cells contribute. Result is NaN where no wet source
overlap exists or where the destination cell is dry.
"""
function regrid_slice(R, src_slice, src_wet, dst_wet)
    nd = length(R.dst_areas)
    f = vec([src_wet[i] && isfinite(src_slice[i]) ? Float64(src_slice[i]) : 0.0 for i in eachindex(src_slice)])
    w = vec([src_wet[i] && isfinite(src_slice[i]) ? 1.0 : 0.0 for i in eachindex(src_slice)])
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

logrange(name, a) = (
    v = filter(isfinite, vec(a)); isempty(v) ?
        @info("  $name: all-NaN") :
        @info(@sprintf("  %s: min=%+.1f mean=%+.1f max=%+.1f", name, minimum(v), mean(v), maximum(v)))
)

################################################################################
# Load grids + slices
################################################################################

(tag1, pm1, exp1, mc1) = models[1]   # OM2-1  (source for regridding)
(tag2, pm2, exp2, mc2) = models[2]   # OM2-025 (destination)

@info "Loading grids"
flush(stdout); flush(stderr)
grid1 = load_tripolar_grid(grid_path(pm1, exp1), CPU())
grid2 = load_tripolar_grid(grid_path(pm2, exp2), CPU())

k1 = find_nearest_depth_index(grid1, depth_m)
k2 = find_nearest_depth_index(grid2, depth_m)
z1 = -znodes(grid1, Center(), Center(), Center())[k1]
z2 = -znodes(grid2, Center(), Center(), Center())[k2]
@info "Slice depth indices" om21_k = k1 om21_z = z1 om2025_k = k2 om2025_z = z2

@info "Slicing OM2-1 ($TW1, $TW2)"
flush(stdout); flush(stderr)
a1_tw1, wet1_k = mean_age_slice(age_fts_path(pm1, exp1, mc1, TW1), grid1, k1)
a1_tw2, _ = mean_age_slice(age_fts_path(pm1, exp1, mc1, TW2), grid1, k1)

@info "Slicing OM2-025 ($TW1, $TW2)"
flush(stdout); flush(stderr)
a2_tw1, wet2_k = mean_age_slice(age_fts_path(pm2, exp2, mc2, TW1), grid2, k2)
a2_tw2, _ = mean_age_slice(age_fts_path(pm2, exp2, mc2, TW2), grid2, k2)

Nx1, Ny1 = size(a1_tw1)
Nx2, Ny2 = size(a2_tw1)

################################################################################
# Regrid OM2-1 → OM2-025 and build the difference fields
################################################################################

@info "Building conservative regridder OM2-1 → OM2-025 (one-time intersection)"
flush(stdout); flush(stderr)
R = CR.Regridder(CenterField(grid2.underlying_grid), CenterField(grid1.underlying_grid); progress = true)
@info "Regridder built" size = size(R)
flush(stdout); flush(stderr)

a1_tw1_on2 = regrid_slice(R, a1_tw1, wet1_k, wet2_k)
a1_tw2_on2 = regrid_slice(R, a1_tw2, wet1_k, wet2_k)

# Δ resolution (025 − 1), on the OM2-025 grid
rdiff_tw1 = a2_tw1 .- a1_tw1_on2
rdiff_tw2 = a2_tw2 .- a1_tw2_on2

# Δ decade (TW2 − TW1), on each native grid
ddiff_om1 = a1_tw2 .- a1_tw1
ddiff_om2 = a2_tw2 .- a2_tw1

@info "Field magnitude summary (years)"
logrange("OM2-1   $TW1", a1_tw1); logrange("OM2-1   $TW2", a1_tw2)
logrange("OM2-025 $TW1", a2_tw1); logrange("OM2-025 $TW2", a2_tw2)
logrange("Δres    $TW1", rdiff_tw1); logrange("Δres    $TW2", rdiff_tw2)
logrange("Δdecade OM2-1", ddiff_om1); logrange("Δdecade OM2-025", ddiff_om2)
flush(stdout); flush(stderr)

################################################################################
# Grid metrics + colour scales
################################################################################

gm1 = gridmetrics_from(grid1, Nx1, Ny1)
gm2 = gridmetrics_from(grid2, Nx2, Ny2)

age_levels = collect(age_cmin:age_dlevel:age_cmax)
age_cmap = cgrad(:viridis, length(age_levels) - 1, categorical = true)
age_range = (age_cmin, age_cmax)

# Diff levels are symmetric and EXCLUDE zero (cf. plot_ventilation.jl L232), so
# the central band [-diff_dlevel, +diff_dlevel] straddles zero as a single band.
# That makes the bin count odd, so withwhitecenter whitens exactly that central
# band → values close to zero render white. mk_piecewise_linear maps the
# (now non-uniform) central band to an equal-width colour bin.
diff_ext = max(abs(diff_cmin), abs(diff_cmax))
diff_pos = collect(diff_dlevel:diff_dlevel:diff_ext)
diff_levels = [-reverse(diff_pos); diff_pos]
n_diff_bins = length(diff_levels) - 1                 # odd
# White-centred diverging map via withwhitecenter (cf. plot_ventilation.jl L235).
# That helper whitens the centre entry of the colour scheme; it works there
# because PRGn is an 11-colour scheme sampled at 11 (identity). :balance is a
# 256-colour scheme, so we first bin it down to n_diff_bins colours as a
# ColorScheme — withwhitecenter then whitens the real centre band without the
# single white entry being diluted on resampling.
balance_binned = Makie.ColorSchemes.ColorScheme(
    [cgrad(:balance, n_diff_bins, categorical = true)[i] for i in 1:n_diff_bins]
)
diff_cmap = cgrad(withwhitecenter(balance_binned), n_diff_bins; categorical = true)
diff_scale = mk_piecewise_linear(diff_levels)
diff_range = extrema(diff_levels)

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

function mapaxis(pos)
    ax = Axis(
        pos;
        backgroundcolor = :lightgray,
        xgridvisible = true, ygridvisible = true,
        # aspect = DataAspect(),
        xticks = (collect(xticks_map), lonticklabel.(xticks_map)),
        yticks = (collect(yticks_map), latticklabel.(yticks_map)),
    )
    return ax
end

function drawmap!(ax, field, gm; isdiff)
    cmap = isdiff ? diff_cmap : age_cmap
    crange = isdiff ? diff_range : age_range
    cscale = isdiff ? diff_scale : identity
    plotmap!(
        ax, field, gm;
        colorrange = crange, colormap = cmap,
        highclip = cmap[end], lowclip = cmap[1],
        colorscale = cscale, lon_window_start,
    )
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

fig = Figure(; size = (1750, 875), fontsize = 15)
g = fig[1, 1] = GridLayout()

# Column headers (row 1; data axes occupy cols 2-4, rows 2-4)
Label(g[1, 2], tag1; font = :bold, tellwidth = false)
Label(g[1, 3], tag2; font = :bold, tellwidth = false)
Label(g[1, 4], "Δ resolution\n($tag2 − $tag1)"; font = :bold, tellwidth = false)

# Row headers (col 1, rotated)
Label(g[2, 1], TW1; font = :bold, rotation = pi / 2, tellheight = false)
Label(g[3, 1], TW2; font = :bold, rotation = pi / 2, tellheight = false)
Label(g[4, 1], "Δ decade\n($TW2 − $TW1)"; font = :bold, rotation = pi / 2, tellheight = false)

# ── Row 1 (TW1) ──
ax_11 = mapaxis(g[2, 2]); drawmap!(ax_11, a1_tw1, gm1; isdiff = false)
ax_12 = mapaxis(g[2, 3]); drawmap!(ax_12, a2_tw1, gm2; isdiff = false)
ax_13 = mapaxis(g[2, 4]); drawmap!(ax_13, rdiff_tw1, gm2; isdiff = true)

# ── Row 2 (TW2) ──
ax_21 = mapaxis(g[3, 2]); drawmap!(ax_21, a1_tw2, gm1; isdiff = false)
ax_22 = mapaxis(g[3, 3]); drawmap!(ax_22, a2_tw2, gm2; isdiff = false)
ax_23 = mapaxis(g[3, 4]); drawmap!(ax_23, rdiff_tw2, gm2; isdiff = true)

# ── Row 3 (Δ decade) ──
ax_31 = mapaxis(g[4, 2]); drawmap!(ax_31, ddiff_om1, gm1; isdiff = true)
ax_32 = mapaxis(g[4, 3]); drawmap!(ax_32, ddiff_om2, gm2; isdiff = true)

# Hide inner tick labels for a cleaner grid (keep left column + bottom row).
for ax in (ax_11, ax_12, ax_13, ax_21, ax_22, ax_23)
    hidexdecorations!(ax; ticks = false, grid = false)
end
for ax in (ax_12, ax_13, ax_22, ax_23, ax_32)
    hideydecorations!(ax; ticks = false, grid = false)
end

# ── Colorbars in the empty [3,3] corner (g[4,4]) ──
cbs = g[4, 4] = GridLayout()
age_ticks = age_cmin:max(age_dlevel, (age_cmax - age_cmin) / 4):age_cmax
Colorbar(
    cbs[1, 1];
    colormap = age_cmap, limits = age_range, highclip = age_cmap[end],
    ticks = collect(age_ticks), label = "Age (years)",
    vertical = false, flipaxis = false, tellwidth = false, tellheight = false,
)
# Diff colorbar in index space (0:N_diff) so each colour bin is equal width and
# ticks sit at the (zero-excluding) level boundaries — the central white band is
# bracketed by ∓diff_dlevel, no "0" tick (cf. plot_ventilation.jl).
Colorbar(
    cbs[2, 1];
    colormap = diff_cmap, colorrange = (0, n_diff_bins),
    lowclip = diff_cmap[1], highclip = diff_cmap[end],
    ticks = (0:n_diff_bins, divergingcbarticklabelformat(diff_levels)),
    label = "Δ Age (years)",
    vertical = false, flipaxis = false, tellwidth = false, tellheight = false,
)

Label(
    g[0, 1:4];
    text = "$leg_long at $(Int(round(depth_m))) m — cross-resolution & cross-decade comparison",
    fontsize = 19, font = :bold, tellwidth = false,
)

# colgap!(g, 8)
# rowgap!(g, 8)
# resize_to_layout!(fig)

################################################################################
# Save
################################################################################

leg_tag = TRAF ? "adjoint" : "forward"
outdir = joinpath(repo_root, "outputs", "cross_resolution", "age_slice")
mkpath(outdir)
outfile = joinpath(outdir, "age_slice_$(Int(round(depth_m)))m_$(leg_tag)_3x3.png")
@info "Saving $outfile"
flush(stdout); flush(stderr)
save(outfile, fig; px_per_unit = 2)

@info "plot_cross_resolution_age_slice.jl complete — saved $outfile"
flush(stdout); flush(stderr)
