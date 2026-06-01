"""
Cross-resolution + cross-decade basin zonal-mean comparison of the forward
periodic age (the per-basin analogue of "Figure 1c" of the cross-resolution
ventilation paper).

Because the basin zonal mean is already a 1×3 panel (Atlantic / Pacific /
Indian), we emit **three figures — one per basin** — each keeping the same 3×3
structure as the depth-slice figure:

                 OM2-1        OM2-025      Δ resolution (025 − 1)
  1968–1977     [1,1] za     [1,2] za     [1,3] diff
  1999–2008     [2,1] za     [2,2] za     [2,3] diff
  Δ decade      [3,1] diff   [3,2] diff   [colorbars]

Each panel is a volume-weighted zonal mean (depth vs latitude), drawn with
`contourf`. Age panels use viridis `0:100:2000`; diff panels use a white-centred
`:balance` ramp with symmetric, zero-excluding levels (single white band
straddling zero, cf. plot_ventilation.jl).

**No 3-D conservative regridding.** The zonal mean is already reduced to
(latitude, depth). For the Δ-resolution column we zonal-average each model on its
own native grid/basin-mask, then linearly interpolate the OM2-1 profile onto the
OM2-025 latitude axis (the two grids share the *same* 50-level vertical grid, so
no depth interpolation is needed) and subtract. This compares what each model
actually produces and avoids re-masking OM2-1 data through OM2-025 coastlines.

Standalone CPU script. Reads the forward `age_periodic_1year.jld2` 1-year
FieldTimeSeries for both resolutions and both time windows (per-model
`model_config` tags differ; configurable via env vars, same as
plot_cross_resolution_age_slice.jl).

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=48GB -q normal -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project src/plot_cross_resolution_basin_zonal.jl
```

Environment variables: identical set to plot_cross_resolution_age_slice.jl
(MODEL_CONFIG_OM21, MODEL_CONFIG_OM2025, SOLVER_TAG, TW1, TW2, TRAF,
AGE_*/DIFF_* colour scales). DEPTH is ignored (full column is used).
"""

@info "Loading packages for cross-resolution basin-zonal plot"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.Architectures: CPU
using CairoMakie
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
const OCEANS = oceanpolygons()
using JLD2
using Printf
using Statistics

include("shared_functions.jl")
include(joinpath(@__DIR__, "shared_utils", "plotting_functions.jl"))

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration (mirrors plot_cross_resolution_age_slice.jl)
################################################################################

const YEAR = 365.25 * 86400.0  # seconds

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
age_var = "age"
config_suffix = TRAF ? "_traf" : ""
leg_long = TRAF ? "Adjoint age Γ↑" : "Forward age Γ↓"
leg_tag = TRAF ? "adjoint" : "forward"

mc_om21 = get(ENV, "MODEL_CONFIG_OM21", "totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12") * config_suffix
mc_om2025 = get(ENV, "MODEL_CONFIG_OM2025", "totaltransport_wdiagnosed_centered2_SRK3_mkappaV_LBS_DTx9") * config_suffix
solver_tag = get(ENV, "SOLVER_TAG", "Pardiso_LSprec")

TW1 = get(ENV, "TW1", "1968-1977")
TW2 = get(ENV, "TW2", "1999-2008")

age_cmin = parse(Float64, get(ENV, "AGE_CMIN", "0"))
age_cmax = parse(Float64, get(ENV, "AGE_CMAX", "2000"))
age_dlevel = parse(Float64, get(ENV, "AGE_DLEVEL", "100"))
diff_cmin = parse(Float64, get(ENV, "DIFF_CMIN", "-500"))
diff_cmax = parse(Float64, get(ENV, "DIFF_CMAX", "500"))
diff_dlevel = parse(Float64, get(ENV, "DIFF_DLEVEL", "100"))

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

@info "plot_cross_resolution_basin_zonal.jl configuration"
@info "- leg            = $leg_tag"
@info "- OM2-1 config   = $mc_om21"
@info "- OM2-025 config = $mc_om2025"
@info "- time windows   = $TW1, $TW2"
@info "- age scale      = ($age_cmin, $age_cmax) step $age_dlevel"
@info "- diff scale     = (±$(max(abs(diff_cmin), abs(diff_cmax)))) step $diff_dlevel"
flush(stdout); flush(stderr)

################################################################################
# Helpers
################################################################################

"""
    mean_age_3D(fts_path, grid, wet3D) -> Array (Nx, Ny, Nz), years, NaN at dry

Time-mean of the full 3-D age over snapshots 1..(n-1). Streams the FTS from disk
(OnDisk) so the big OM2-025 file never lives fully in memory.
"""
function mean_age_3D(fts_path, grid, wet3D)
    isfile(fts_path) || error("Age FTS not found: $fts_path")
    fts = FieldTimeSeries(fts_path, age_var; backend = OnDisk())
    n_avg = length(fts.times) - 1
    accum = zeros(Float64, size(wet3D))
    for n in 1:n_avg
        age_n = interior(fts[n])
        @. accum += ifelse(wet3D, Float64(age_n), 0.0)
    end
    out = similar(accum)
    @. out = ifelse(wet3D, accum / (n_avg * YEAR), NaN)   # → years
    # min_yr = -1e4 (not 0): small negative ages near the surface sink are a
    # normal centered-advection undershoot (here down to ~-100 yr), not a solver
    # pathology. The guard still catches catastrophic blowups (|age| > 1e4) and
    # non-finite values, matching the project's age-sanity convention.
    check_age_field(out, wet3D, grid; kind = "zonal-mean age 3D", min_yr = -1.0e4, max_yr = 1.0e4, label = fts_path)
    return out
end

"""
    lat_centres(grid, Nx, Ny) -> Vector (Ny)

Representative latitude per j-row (mean of cell-centre latitude along i).
"""
function lat_centres(grid, Nx, Ny)
    ug = grid isa Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lat = Array(ug.φᶜᶜᵃ[1:Nx, 1:Ny])
    return dropdims(mean(lat; dims = 1); dims = 1)
end

"""
    interp_lat(za_src, lat_src, lat_dst) -> Matrix (length(lat_dst), Nz)

Linear interpolation of each depth column of `za_src` (Ny_src × Nz) from
`lat_src` onto `lat_dst`. NaNs (dry / outside-basin) are skipped; targets outside
the finite latitude span of a column return NaN. (Any value interpolated across a
basin gap is harmless: the Δ-resolution panel masks it wherever the destination
zonal mean is itself NaN.)
"""
function interp_lat(za_src, lat_src, lat_dst)
    Nz = size(za_src, 2)
    out = fill(NaN, length(lat_dst), Nz)
    for k in 1:Nz
        col = @view za_src[:, k]
        idx = findall(isfinite, col)
        isempty(idx) && continue
        xs = Float64.(lat_src[idx])
        ys = Float64.(col[idx])
        sp = sortperm(xs)
        xs = xs[sp]; ys = ys[sp]
        for (jd, λ) in enumerate(lat_dst)
            (λ < xs[1] || λ > xs[end]) && continue
            j = searchsortedlast(xs, λ)
            if j == 0
                out[jd, k] = ys[1]
            elseif j >= length(xs)
                out[jd, k] = ys[end]
            else
                x0, x1 = xs[j], xs[j + 1]
                y0, y1 = ys[j], ys[j + 1]
                out[jd, k] = x1 == x0 ? y0 : y0 + (λ - x0) / (x1 - x0) * (y1 - y0)
            end
        end
    end
    return out
end

logrange(name, a) = (
    v = filter(isfinite, vec(a)); isempty(v) ?
        @info("  $name: all-NaN") :
        @info(@sprintf("  %s: min=%+.1f mean=%+.1f max=%+.1f", name, minimum(v), mean(v), maximum(v)))
)

################################################################################
# Colour scales (shared by all three basin figures; identical to the slice fig)
################################################################################

age_levels = collect(age_cmin:age_dlevel:age_cmax)
age_cmap = cgrad(:viridis, length(age_levels) - 1, categorical = true)
age_range = (age_cmin, age_cmax)

# Diff levels symmetric, EXCLUDING zero → single white band ∓diff_dlevel about 0
# (cf. plot_ventilation.jl L232). Odd bin count, so withwhitecenter whitens it.
diff_ext = max(abs(diff_cmin), abs(diff_cmax))
diff_pos = collect(diff_dlevel:diff_dlevel:diff_ext)
diff_levels = [-reverse(diff_pos); diff_pos]
n_diff_bins = length(diff_levels) - 1
# :balance is a 256-colour scheme; bin it to n_diff_bins colours first so
# withwhitecenter whitens the true centre band without resampling dilution.
balance_binned = Makie.ColorSchemes.ColorScheme(
    [cgrad(:balance, n_diff_bins, categorical = true)[i] for i in 1:n_diff_bins]
)
diff_cmap = cgrad(withwhitecenter(balance_binned), n_diff_bins; categorical = true)

################################################################################
# Load grids, masks, volumes; compute per-(model, TW, basin) zonal means
################################################################################

basin_keys = [("Atlantic", :ATL), ("Pacific", :PAC), ("Indian", :IND)]

@info "Loading grids and computing zonal means"
flush(stdout); flush(stderr)

# za[model_tag][tw][basin_sym] = Matrix(Ny, Nz); also keep lat axis + depth.
# (depth_vals mutated in-place to dodge top-level for-loop scoping on rebinding;
# both grids share the same 50-level vertical grid.)
za = Dict{String, Any}()
lat_of = Dict{String, Vector{Float64}}()
depth_vals = Float64[]

for (tag, pm, exp, mc) in models
    @info "  $tag: loading grid"
    flush(stdout); flush(stderr)
    grid = load_tripolar_grid(grid_path(pm, exp), CPU())
    (; wet3D) = compute_wet_mask(grid)
    Nx, Ny, Nz = size(wet3D)
    vol3D = Array(interior(compute_volume(grid)))
    basins = compute_ocean_basin_masks(grid, wet3D)
    lat_of[tag] = Float64.(lat_centres(grid, Nx, Ny))
    isempty(depth_vals) && append!(depth_vals, -collect(znodes(grid, Center(), Center(), Center())))

    za[tag] = Dict{String, Any}()
    for tw in (TW1, TW2)
        @info "  $tag: zonal means for $tw"
        flush(stdout); flush(stderr)
        age3D = mean_age_3D(age_fts_path(pm, exp, mc, tw), grid, wet3D)
        za[tag][tw] = Dict{Symbol, Matrix{Float64}}()
        for (_, sym) in basin_keys
            za[tag][tw][sym] = zonalaverage(age3D, vol3D, getfield(basins, sym))
        end
        age3D = nothing   # free before next TW
    end
    grid = nothing
end

maxdepth = ceil(maximum(depth_vals) / 1000) * 1000

################################################################################
# Figure builder (one per basin)
################################################################################

lat_ticks = -60:30:60

function zonalaxis(pos)
    return Axis(
        pos;
        backgroundcolor = :lightgray,
        xgridvisible = true, ygridvisible = true,
        xticks = (collect(lat_ticks), latticklabel.(lat_ticks)),
        yticks = (0:1000:maxdepth, string.(Int.(0:1000:maxdepth))),
    )
end

function zonalpanel!(ax, lat, za_field; isdiff)
    if isdiff
        cf = contourf!(
            ax, lat, depth_vals, za_field;
            levels = diff_levels, colormap = diff_cmap,
            extendlow = diff_cmap[1], extendhigh = diff_cmap[end], nan_color = :lightgray,
        )
    else
        cf = contourf!(
            ax, lat, depth_vals, za_field;
            levels = age_levels, colormap = age_cmap,
            extendhigh = age_cmap[end], nan_color = :lightgray,
        )
    end
    translate!(cf, 0, 0, -100)
    xlims!(ax, -90, 90)
    ylims!(ax, maxdepth, 0)
    return cf
end

function build_basin_figure(basin_name, sym)
    lat1 = lat_of["OM2-1"]
    lat2 = lat_of["OM2-025"]

    a1_t1 = za["OM2-1"][TW1][sym];  a1_t2 = za["OM2-1"][TW2][sym]
    a2_t1 = za["OM2-025"][TW1][sym]; a2_t2 = za["OM2-025"][TW2][sym]

    # Δ resolution (025 − 1) on the OM2-025 latitude axis
    rdiff_t1 = a2_t1 .- interp_lat(a1_t1, lat1, lat2)
    rdiff_t2 = a2_t2 .- interp_lat(a1_t2, lat1, lat2)
    # Δ decade (TW2 − TW1) on each native axis
    ddiff_1 = a1_t2 .- a1_t1
    ddiff_2 = a2_t2 .- a2_t1

    @info "[$basin_name] magnitude summary (years)"
    logrange("OM2-1   $TW1", a1_t1); logrange("OM2-025 $TW1", a2_t1)
    logrange("Δres    $TW1", rdiff_t1); logrange("Δres    $TW2", rdiff_t2)
    logrange("Δdecade OM2-1", ddiff_1); logrange("Δdecade OM2-025", ddiff_2)

    fig = Figure(; size = (1750, 875), fontsize = 15)
    g = fig[1, 1] = GridLayout()

    Label(g[1, 2], "OM2-1"; font = :bold, tellwidth = false)
    Label(g[1, 3], "OM2-025"; font = :bold, tellwidth = false)
    Label(g[1, 4], "Δ resolution\n(OM2-025 − OM2-1)"; font = :bold, tellwidth = false)
    Label(g[2, 1], TW1; font = :bold, rotation = pi / 2, tellheight = false)
    Label(g[3, 1], TW2; font = :bold, rotation = pi / 2, tellheight = false)
    Label(g[4, 1], "Δ decade\n($TW2 − $TW1)"; font = :bold, rotation = pi / 2, tellheight = false)

    ax_11 = zonalaxis(g[2, 2]); zonalpanel!(ax_11, lat1, a1_t1; isdiff = false)
    ax_12 = zonalaxis(g[2, 3]); zonalpanel!(ax_12, lat2, a2_t1; isdiff = false)
    ax_13 = zonalaxis(g[2, 4]); zonalpanel!(ax_13, lat2, rdiff_t1; isdiff = true)
    ax_21 = zonalaxis(g[3, 2]); zonalpanel!(ax_21, lat1, a1_t2; isdiff = false)
    ax_22 = zonalaxis(g[3, 3]); zonalpanel!(ax_22, lat2, a2_t2; isdiff = false)
    ax_23 = zonalaxis(g[3, 4]); zonalpanel!(ax_23, lat2, rdiff_t2; isdiff = true)
    ax_31 = zonalaxis(g[4, 2]); zonalpanel!(ax_31, lat1, ddiff_1; isdiff = true)
    ax_32 = zonalaxis(g[4, 3]); zonalpanel!(ax_32, lat2, ddiff_2; isdiff = true)

    for ax in (ax_11, ax_12, ax_13, ax_21, ax_22, ax_23)
        hidexdecorations!(ax; ticks = false, grid = false)
    end
    for ax in (ax_12, ax_13, ax_22, ax_23, ax_32)
        hideydecorations!(ax; ticks = false, grid = false)
    end
    for ax in (ax_11, ax_21, ax_31)
        ax.ylabel = "Depth (m)"
    end
    for ax in (ax_31, ax_32)
        ax.xlabel = "Latitude"
    end

    # Colorbars in the empty [3,3] corner.
    cbs = g[4, 4] = GridLayout()
    age_ticks = age_cmin:max(age_dlevel, (age_cmax - age_cmin) / 4):age_cmax
    Colorbar(
        cbs[1, 1];
        colormap = age_cmap, limits = age_range, highclip = age_cmap[end],
        ticks = collect(age_ticks), label = "Age (years)",
        vertical = false, flipaxis = false, tellwidth = false, tellheight = false,
    )
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
        text = "$leg_long — $basin_name basin zonal mean — cross-resolution & cross-decade",
        fontsize = 19, font = :bold, tellwidth = false,
    )

    colgap!(g, 8)
    rowgap!(g, 8)
    return fig
end

################################################################################
# Render + save (one PNG per basin)
################################################################################

outdir = joinpath(repo_root, "outputs", "cross_resolution", "zonal")
mkpath(outdir)

for (basin_name, sym) in basin_keys
    @info "Building $basin_name figure"
    flush(stdout); flush(stderr)
    fig = build_basin_figure(basin_name, sym)
    outfile = joinpath(outdir, "zonal_$(lowercase(basin_name))_$(leg_tag)_3x3.png")
    @info "Saving $outfile"
    save(outfile, fig; px_per_unit = 2)
end

@info "plot_cross_resolution_basin_zonal.jl complete — 3 basin figures saved to $outdir"
flush(stdout); flush(stderr)
