# Plot density-space MOC (Meridional Overturning Circulation) streamfunction
#
# Produces:
#   1. Static PNG of the time-mean resolved MOC in density space
#   2. Static PNG of the time-mean GM MOC in density space
#   3. Static PNG of the time-mean total MOC (resolved + GM) in density space
#
# Works directly from preprocessed NetCDF climatologies (ty_trans_rho, ty_trans_rho_gm)
# without requiring the Oceananigans grid. Basin masks use OceanBasins.jl with
# grid_yu_ocean/grid_xt_ocean coordinates from the MOM output.
#
# Usage:
#   PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1958-1987 julia --project src/plot_MOC_rho.jl

@info "Loading packages"

using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using CairoMakie
using OceanBasins
using Format
using Statistics: mean

include("shared_functions.jl")

# ── Configuration ─────────────────────────────────────────────────────────

(; parentmodel, experiment_dir, monthly_dir, yearly_dir) = load_project_config()

PARENT_MODEL = get(ENV, "PARENT_MODEL", "ACCESS-OM2-1")
EXPERIMENT = get(
    ENV, "EXPERIMENT",
    PARENT_MODEL == "ACCESS-OM2-1" ? "1deg_jra55_iaf_omip2_cycle6" : "025deg_jra55_iaf_omip2_cycle6",
)
TIME_WINDOW = get(ENV, "TIME_WINDOW", "1960-1979")

const ρ₀ = 1035.0  # reference density (kg/m³)

# Output directory
outputdir = joinpath(@__DIR__, "..", "outputs", PARENT_MODEL, EXPERIMENT, TIME_WINDOW, "MOC")
mkpath(outputdir)

# ── Load data ─────────────────────────────────────────────────────────────

@info "Loading ty_trans_rho and ty_trans_rho_gm yearly data"

ty_ds = open_dataset(joinpath(yearly_dir, "ty_trans_rho_yearly.nc"); driver = :netcdf)
ty_gm_ds = open_dataset(joinpath(yearly_dir, "ty_trans_rho_gm_yearly.nc"); driver = :netcdf)

ty_data = readcubedata(ty_ds.ty_trans_rho).data       # (xt, yu, potrho) — Julia column-major
ty_gm_data = readcubedata(ty_gm_ds.ty_trans_rho_gm).data  # (xt, yu, potrho)

potrho = ty_ds.potrho.val
lat = ty_ds.grid_yu_ocean.val
lon = ty_ds.grid_xt_ocean.val

Nrho = length(potrho)
Ny = length(lat)
Nx = length(lon)
@info "Data loaded: Nrho=$Nrho, Ny=$Ny, Nx=$Nx"

# Replace NaN/missing with 0 for summation
map!(x -> isnan(x) ? zero(x) : x, ty_data, ty_data)
map!(x -> isnan(x) ? zero(x) : x, ty_gm_data, ty_gm_data)

# ── Basin masks ───────────────────────────────────────────────────────────

@info "Computing basin masks"
const OCEANS = oceanpolygons()

# Build 2D lat/lon arrays for basin detection
lon2D = repeat(reshape(lon, 1, Nx); outer = (Ny, 1))  # (Ny, Nx)
lat2D = repeat(lat; outer = (1, Nx))                    # (Ny, Nx)
flat_lat = vec(lat2D)
flat_lon = vec(lon2D)

atl_2D = reshape(isatlantic(flat_lat, flat_lon, OCEANS), Ny, Nx) .& (lat2D .>= -30)
pac_2D = reshape(ispacific(flat_lat, flat_lon, OCEANS), Ny, Nx)
ind_2D = reshape(isindian(flat_lat, flat_lon, OCEANS), Ny, Nx)
indopac_2D = (ind_2D .| pac_2D) .& (lat2D .>= -30)
gbl_2D = trues(Ny, Nx)

basin_keys = (:ATL, :INDOPAC, :GBL)
basin_strs = ("Atlantic", "Indo-Pacific", "Global")
basins = (; ATL = atl_2D, INDOPAC = indopac_2D, GBL = gbl_2D)

# Latitude limits for each basin
basin_latlims = map(basins) do mask
    lats_in_basin = lat[vec(any(mask; dims = 2))]
    isempty(lats_in_basin) ? (-90.0, 90.0) : (minimum(lats_in_basin), maximum(lats_in_basin))
end

# ── Streamfunction computation ────────────────────────────────────────────

"""
Compute the resolved MOC streamfunction in density space.

ty_trans_rho has shape (Nx, Ny, Nrho) in Julia column-major order.
Apply basin mask, sum over longitude, cumsum over density.
Returns ψ in Sv as a (Ny, Nrho) array with NaN for land.
"""
function compute_moc_rho(ty, basin_mask)
    # ty is (Nx, Ny, Nrho), basin_mask is (Ny, Nx)
    mask3D = reshape(permutedims(basin_mask), Nx, Ny, 1)  # (Nx, Ny, 1)
    ty_masked = @. ifelse(mask3D, ty, 0.0)

    # Sum over longitude (dim 1) → (Ny, Nrho)
    ty_zonal = dropdims(sum(ty_masked; dims = 1); dims = 1)  # (Ny, Nrho)

    # Streamfunction: cumsum over density (dim 2), subtract total
    cs = cumsum(ty_zonal; dims = 2)              # (Ny, Nrho)
    total = cs[:, end:end]                        # (Ny, 1)
    ψ = cs .- total                               # ψ = 0 at densest

    # Convert kg/s to Sv
    ψ ./= (ρ₀ * 1.0e6)

    # Mask: NaN where no transport exists at this (lat, rho)
    ty_exists = dropdims(any(abs.(ty_masked) .> 0; dims = 1); dims = 1)  # (Ny, Nrho)
    ψ[.!ty_exists] .= NaN

    return ψ
end

"""
Compute the GM MOC streamfunction in density space.

ty_trans_rho_gm requires only zonal summation (no cumsum).
Returns ψ_gm in Sv as a (Ny, Nrho) array with NaN for land.
"""
function compute_moc_rho_gm(ty_gm, basin_mask)
    mask3D = reshape(permutedims(basin_mask), Nx, Ny, 1)
    ty_masked = @. ifelse(mask3D, ty_gm, 0.0)

    # Sum over longitude (dim 1) → (Ny, Nrho)
    ty_zonal = dropdims(sum(ty_masked; dims = 1); dims = 1)

    # Convert kg/s to Sv
    ψ_gm = ty_zonal ./ (ρ₀ * 1.0e6)

    # Mask
    ty_exists = dropdims(any(abs.(ty_masked) .> 0; dims = 1); dims = 1)
    ψ_gm[.!ty_exists] .= NaN

    return ψ_gm
end

# ── Contourf plotting setup ───────────────────────────────────────────────

levels = -24:2:24
colormap = cgrad(:curl, length(levels) + 1; categorical = true, rev = true)
extendlow = colormap[1]
extendhigh = colormap[end]
colormap_inner = cgrad(colormap[2:(end - 1)]; categorical = true)

# ── Density axis scaling ─────────────────────────────────────────────────
# Power-law scaling emphasises the dense (deep) end of the density axis,
# similar to how log-depth emphasises the upper ocean in z-space plots.

ρmin, ρmax = extrema(potrho)
ρmin -= eps(ρmin)
ρ_scale = Makie.ReversibleScale(
    ρ -> (ρ - ρmin)^4,
    x -> x^(1 / 4) + ρmin;
    limits = (ρmin, ρmax),
)

# ── Reusable figure-building function ─────────────────────────────────────

function build_moc_rho_figure(ψ_dict, title_str)
    fig = Figure(; size = (1200, 400), fontsize = 18)
    Label(fig[-1, 1:length(basin_keys)]; text = title_str, fontsize = 20, tellwidth = false)

    for (icol, (basin_key, basin_mask)) in enumerate(pairs(basins))
        ax = Axis(
            fig[1, icol];
            backgroundcolor = :lightgray,
            xgridvisible = true, ygridvisible = true,
            xgridcolor = (:black, 0.05), ygridcolor = (:black, 0.05),
            ylabel = "Potential density σ₀ (kg/m³)",
            yscale = ρ_scale,
            yreversed = true,
        )

        ψ = ψ_dict[basin_key]

        co = contourf!(
            ax, lat, potrho, ψ;
            levels,
            colormap = colormap_inner,
            nan_color = :lightgray,
            extendlow,
            extendhigh,
        )
        translate!(co, 0, 0, -100)
        contour!(ax, lat, potrho, ψ; levels = [10, 20], color = :black, linewidth = 0.5)
        contour!(
            ax, lat, potrho, ψ;
            levels = [-20, -10], color = :black, linewidth = 0.5, linestyle = :dash,
        )

        xtick_vals = collect(-90:30:90)
        xtick_labels = latticklabel.(xtick_vals)
        ax.xticks = (xtick_vals, xtick_labels)
        ax.yticks = round.(range(ρmin + eps(ρmin), ρmax; length = 6); digits = 1)
        xlims!(ax, basin_latlims[basin_key])
        ylims!(ax, (ρmin, ρmax))

        hidexdecorations!(ax; label = false, ticklabels = false, ticks = false, grid = false)
        hideydecorations!(
            ax; label = icol > 1, ticklabels = icol > 1, ticks = icol > 1, grid = false,
        )

        Label(fig[0, icol], "$(basin_strs[icol]) MOC"; tellwidth = false)

        colsize!(
            fig.layout, icol,
            Auto(basin_latlims[basin_key][2] - basin_latlims[basin_key][1]),
        )

        # Colorbar on the right of the last subplot
        icol == length(basin_keys) || continue
        Colorbar(
            fig[1, icol + 1], co;
            vertical = true, flipaxis = true,
            tickformat = x -> map(t -> replace(format("{:+d}", t), "-" => "−"), x),
            label = "MOC (Sv)",
        ).height = Relative(1)
    end

    rowgap!(fig.layout, 20)
    colgap!(fig.layout, 30)

    return fig
end

# ── Compute and plot ──────────────────────────────────────────────────────

TIME_WINDOW_LABEL = replace(TIME_WINDOW, "-" => "–")

# Resolved MOC
@info "Computing resolved MOC (density space)"
ψ_resolved = Dict(bk => compute_moc_rho(ty_data, basins[bk]) for bk in basin_keys)

fig_res = build_moc_rho_figure(ψ_resolved, "$PARENT_MODEL $TIME_WINDOW_LABEL Resolved MOC (σ₀)")
outputfile = joinpath(outputdir, "MOC_rho_resolved_mean.png")
@info "Saving $outputfile"
save(outputfile, fig_res; px_per_unit = 3)

# GM MOC
@info "Computing GM MOC (density space)"
ψ_gm = Dict(bk => compute_moc_rho_gm(ty_gm_data, basins[bk]) for bk in basin_keys)

fig_gm = build_moc_rho_figure(ψ_gm, "$PARENT_MODEL $TIME_WINDOW_LABEL GM MOC (σ₀)")
outputfile = joinpath(outputdir, "MOC_rho_gm_mean.png")
@info "Saving $outputfile"
save(outputfile, fig_gm; px_per_unit = 3)

# Total MOC (resolved + GM)
@info "Computing total MOC (density space)"
ψ_total = Dict(bk => ψ_resolved[bk] .+ ψ_gm[bk] for bk in basin_keys)

fig_tot = build_moc_rho_figure(ψ_total, "$PARENT_MODEL $TIME_WINDOW_LABEL Total MOC (σ₀)")
outputfile = joinpath(outputdir, "MOC_rho_total_mean.png")
@info "Saving $outputfile"
save(outputfile, fig_tot; px_per_unit = 3)

@info "Done! Outputs in $outputdir"
