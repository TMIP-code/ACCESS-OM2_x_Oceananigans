# Plot depth-space MOC (Meridional Overturning Circulation) streamfunction
#
# Produces:
#   1. Static PNG of the time-mean MOC (from yearly-averaged ty_trans)
#   2. MP4 animation of the monthly MOC cycle (from 12-month climatology ty_trans)
#
# Loads the Oceananigans grid and places ty_trans on it as a YFaceField,
# using the same convention as prep_velocities.jl. The immersed boundary
# mask provides land masking; basin masks come from compute_ocean_basin_masks.
#
# Usage:
#   PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1958-1987 julia --project src/plot_MOC.jl

@info "Loading packages"

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using JLD2
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
    PARENT_MODEL == "ACCESS-OM2-1" ? "1deg_jra55_iaf_omip2_cycle6" : "025deg_jra55_iaf_omip2_cycle6"
)
TIME_WINDOW = get(ENV, "TIME_WINDOW", "1960-1979")

const ρ₀ = 1035.0  # reference density (kg/m³)

# Output directory
outputdir = joinpath(@__DIR__, "..", "outputs", PARENT_MODEL, EXPERIMENT, TIME_WINDOW, "MOC")
mkpath(outputdir)

# latticklabel, DEPTH_YLIM, DEPTH_YTICKS, DEPTH_YTICKLABELS, MONTH_NAMES
# are in shared_utils/analysis_and_plotting.jl

# ── Load grid ─────────────────────────────────────────────────────────────

@info "Loading grid"
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

# Build wet mask at YFace locations using mask_immersed_field!
ty_wet = YFaceField(grid)
set!(ty_wet, 1.0)
mask_immersed_field!(ty_wet, NaN)
wet3D_yface = .!isnan.(Array(interior(ty_wet)))
Nx′, Ny_f, Nz′ = size(wet3D_yface)

# Grid coordinates at (Center, Face) locations — matching ty_trans
ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
lat_cf = Array(ug.φᶜᶠᵃ[1:Nx′, 1:Ny_f])
lon_cf = Array(ug.λᶜᶠᵃ[1:Nx′, 1:Ny_f])

# Basin masks at YFace locations
const OCEANS = oceanpolygons()
flat_lat = vec(lat_cf)
flat_lon = vec(lon_cf)

atl_2D = reshape(isatlantic(flat_lat, flat_lon, OCEANS), Nx′, Ny_f) .& (lat_cf .>= -30)
pac_2D = reshape(ispacific(flat_lat, flat_lon, OCEANS), Nx′, Ny_f)
ind_2D = reshape(isindian(flat_lat, flat_lon, OCEANS), Nx′, Ny_f)
indopac_2D = (ind_2D .| pac_2D) .& (lat_cf .>= -30)
gbl_2D = trues(Nx′, Ny_f)

basin_keys = (:ATL, :INDOPAC, :GBL)
basin_strs = ("Atlantic", "Indo-Pacific", "Global")
basins = (; ATL = atl_2D, INDOPAC = indopac_2D, GBL = gbl_2D)

# Representative latitude for the y-axis (maximum along i at Face location)
lat_repr = dropdims(maximum(lat_cf; dims = 1); dims = 1)  # (Ny_f,)

# North pole latitude: where the tripolar grid folds and lat/lon is no longer rectilinear.
# Detect by finding the first j where latitudes diverge across longitudes at (Face, Face).
lat_range = vec(maximum(ug.φᶠᶠᵃ; dims = 1) .- minimum(ug.φᶠᶠᵃ; dims = 1))
j_fold = findfirst(lat_range .> 0)
north_pole_latitude = ug.φᶠᶠᵃ[1, j_fold]
@info "North pole latitude: $(round(north_pole_latitude; digits = 1))° (j=$j_fold)"

# Depth values (positive downward)
z = znodes(grid, Center(), Center(), Center())
depth_vals = -z  # positive downward, length Nz

# Compute latitude limits for each basin
basin_latlims = map(basins) do mask
    lats_in_basin = lat_repr[vec(any(mask; dims = 1))]
    isempty(lats_in_basin) ? (-90.0, 90.0) : (minimum(lats_in_basin), maximum(lats_in_basin))
end

# fill_Cgrid_transport_from_MOM_output! is in shared_utils/data_loading.jl

# ── Load ty_trans data ────────────────────────────────────────────────────

@info "Loading ty_trans data"

# Open monthly climatology
ty_ds = open_dataset(joinpath(monthly_dir, "ty_trans_monthly.nc"))

# Open yearly average
ty_yearly_ds = open_dataset(joinpath(yearly_dir, "ty_trans_yearly.nc"))

# Pre-allocate fields for placing MOM transports on the grid
north_t = FPivotZipperBoundaryCondition(-1)
tx_bcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = north_t)
ty_bcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = north_t)
tx = XFaceField(grid; boundary_conditions = tx_bcs)
ty = YFaceField(grid; boundary_conditions = ty_bcs)

# Dummy tx_data (zeros) — we only need ty_trans but the kernel handles both
tx_zeros = zeros(Nx, Ny - 1, Nz)

# ── Streamfunction computation ────────────────────────────────────────────

"""
Compute the MOC streamfunction from a ty_trans field on the grid.

Extract interior, apply basin mask, sum over longitude, cumsum from bottom.
Returns ψ in Sv as a (Ny_f, Nz') array with NaN for land.
"""
function compute_moc(ty, basin_mask)
    ty_int = Array(interior(ty))  # (Nx', Ny_f, Nz') with NaN on land

    # Apply basin mask (2D) — set out-of-basin cells to 0 (not NaN, so sum works)
    mask3D = reshape(basin_mask, Nx′, Ny_f, 1)
    ty_masked = @. ifelse(mask3D, ifelse(isnan(ty_int), 0.0, ty_int), 0.0)

    # Sum over longitude
    ty_zonal = dropdims(sum(ty_masked; dims = 1); dims = 1)  # (Ny_f, Nz')

    # Streamfunction: cumsum from bottom (k=1) upward — ψ = 0 at bottom by construction
    ψ = cumsum(ty_zonal; dims = 2)

    # Convert kg/s to Sv
    ψ ./= (ρ₀ * 1.0e6)

    # Mask: NaN where no wet cell exists in the basin at this (lat, depth)
    wet_in_basin = wet3D_yface .& mask3D
    has_ocean = dropdims(any(wet_in_basin; dims = 1); dims = 1)  # (Ny_f, Nz')
    ψ[.!has_ocean] .= NaN

    return ψ
end

# ── Contourf plotting setup ───────────────────────────────────────────────

levels = -24:2:24
colormap = cgrad(:curl, length(levels) + 1; categorical = true, rev = true)
extendlow = colormap[1]
extendhigh = colormap[end]
colormap_inner = cgrad(colormap[2:(end - 1)]; categorical = true)

# ── Static PNG: time-mean MOC ─────────────────────────────────────────────

@info "Plotting time-mean MOC"

# Load yearly ty_trans onto grid
ty_yearly_data = readcubedata(ty_yearly_ds.ty_trans).data
map!(x -> isnan(x) ? zero(x) : x, ty_yearly_data, ty_yearly_data)
fill_Cgrid_transport_from_MOM_output!(tx, ty, grid, tx_zeros, ty_yearly_data)
mask_immersed_field!(ty, NaN)

# ── Build figure (once, with Observables) ─────────────────────────────────

# Compute time-mean MOC for initial display
ψ_mean = Dict(bk => compute_moc(ty, basins[bk]) for bk in basin_keys)

fig = Figure(; size = (1200, 400), fontsize = 18)
TIME_WINDOW_LABEL = replace(TIME_WINDOW, "-" => "–")
title_obs = Observable("$PARENT_MODEL $TIME_WINDOW_LABEL MOC")
Label(fig[-1, 1:length(basin_keys)]; text = title_obs, fontsize = 20, tellwidth = false)

ψ_obs = Dict(bk => Observable(copy(ψ_mean[bk])) for bk in basin_keys)

for (icol, (basin_key, basin_mask)) in enumerate(pairs(basins))
    ax = Axis(
        fig[1, icol];
        backgroundcolor = :lightgray,
        xgridvisible = true, ygridvisible = true,
        xgridcolor = (:black, 0.05), ygridcolor = (:black, 0.05),
        ylabel = "Depth (m)",
    )

    co = contourf!(
        ax, lat_repr, depth_vals, ψ_obs[basin_key];
        levels,
        colormap = colormap_inner,
        nan_color = :lightgray,
        extendlow,
        extendhigh,
    )
    translate!(co, 0, 0, -100)
    contour!(ax, lat_repr, depth_vals, ψ_obs[basin_key]; levels = [10, 20], color = :black, linewidth = 0.5)
    contour!(ax, lat_repr, depth_vals, ψ_obs[basin_key]; levels = [-20, -10], color = :black, linewidth = 0.5, linestyle = :dash)

    ax.yticks = (collect(DEPTH_YTICKS), DEPTH_YTICKLABELS)
    xtick_vals = collect(-90:30:90)
    xtick_labels = latticklabel.(xtick_vals)
    xtick_labels[end] = "Fold"
    ax.xticks = (xtick_vals, xtick_labels)
    ylims!(ax, DEPTH_YLIM)
    xlims!(ax, basin_latlims[basin_key])
    vlines!(ax, north_pole_latitude; color = :black, linewidth = 1, linestyle = :dash)

    hidexdecorations!(ax; label = false, ticklabels = false, ticks = false, grid = false)
    hideydecorations!(ax; label = icol > 1, ticklabels = icol > 1, ticks = icol > 1, grid = false)

    Label(fig[0, icol], "$(basin_strs[icol]) MOC"; tellwidth = false)

    colsize!(fig.layout, icol, Auto(basin_latlims[basin_key][2] - basin_latlims[basin_key][1]))

    # Add colorbar to the right of the last subplot
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

# ── Save static PNG (time mean) ──────────────────────────────────────────

outputfile = joinpath(outputdir, "MOC_mean.png")
@info "Saving $outputfile"
save(outputfile, fig; px_per_unit = 3)

# ── Animation: monthly MOC cycle ─────────────────────────────────────────

@info "Creating monthly MOC animation"

# Pre-compute all 12 monthly MOC streamfunctions per basin for interpolation
function load_monthly_ty!(ty, tx, grid, ty_ds, tx_zeros, m)
    ty_data = readcubedata(ty_ds.ty_trans[month = At(m)]).data
    map!(x -> isnan(x) ? zero(x) : x, ty_data, ty_data)
    fill_Cgrid_transport_from_MOM_output!(tx, ty, grid, tx_zeros, ty_data)
    mask_immersed_field!(ty, NaN)
    return nothing
end

ψ_all = Dict{Symbol, Vector{Matrix{Float64}}}()
for bk in basin_keys
    ψ_all[bk] = Matrix{Float64}[]
end
for m in 1:12
    load_monthly_ty!(ty, tx, grid, ty_ds, tx_zeros, m)
    for bk in basin_keys
        push!(ψ_all[bk], compute_moc(ty, basins[bk]))
    end
end

function interpolate_ψ(ψ_vec, t)
    # t in [0, 12), cyclical. Linearly interpolate between monthly snapshots.
    m0 = floor(Int, t) + 1
    m1 = mod1(m0 + 1, 12)
    α = t - floor(t)
    return @. (1 - α) * ψ_vec[m0] + α * ψ_vec[m1]
end

n_frames = 144
framerate = 24
frame_times = range(0, 12; length = n_frames + 1)[1:n_frames]

outputfile_anim = joinpath(outputdir, "MOC_monthly.mp4")
@info "Recording animation to $outputfile_anim"

record(fig, outputfile_anim, 1:n_frames; framerate) do i
    t = frame_times[i]
    for bk in basin_keys
        ψ_obs[bk][] = interpolate_ψ(ψ_all[bk], t)
    end
    month_frac = t + 0.5
    title_obs[] = @sprintf("%s %s MOC (%.1f months)", PARENT_MODEL, TIME_WINDOW_LABEL, month_frac)
end

@info "Done! Outputs in $outputdir"
