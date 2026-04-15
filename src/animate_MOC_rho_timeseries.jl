# Animate the full monthly timeseries of global density-space MOC.
#
# Input:  /scratch/y99/TMIP/data/{MODEL}/{EXPERIMENT}/rhospace/psi_tot_global.nc
# Output: outputs/{MODEL}/{EXPERIMENT}/MOC_rho_global_timeseries.mp4
#
# Usage:
#   PARENT_MODEL=ACCESS-OM2-1 julia --project src/animate_MOC_rho_timeseries.jl

@info "Loading packages"

using NCDatasets
using CairoMakie
using Format
using Printf: @sprintf
using Dates: year, month

include("shared_functions.jl")

# ── Configuration ─────────────────────────────────────────────────────────

PARENT_MODEL = get(ENV, "PARENT_MODEL", "ACCESS-OM2-1")
EXPERIMENT = get(
    ENV, "EXPERIMENT",
    PARENT_MODEL == "ACCESS-OM2-1" ? "1deg_jra55_iaf_omip2_cycle6" :
        PARENT_MODEL == "ACCESS-OM2-025" ? "025deg_jra55_iaf_omip2_cycle6" :
        PARENT_MODEL == "ACCESS-OM2-01" ? "01deg_jra55v140_iaf_cycle4" : "",
)
isempty(EXPERIMENT) && error("Unknown PARENT_MODEL=$PARENT_MODEL; set EXPERIMENT env var")

PROJECT = get(ENV, "PROJECT", "y99")
infile = "/scratch/$PROJECT/TMIP/data/$PARENT_MODEL/$EXPERIMENT/rhospace/psi_tot_global.nc"
outputdir = joinpath(@__DIR__, "..", "outputs", PARENT_MODEL, EXPERIMENT)
mkpath(outputdir)
outfile = joinpath(outputdir, "MOC_rho_global_timeseries.mp4")

@info "Reading $infile"
ds = NCDataset(infile)
# NCDatasets returns Union{Missing, T}; coerce to plain Float64 with NaN for missings
ψ_raw = ds["psi_tot"][:, :, :]    # (grid_yu_ocean, potrho, time)
ψ_all = Array{Float64}(undef, size(ψ_raw))
@. ψ_all = ifelse(ismissing(ψ_raw), NaN, ψ_raw)
lat = Float64.(ds["grid_yu_ocean"][:])
potrho = Float64.(ds["potrho"][:])
times = ds["time"][:]             # Vector of DateTime
close(ds)

Ny, Nrho, Ntime = size(ψ_all)
@info "Loaded ψ cube: Ny=$Ny, Nrho=$Nrho, Ntime=$Ntime"

# ── Plotting setup ────────────────────────────────────────────────────────

levels = -24:2:24
colormap = cgrad(:curl, length(levels) + 1; categorical = true, rev = true)
extendlow = colormap[1]
extendhigh = colormap[end]
colormap_inner = cgrad(colormap[2:(end - 1)]; categorical = true)

ρmin, ρmax = extrema(potrho)
ρmin -= eps(ρmin)
ρ_scale = Makie.ReversibleScale(
    ρ -> (ρ - ρmin)^4,
    x -> x^(1 / 4) + ρmin;
    limits = (ρmin, ρmax),
)

# ── Build figure with Observables ─────────────────────────────────────────

ψ_buf = similar(ψ_all[:, :, 1])           # (Ny, Nrho) — reusable buffer
ψ_buf .= ψ_all[:, :, 1]
ψ_obs = Observable(copy(ψ_buf))

title_str_0 = @sprintf("%s — Global MOC σ₀ (%04d-%02d)", PARENT_MODEL, year(times[1]), month(times[1]))
title_obs = Observable(title_str_0)

fig = Figure(; size = (900, 500), fontsize = 18)
Label(fig[0, 1]; text = title_obs, fontsize = 20, tellwidth = false)

ax = Axis(
    fig[1, 1];
    backgroundcolor = :lightgray,
    xgridvisible = true, ygridvisible = true,
    xgridcolor = (:black, 0.05), ygridcolor = (:black, 0.05),
    ylabel = "Potential density σ₀ (kg/m³)",
    xlabel = "Latitude",
    yscale = ρ_scale,
    yreversed = true,
    limits = (extrema(lat)..., ρmin, ρmax),
)

co = contourf!(
    ax, lat, potrho, ψ_obs;
    levels,
    colormap = colormap_inner,
    nan_color = :lightgray,
    extendlow,
    extendhigh,
)
translate!(co, 0, 0, -100)
contour!(ax, lat, potrho, ψ_obs; levels = [10, 20], color = :black, linewidth = 0.5)
contour!(ax, lat, potrho, ψ_obs; levels = [-20, -10], color = :black, linewidth = 0.5, linestyle = :dash)

xtick_vals = collect(-90:30:90)
xtick_labels = latticklabel.(xtick_vals)
ax.xticks = (xtick_vals, xtick_labels)
ρticks = [ρ for ρ in ceil(ρmin):1.0:floor(ρmax) if ρ > ρmin + 0.1]
ax.yticks = round.(ρticks; digits = 1)

Colorbar(
    fig[1, 2], co;
    vertical = true, flipaxis = true,
    tickformat = x -> map(t -> replace(format("{:+d}", t), "-" => "−"), x),
    label = "MOC (Sv)",
).height = Relative(1)

# ── Animation ─────────────────────────────────────────────────────────────

framerate = 24

@info "Recording $Ntime frames at $framerate fps → $outfile"
record(fig, outfile, 1:Ntime; framerate) do i
    @views ψ_buf .= ψ_all[:, :, i]
    ψ_obs.val .= ψ_buf
    notify(ψ_obs)
    title_obs[] = @sprintf(
        "%s — Global MOC σ₀ (%04d-%02d)", PARENT_MODEL, year(times[i]), month(times[i]),
    )
end

@info "Done: $outfile"
