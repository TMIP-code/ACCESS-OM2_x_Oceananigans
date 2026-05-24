"""
Plot the surface ventilation diagnostic `calVdown_norm` produced by
`compute_ventilation_diagnostic.jl`, using the Pasquier *et al.* 2024
(JGR-Oceans, doi:10.1029/2024JC021043) plotting style: `contourf!` on a
pseudo-log colour scale with the orange ramp from `ColorSchemes.Oranges`
and levels `[0, 10, 30, 100, 300, 1000]` in units `% v_tot / (10,000 km²)`.

For a single (PM, EXP, TW, MC) run, loads
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/ventilation.jld2
and produces a horizontal contourf map PNG (`calVdown_{forward|adjoint}_contourf.png`)
in the matching
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/plots/
directory. Handles both forward (IAF) and adjoint (TRAF) legs uniformly,
since `model_config` carries the `_traf` suffix automatically when `TRAF=yes`.

The tripolar `(λᶜᶜᵃ, φᶜᶜᵃ)` cell centres (2-D matrices) are re-gridded onto a
regular `(lon_1d, lat_1d)` grid by nearest-neighbour binning so the standard
`contourf!` recipe applies cleanly. This loses tripolar fidelity north of the
fold, but the surface-ventilation diagnostic is dominated by Southern Ocean
and North Atlantic features that are well-represented this way.

Cross-resolution / cross-time-window comparison panels, seasonality plots,
the decadal-difference panel, coastlines, the Ω sub-basin mask, and the
zonal-integral side panel are all explicitly out of scope here (see
docs/ventilation_figures.md "Out of scope").

Usage — interactive:
```
qsub -I -P y99 -l mem=24GB -q express -l walltime=00:30:00 -l ncpus=4 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/plot_ventilation.jl")
```
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using CairoMakie
using JLD2
using Printf
using Statistics

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

(; parentmodel, experiment, time_window, experiment_dir, outputdir) = load_project_config()

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

TRAF = lowercase(get(ENV, "TRAF", "no")) == "yes"
leg_tag = TRAF ? "adjoint" : "forward"
field_label = TRAF ? raw"$\mathcal{V}\!\!\downarrow_\mathrm{traf}$" : raw"$\mathcal{V}\!\!\downarrow_\mathrm{iaf}$"

ls = parse_lump_and_spray()

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"

periodic_root = isempty(gpu_tag) ?
    joinpath(outputdir, "periodic", model_config) :
    joinpath(outputdir, "periodic", model_config, gpu_tag)

# Mirror compute_ventilation_diagnostic.jl's search: prefer the new naming, fall back to legacy `NK/`.
candidate_dirs = unique([joinpath(periodic_root, "NK$(ls.dir_suffix)"), joinpath(periodic_root, "NK")])
hit = findfirst(d -> isfile(joinpath(d, "ventilation.jld2")), candidate_dirs)
hit === nothing && error(
    "ventilation.jld2 not found. Tried: " * join(candidate_dirs, ", ") *
        ". Run compute_ventilation_diagnostic.jl first.",
)
nk_output_dir = candidate_dirs[hit]
ventilation_file = joinpath(nk_output_dir, "ventilation.jld2")

plot_dir = joinpath(nk_output_dir, "plots")
mkpath(plot_dir)

@info "plot_ventilation.jl configuration"
@info "- PARENT_MODEL  = $parentmodel"
@info "- EXPERIMENT    = $experiment"
@info "- TIME_WINDOW   = $time_window"
@info "- model_config  = $model_config"
@info "- leg           = $leg_tag"
@info "- input         = $ventilation_file"
@info "- output dir    = $plot_dir"
flush(stdout); flush(stderr)

################################################################################
# Load data and grid
################################################################################

@info "Loading $ventilation_file"
flush(stdout); flush(stderr)
data = load(ventilation_file)
calVdown_raw = data["calVdown_raw"]    # (Nx′, Ny′), m, NaN at dry cells
calVdown_norm = data["calVdown_norm"]   # (Nx′, Ny′), % v_tot / (10,000 km²), NaN at dry cells
wet_surf = data["wet_surf"]
vtot = data["vtot"]
n_avg = data["n_avg"]

@info "Loading grid for plotting coordinates"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
ug = grid.underlying_grid
Nx′, Ny′ = size(calVdown_norm)
lon2D = Array(ug.λᶜᶜᵃ[1:Nx′, 1:Ny′])    # 2-D tripolar longitudes (deg)
lat2D = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])    # 2-D tripolar latitudes  (deg)

# Wrap longitudes to [-180, 180); contourf! needs a monotone axis.
lon2D_wrapped = @. mod(lon2D + 180, 360) - 180

raw_vals = filter(isfinite, calVdown_raw)
nrm_vals = filter(isfinite, calVdown_norm)
@info @sprintf(
    "calVdown_raw  [m]:               min = %.3e  mean = %.3e  max = %.3e",
    minimum(raw_vals), mean(raw_vals), maximum(raw_vals)
)
@info @sprintf(
    "calVdown_norm [%% v_tot / 1e4km²]: min = %.3e  mean = %.3e  max = %.3e",
    minimum(nrm_vals), mean(nrm_vals), maximum(nrm_vals)
)
for q in (0.5, 0.9, 0.99, 0.999)
    @info @sprintf("calVdown_norm quantile q=%.3f → %.3e", q, quantile(nrm_vals, q))
end

################################################################################
# Pasquier 2024 plotting recipe (template:
# /home/561/bp3051/Projects/MatrixMarineCarbonCycleModel/src/plotting/paper3/
#   plot_paper3_calVdown_maps_ZINT2.jl, lines 25-34, 107-120, 174-179)
################################################################################

# Colour-bar levels (% v_tot / (10,000 km²))
levelsPI = unique(Float32, clamp.([0; kron(10 .^ (1:3), [1, 3])], 0, 1000))
@info "Contour levels (% v_tot / (10,000 km²)): $levelsPI"

# Pseudo-log mapping reused from the template — limits=(0, 3) bracket myscale(1000) ≈ 2.3.
myscale = ReversibleScale(
    x -> sign(x) * log10(abs(x / 5) + 1),
    x -> sign(x) * (exp10(abs(x)) - 1) * 5;
    limits = (0.0f0, 3.0f0),
    name = :myscale,
)
pseudologlevelsPI = myscale.(levelsPI)

# White at the bottom → orange ramp.
function withwhitelow(cs)
    cs = deepcopy(cs)
    cs.colors[1] = Makie.Colors.colorant"white"
    return cs
end

colormapPI = cgrad(withwhitelow(Makie.ColorSchemes.Oranges), length(levelsPI); categorical = true)
extendhighPI = colormapPI[end]
colormapPI = colormapPI[1:(end - 1)]

# Tick formatters from the template.
function lonticklabel(lon)
    lon = mod(lon + 180, 360) - 180
    return if lon == 0
        "0°"
    elseif (lon ≈ 180) || (lon ≈ -180)
        "180°"
    elseif lon > 0
        "$(string(Int(lon)))°E"
    else
        "$(string(-Int(lon)))°W"
    end
end
function latticklabel(lat)
    return if lat == 0
        "0°"
    elseif lat > 0
        "$(string(Int(lat)))°N"
    else
        "$(string(-Int(lat)))°S"
    end
end

################################################################################
# Re-grid tripolar (lon2D, lat2D, calVdown_norm) → regular (lon1D, lat1D, Z)
# Nearest-neighbour binning over a 1° regular grid (matches the source 1°
# resolution at OM2-1; finer resolutions are still ≤ 0.25° so a 1° target loses
# only minor detail in the Southern Ocean fronts. Bumped to 0.5° for OM2-025 /
# OM2-01 below.)
################################################################################

dlon = parentmodel == "ACCESS-OM2-1" ? 1.0 : 0.5
dlat = parentmodel == "ACCESS-OM2-1" ? 1.0 : 0.5

lon1D = collect((-180 + dlon / 2):dlon:(180 - dlon / 2))
lat1D = collect((-90 + dlat / 2):dlat:(90 - dlat / 2))
Nlon = length(lon1D)
Nlat = length(lat1D)

# Bin sums and counts; nearest neighbour (no interpolation).
Z_sum = zeros(Float64, Nlon, Nlat)
Z_cnt = zeros(Int, Nlon, Nlat)

@inline function lon_idx(λ)
    λw = mod(λ + 180, 360) - 180
    i = floor(Int, (λw + 180) / dlon) + 1
    i < 1 && (i = 1)
    i > Nlon && (i = Nlon)
    return i
end
@inline function lat_idx(φ)
    j = floor(Int, (φ + 90) / dlat) + 1
    j < 1 && (j = 1)
    j > Nlat && (j = Nlat)
    return j
end

for jj in 1:Ny′, ii in 1:Nx′
    v = calVdown_norm[ii, jj]
    isfinite(v) || continue
    i = lon_idx(lon2D_wrapped[ii, jj])
    j = lat_idx(lat2D[ii, jj])
    Z_sum[i, j] += v
    Z_cnt[i, j] += 1
end

Z = fill(NaN, Nlon, Nlat)
for j in 1:Nlat, i in 1:Nlon
    if Z_cnt[i, j] > 0
        Z[i, j] = Z_sum[i, j] / Z_cnt[i, j]
    end
end

n_regridded = count(isfinite, Z)
@info "Re-gridded to ($Nlon × $Nlat) at $(dlon)° resolution; $n_regridded filled cells"

# Apply the pseudo-log mapping (myscale handles NaN → NaN cleanly).
pseudologZ = map(z -> isfinite(z) ? myscale(z) : NaN, Z)

# Longitude duplication trick from the template so the map wraps cleanly past
# the dateline / chosen lonlims.
lonext = [lon1D; lon1D .+ 360]
pseudologZext = vcat(pseudologZ, pseudologZ)

# Default lonlims: 30°E .. 30°E + 360° (matches template, gives an Atlantic-
# centred map with the Pacific in the middle and Africa on the left edge).
lonlims = 30 .+ (0, 360)
ilonkeep = @. lonlims[1] ≤ lonext ≤ lonlims[2]

################################################################################
# Single-panel contourf map (Pasquier 2024 panel (a))
################################################################################

@info "Generating contourf map → $(joinpath(plot_dir, "calVdown_$(leg_tag)_contourf.png"))"
flush(stdout); flush(stderr)

begin
    fig = Figure(; size = (1100, 600), figure_padding = (4, 4, 4, 4))

    ax = Axis(
        fig[1, 1];
        title = "Surface ventilation $field_label — $parentmodel, $time_window ($leg_tag)\n" *
            "annual mean of 1-year FTS (n_avg=$n_avg snapshots); v_tot = $(@sprintf("%.3e", vtot)) m³",
        xlabel = "Longitude",
        ylabel = "Latitude",
        backgroundcolor = :lightgray,
        xgridvisible = false, ygridvisible = false,
    )

    co = contourf!(
        ax, lonext[ilonkeep], lat1D, pseudologZext[ilonkeep, :];
        levels = pseudologlevelsPI,
        colormap = colormapPI,
        nan_color = :lightgray,
        extendhigh = extendhighPI,
    )

    xticks = 0:90:1000
    ax.xticks = (xticks, lonticklabel.(xticks))
    yticks = -90:30:90
    ax.yticks = (yticks, latticklabel.(yticks))
    xlims!(ax, lonlims)
    ylims!(ax, (-90, 90))

    Colorbar(
        fig[1, 2], co;
        ticks = (pseudologlevelsPI, string.(Int.(levelsPI))),
        label = rich("% v", subscript("tot"), " / (10,000 km", superscript("2"), ")"),
    )

    outputfile = joinpath(plot_dir, "calVdown_$(leg_tag)_contourf.png")
    @info "Saving $outputfile"
    save(outputfile, fig)
end

@info "plot_ventilation.jl complete"
flush(stdout); flush(stderr)
