"""
Plot the surface ventilation diagnostic `calVdown` produced by
`compute_ventilation_diagnostic.jl`.

For a single (PM, EXP, TW, MC) run, loads
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/ventilation.jld2
and produces a horizontal map PNG (`calVdown.png`) in the matching
  outputs/{PM}/{EXP}/{TW}/periodic/{MC}/NK/plots/
directory. Handles both forward (IAF) and adjoint (TRAF) legs uniformly,
since `model_config` carries the `_traf` suffix automatically when
`TRAF=yes`.

Cross-resolution / cross-time-window comparison panels are out of scope
here; see `compare_NK_ages.jl` for the pattern.

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
# Load data
################################################################################

@info "Loading $ventilation_file"
flush(stdout); flush(stderr)
data = load(ventilation_file)
calVdown = data["calVdown"]      # (Nx′, Ny′), m, NaN at dry cells
wet_surf = data["wet_surf"]

@info "Loading grid for plotting coordinates"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
ug = grid.underlying_grid
Nx′, Ny′ = size(calVdown)
lon = Array(ug.λᶜᶜᵃ[1:Nx′, 1:Ny′])
lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])

cv_min, cv_max = extrema(filter(isfinite, calVdown))
cv_mean = mean(filter(isfinite, calVdown))
@info @sprintf("calVdown: min = %.3e m, mean = %.3e m, max = %.3e m", cv_min, cv_mean, cv_max)

################################################################################
# Plot horizontal map
################################################################################

# Choose a robust colour range: clip to a percentile so a few extreme cells
# don't wash out the rest of the map.
finite_vals = filter(isfinite, calVdown)
qhi = quantile(finite_vals, 0.99)
clim_hi = max(qhi, 1.0)   # at least 1 m so the colorbar is meaningful
colorrange_map = (0.0, clim_hi)

@info "Generating horizontal map (i,j indices for tripolar)"
flush(stdout); flush(stderr)

begin
    fig = Figure(; size = (1100, 600))
    ax = Axis(
        fig[1, 1];
        title = "Surface ventilation $field_label — $parentmodel, $time_window ($leg_tag)",
        xlabel = "i (zonal index)",
        ylabel = "j (meridional index)",
    )
    hm = heatmap!(ax, calVdown; colorrange = colorrange_map, colormap = :viridis, nan_color = :black, highclip = :magenta)
    Colorbar(fig[1, 2], hm; label = "calVdown (m)")
    outputfile = joinpath(plot_dir, "calVdown_$(leg_tag)_ij.png")
    @info "Saving $outputfile"
    save(outputfile, fig)
end

@info "Generating horizontal map (lon, lat)"
flush(stdout); flush(stderr)

begin
    fig = Figure(; size = (1100, 600))
    ax = Axis(
        fig[1, 1];
        title = "Surface ventilation $field_label — $parentmodel, $time_window ($leg_tag)",
        xlabel = "Longitude",
        ylabel = "Latitude",
    )
    # scatter with small markers to handle the tripolar (non-rectilinear) grid
    finite_mask = wet_surf .& isfinite.(calVdown)
    sc = scatter!(
        ax,
        vec(lon[finite_mask]),
        vec(lat[finite_mask]);
        color = vec(calVdown[finite_mask]),
        colorrange = colorrange_map,
        colormap = :viridis,
        markersize = 3,
        highclip = :magenta,
    )
    Colorbar(fig[1, 2], sc; label = "calVdown (m)")
    xlims!(ax, -280, 80)
    ylims!(ax, -90, 90)
    outputfile = joinpath(plot_dir, "calVdown_$(leg_tag)_lonlat.png")
    @info "Saving $outputfile"
    save(outputfile, fig)
end

@info "plot_ventilation.jl complete"
flush(stdout); flush(stderr)
