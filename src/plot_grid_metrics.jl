"""
Plot all grid coordinates and metrics as heatmaps (with halos).

Loads the grid from the preprocessed grid.jld2 and plots each metric
using the `plot_metric` function from the Oceananigans tripolar grid
validation script.

Usage:
    julia --project src/plot_grid_metrics.jl
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: RightFaceFolded, RightCenterFolded, topology
using CairoMakie
using JLD2

include("shared_functions.jl")

################################################################################
# plot_metric (from Oceananigans validation/orthogonal_spherical_shell_grid/tripolargrid.jl)
################################################################################

"""Determine Location from 3 characters at the end?"""
function celllocation(char::Char)
    return char == 'ᶜ' ? Center :
        char == 'ᶠ' ? Face :
        char == 'ᵃ' ? Center :
        throw(ArgumentError("Unknown cell location character: $char"))
end
function celllocation(str::String)
    N = ncodeunits(str)
    iz = prevind(str, N)
    z = celllocation(str[iz])
    iy = prevind(str, iz)
    y = celllocation(str[iy])
    ix = prevind(str, iy)
    x = celllocation(str[ix])
    return (x, y, z)
end
celllocation(sym::Symbol) = celllocation(String(sym))

"""
    plot_metric(grid, metric_symbol; prefix = "")

Plots a heatmap of the given metric and saves it to a PNG file.
Adds a polygon representing the interior points.
"""
function plot_metric(grid, metric_symbol; prefix = "")
    xdata = getproperty(grid, metric_symbol)
    (Hx, Hy) = .-xdata.offsets
    (Nx, Ny) = Base.size(xdata) .- 2 .* (Hx, Hy)
    # location-referenced pivot point indices
    c_pivot_i = v_pivot_i = grid.Nx ÷ 2 + 0.5
    u_pivot_i = grid.Nx ÷ 2 + 1
    c_pivot_j = u_pivot_j = (topology(grid, 2) == RightCenterFolded) ? grid.Ny : grid.Ny + 0.5
    v_pivot_j = c_pivot_j + 0.5
    loc = celllocation(metric_symbol)
    pivot_i, pivot_j = if loc == (Center, Center, Center)
        c_pivot_i, c_pivot_j
    elseif loc == (Face, Face, Center)
        u_pivot_i, v_pivot_j
    elseif loc == (Center, Face, Center)
        c_pivot_i, v_pivot_j
    elseif loc == (Face, Center, Center)
        u_pivot_i, c_pivot_j
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "i",
        ylabel = "j",
        aspect = DataAspect(),
        xticks = [1, Nx],
        yticks = [1, Ny],
    )
    extraopt = (; nan_color = :gray)
    hm = heatmap!(ax, (1 - Hx):(Nx + Hx), (1 - Hy):(Ny + Hy), xdata[:, :].parent; extraopt...)
    scatter!(ax, [pivot_i], [pivot_j]; color = :red, marker = :star5, markersize = 15)
    ax.title = "$metric_symbol"
    Colorbar(fig[2, 1], hm; vertical = false, tellwidth = false)
    save("$(prefix)_$(metric_symbol).png", fig)
    return fig
end

################################################################################
# Load grid and plot
################################################################################

(; parentmodel, experiment_dir) = load_project_config()
grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading grid from $grid_file"
flush(stdout); flush(stderr)

grid = load_tripolar_grid(grid_file, CPU())
ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid

@info "Grid: Nx=$(ug.Nx), Ny=$(ug.Ny), Nz=$(ug.Nz), Hx=$(ug.Hx), Hy=$(ug.Hy)"
flush(stdout); flush(stderr)

# Output directory
plot_dir = joinpath(experiment_dir, "plots", "grid_metrics")
mkpath(plot_dir)
prefix = joinpath(plot_dir, "$(parentmodel)")

metrics = (
    :λᶜᶜᵃ, :φᶜᶜᵃ, :Δxᶜᶜᵃ, :Δyᶜᶜᵃ, :Azᶜᶜᵃ,
    :λᶜᶠᵃ, :φᶜᶠᵃ, :Δxᶜᶠᵃ, :Δyᶜᶠᵃ, :Azᶜᶠᵃ,
    :λᶠᶜᵃ, :φᶠᶜᵃ, :Δxᶠᶜᵃ, :Δyᶠᶜᵃ, :Azᶠᶜᵃ,
    :λᶠᶠᵃ, :φᶠᶠᵃ, :Δxᶠᶠᵃ, :Δyᶠᶠᵃ, :Azᶠᶠᵃ,
)

for metric in metrics
    @info "Plotting $metric"
    flush(stdout)
    plot_metric(ug, metric; prefix)
end

@info "All plots saved to $plot_dir"
flush(stdout); flush(stderr)
