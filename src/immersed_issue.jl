# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()

using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using GLMakie

z = repeat(reshape(range(-1, stop = 0, length = 5), 1, 1, 5), 7, 2, 1)
z .+= randn(size(z)) .* 0.01 # add some noise


underlying_grid = RectilinearGrid(
    size = (7, 2, 4),
    x = (0, 1),
    y = (0, 1),
    # z = (-1, 0),
    z = z,
    topology = (Bounded, Bounded, Bounded),
)

# Chose some bottom depths to illustrate different cases
bottom = [
    -1.1 # all the cells are wet
    -1.0 # all wet
    -0.5 # half the cells are wet
    -0.2 # partial cell
    -0.0 # all dry
    +0.0 # all dry
    +0.1 # all dry
]

grid = ImmersedBoundaryGrid(
    underlying_grid, PartialCellBottom(bottom);
)

# Make a constant tracer = 0.5 (so that it looks blue on RdBu colormap)
c = CenterField(grid)
c.data .= 0.5

# Mask the immersed field (the culprit)
mask_immersed_field!(c, NaN)

# Plot the tracer
fig, ax, plt = heatmap(c, colormap = :RdBu, colorrange = (-1, 1), nan_color = :black)

# Overlay the bottom topography
Δx = grid.Δxᶜᵃᵃ[1]
for (ix, x) in enumerate(xnodes(grid, Center(), Center(), Center()))
    z = bottom[ix]
    lines!(ax, [x - Δx / 2, x + Δx / 2], [z, z]; color = :red)
    text!(ax, x, z; text = "bottom\nz = $z", color = :red, align = (:center, :bottom), offset = (0, 3))
end

# Adjust axes limits to see text
ax.limits = (0, 1, -1.2, 0.4)

fig

# save(joinpath("output", "immersed_field_heatmap.png"), fig)