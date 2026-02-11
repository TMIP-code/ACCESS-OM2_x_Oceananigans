
# using Oceananigans
# # Comment/uncomment the following lines to enable/disable GPU
# if contains(ENV["HOSTNAME"], "gpu")
#     using CUDA
#     CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
#     @show CUDA.versioninfo()
#     arch = GPU()
# else
#     arch = CPU()
# end
# @info "Using $arch architecture"

# using Oceananigans.Grids: Bounded, OrthogonalSphericalShellGrid, Periodic, RightFaceFolded, on_architecture
# using Oceananigans.OrthogonalSphericalShellGrids: Tripolar

# using JLD2

# parentmodel = "ACCESS-OM2-1"
# # parentmodel = "ACCESS-OM2-025"
# # parentmodel = "ACCESS-OM2-01"
# outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"

# grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")
# gd = load(grid_file) # gd for grid Dict
# FT = Float64
# underlying_grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
#     arch,
#     gd["Nx"], gd["Ny"], gd["Nz"],
#     gd["Hx"], gd["Hy"], gd["Hz"],
#     convert(FT, gd["Lz"]),
#     on_architecture(arch, map(FT, gd["λᶜᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["λᶠᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["λᶜᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["λᶠᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["φᶜᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["φᶠᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["φᶜᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["φᶠᶠᵃ"])),
#     on_architecture(arch, gd["z"]),
#     on_architecture(arch, map(FT, gd["Δxᶜᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["Δxᶠᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["Δxᶜᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["Δxᶠᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["Δyᶜᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["Δyᶠᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["Δyᶜᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["Δyᶠᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["Azᶜᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["Azᶠᶜᵃ"])),
#     on_architecture(arch, map(FT, gd["Azᶜᶠᵃ"])),
#     on_architecture(arch, map(FT, gd["Azᶠᶠᵃ"])),
#     convert(FT, gd["radius"]),
#     # TODO: this mapping to Tripolar should be replaced with a custom one
#     Tripolar(gd["north_poles_latitude"], gd["first_pole_longitude"], gd["southernmost_latitude"])
# )
# grid = ImmersedBoundaryGrid(
#     underlying_grid, PartialCellBottom(gd["bottom"]);
#     active_cells_map = true,
#     active_z_columns = true,
# )

using Oceananigans.Operators: Δzᶜᶜᶜ

Δz3D = [Δzᶜᶜᶜ(Tuple(I)..., grid) for I in CartesianIndices((grid.Nx, grid.Ny, grid.Nz))]

using Oceananigans.Grids: zspacings

Δz = Field(zspacings(grid, Center(), Center(), Center()))
compute!(Δz)

Δz3D = interior(Δz)

Δz1D = diff(grid.z.cᵃᵃᶠ[1:Nz + 1])

@show extrema(Δz3D ./ reshape(Δz1D, 1, 1, :))

Δz3D_jan = dht[:, :, Nz:-1:1, 1] |> Array

fNaN = CenterField(grid)
mask_immersed_field!(fNaN, NaN)
wet3D = .!isnan.(interior(fNaN))
idx = findall(wet3D)
Nidx = length(idx)

@show extrema(Δz3D_jan[idx] ./ Δz3D[idx])

using CairoMakie

fig, ax, plt = hist(100 * Δz3D_jan[idx] ./ Δz3D[idx]; bins = 80:0.1:120)
save(joinpath(outputdir, "dzstar3D_jan_vs_dz3D_$(parentmodel).png"), fig)