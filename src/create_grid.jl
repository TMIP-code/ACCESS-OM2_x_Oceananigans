"""
Create and save the grid to JLD2 format.

Usage:
    julia --project create_grid.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2

# Set up architecture (CPU for grid creation)
arch = CPU()
@info "Using $arch architecture"

# Configuration
parentmodel = "ACCESS-OM2-1"
# parentmodel = "ACCESS-OM2-025"
# parentmodel = "ACCESS-OM2-01"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"
mkpath(outputdir)

include("tripolargrid_reader.jl")

################################################################################
# Grid Creation
################################################################################

@info "Horizontal supergrid"

resolution_str = split(parentmodel, "-")[end]
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", "mom$(resolution_str)deg.nc")
supergrid_ds = open_dataset(supergridfile)

# Unpack supergrid data
println("Reading supergrid data into memory...")
MOMsupergrid = (;
    x = readcubedata(supergrid_ds.x).data,
    y = readcubedata(supergrid_ds.y).data,
    dx = readcubedata(supergrid_ds.dx).data,
    dy = readcubedata(supergrid_ds.dy).data,
    area = readcubedata(supergrid_ds.area).data,
    # For dimensions, use indices lengths instead of full ranges.
    # This is more consistent with Oceananigans.jl conventions for Nx, Ny, etc.
    nx = length(supergrid_ds.nx.val),
    nxp = length(supergrid_ds.nxp.val),
    ny = length(supergrid_ds.ny.val),
    nyp = length(supergrid_ds.nyp.val),
)

println("Reading vertical grid data into memory...")
@show MOM_input_vgrid_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_$(resolution_str)deg/ocean_vgrid.nc"
z_ds = open_dataset(MOM_input_vgrid_file)
zeta = -reverse(vec(z_ds["zeta"].data[1:2:end]))
z = MutableVerticalDiscretization(zeta) # from surface to bottom
Nz = length(zeta) - 1

println("Building Horizontal grid...")
underlying_grid = tripolargrid_from_supergrid(
    arch;
    MOMsupergrid...,
    halosize = (4, 4, 4),
    radius = Oceananigans.defaults.planet_radius,
    z,
    Nz,
)

@info "Vertical grid"

parentmodel_ik11path = parentmodel == "ACCESS-OM2-1" ? "access-om2" : "access-om2-$(resolution_str)"
@show MOM_input_topo_file = "/g/data/ik11/inputs/access-om2/input_20201102/mom_$(resolution_str)deg/topog.nc"
bottom_ds = open_dataset(MOM_input_topo_file)
bottom = -readcubedata(bottom_ds["depth"]).data
bottom = replace(bottom, 9999.0 => 0.0)

# Check topography
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
@show MOM_output_grid_inputdir = "/g/data/ik11/outputs/$(parentmodel_ik11path)/$experiment/output305/ocean/"
MOM_output_grid_ds = open_dataset(joinpath(MOM_output_grid_inputdir, "ocean_grid.nc"))
ht = readcubedata(MOM_output_grid_ds.ht).data
ht = replace(ht, missing => 0.0)
kmt = readcubedata(MOM_output_grid_ds.kmt).data
kbottom = round.(Union{Missing, Int}, Nz .- kmt .+ 1)

@assert ht == -bottom
for idx in eachindex(kbottom)
    local k = kbottom[idx]
    ismissing(kmt[idx]) && continue
    @assert zeta[k] ≤ bottom[idx] < zeta[k + 1]
end
@info "z coordinate/grid checks passed."

# Then immerge the grid cells with partial cells at the bottom
bottom = on_architecture(arch, bottom)
grid = ImmersedBoundaryGrid(
    underlying_grid, PartialCellBottom(bottom);
    active_cells_map = true,
    active_z_columns = true,
)

################################################################################
# Save grid to JLD2
################################################################################

@info "Saving grid to JLD2"

code_to_reconstruct_the_grid = """
    using Oceananigans
    using Adapt

    gd = load(grid_file) # gd for grid Dict
    underlying_grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
        arch,
        gd["Nx"], gd["Ny"], gd["Nz"],
        gd["Hx"], gd["Hy"], gd["Hz"],
        convert(FT, gd["Lz"]),
        on_architecture(arch, map(FT, gd["λᶜᶜᵃ"])),
        on_architecture(arch, map(FT, gd["λᶠᶜᵃ"])),
        on_architecture(arch, map(FT, gd["λᶜᶠᵃ"])),
        on_architecture(arch, map(FT, gd["λᶠᶠᵃ"])),
        on_architecture(arch, map(FT, gd["φᶜᶜᵃ"])),
        on_architecture(arch, map(FT, gd["φᶠᶜᵃ"])),
        on_architecture(arch, map(FT, gd["φᶜᶠᵃ"])),
        on_architecture(arch, map(FT, gd["φᶠᶠᵃ"])),
        on_architecture(arch, gd["z"]),
        on_architecture(arch, map(FT, gd["Δxᶜᶜᵃ"])),
        on_architecture(arch, map(FT, gd["Δxᶠᶜᵃ"])),
        on_architecture(arch, map(FT, gd["Δxᶜᶠᵃ"])),
        on_architecture(arch, map(FT, gd["Δxᶠᶠᵃ"])),
        on_architecture(arch, map(FT, gd["Δyᶜᶜᵃ"])),
        on_architecture(arch, map(FT, gd["Δyᶠᶜᵃ"])),
        on_architecture(arch, map(FT, gd["Δyᶜᶠᵃ"])),
        on_architecture(arch, map(FT, gd["Δyᶠᶠᵃ"])),
        on_architecture(arch, map(FT, gd["Azᶜᶜᵃ"])),
        on_architecture(arch, map(FT, gd["Azᶠᶜᵃ"])),
        on_architecture(arch, map(FT, gd["Azᶜᶠᵃ"])),
        on_architecture(arch, map(FT, gd["Azᶠᶠᵃ"])),
        convert(FT, gd["radius"]),
        Tripolar(gd["north_poles_latitude"], gd["first_pole_longitude"], gd["southernmost_latitude"])
    )
    grid = ImmersedBoundaryGrid(
        underlying_grid, PartialCellBottom(gd["bottom"]);
        active_cells_map = true,
        active_z_columns = true,
    )
"""

save(
    joinpath(outputdir, "$(parentmodel)_grid.jld2"),
    Dict(
        "Note" => "This file was created by Benoit Pasquier (2026) from work in progress and thus comes with zero guarantees!",
        "Nx" => underlying_grid.Nx,
        "Ny" => underlying_grid.Ny,
        "Nz" => underlying_grid.Nz,
        "Hx" => underlying_grid.Hx,
        "Hy" => underlying_grid.Hy,
        "Hz" => underlying_grid.Hz,
        "Lz" => underlying_grid.Lz,
        "λᶜᶜᵃ" => underlying_grid.λᶜᶜᵃ,
        "λᶠᶜᵃ" => underlying_grid.λᶠᶜᵃ,
        "λᶜᶠᵃ" => underlying_grid.λᶜᶠᵃ,
        "λᶠᶠᵃ" => underlying_grid.λᶠᶠᵃ,
        "φᶜᶜᵃ" => underlying_grid.φᶜᶜᵃ,
        "φᶠᶜᵃ" => underlying_grid.φᶠᶜᵃ,
        "φᶜᶠᵃ" => underlying_grid.φᶜᶠᵃ,
        "φᶠᶠᵃ" => underlying_grid.φᶠᶠᵃ,
        "Δxᶜᶜᵃ" => underlying_grid.Δxᶜᶜᵃ,
        "Δxᶠᶜᵃ" => underlying_grid.Δxᶠᶜᵃ,
        "Δxᶜᶠᵃ" => underlying_grid.Δxᶜᶠᵃ,
        "Δxᶠᶠᵃ" => underlying_grid.Δxᶠᶠᵃ,
        "Δyᶜᶜᵃ" => underlying_grid.Δyᶜᶜᵃ,
        "Δyᶠᶜᵃ" => underlying_grid.Δyᶠᶜᵃ,
        "Δyᶜᶠᵃ" => underlying_grid.Δyᶜᶠᵃ,
        "Δyᶠᶠᵃ" => underlying_grid.Δyᶠᶠᵃ,
        "Azᶜᶜᵃ" => underlying_grid.Azᶜᶜᵃ,
        "Azᶠᶜᵃ" => underlying_grid.Azᶠᶜᵃ,
        "Azᶜᶠᵃ" => underlying_grid.Azᶜᶠᵃ,
        "Azᶠᶠᵃ" => underlying_grid.Azᶠᶠᵃ,
        "z" => underlying_grid.z,
        "bottom" => bottom,
        "radius" => underlying_grid.radius,
        "north_poles_latitude" => underlying_grid.conformal_mapping.north_poles_latitude,
        "first_pole_longitude" => underlying_grid.conformal_mapping.first_pole_longitude,
        "southernmost_latitude" => underlying_grid.conformal_mapping.southernmost_latitude,
        "code_to_reconstruct_the_grid" => code_to_reconstruct_the_grid,
    )
)

@info "Grid saved to $(joinpath(outputdir, "$(parentmodel)_grid.jld2"))"
