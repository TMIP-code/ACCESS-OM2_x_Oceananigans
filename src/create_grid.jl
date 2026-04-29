"""
Create and save the grid to JLD2 format.

Usage:
    julia --project create_grid.jl
"""

@info "Loading packages and functions"
flush(stdout); flush(stderr)

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
using YAML

# Set up architecture (CPU for grid creation)
arch = CPU()
@info "Using $arch architecture"
flush(stdout); flush(stderr)

# Configuration
include("shared_functions.jl")
(; parentmodel, experiment, experiment_dir) = load_project_config()
mkpath(experiment_dir)

parentsimulation = experiment
@show parentsimulation
configs = YAML.load_file("ACCESS-OM2_configs.yaml")
@show config_path = configs[parentsimulation]
config = YAML.load_file(config_path)

# Find ocean submodel input directories
# (some configs list multiple input dirs for the ocean submodel)
ocean_submodel = only(filter(s -> s["name"] == "ocean", config["submodels"]))
ocean_input_dirs = ocean_submodel["input"] isa AbstractVector ? ocean_submodel["input"] : [ocean_submodel["input"]]

# Helper to find a file across the ocean input directories
function find_in_inputs(filename, dirs)
    for d in dirs
        f = joinpath(d, filename)
        isfile(f) && return f
    end
    error("$filename not found in input directories: $dirs")
end

################################################################################
# Grid Creation
################################################################################

@info "Horizontal supergrid"
flush(stdout); flush(stderr)

resolution_str = split(parentmodel, "-")[end]
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", "mom$(resolution_str)deg.nc")
supergrid_ds = open_dataset(supergridfile)

# Unpack supergrid data
println("Reading supergrid data into memory...")
flush(stdout); flush(stderr)
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
@show MOM_input_vgrid_file = find_in_inputs("ocean_vgrid.nc", ocean_input_dirs)
flush(stdout); flush(stderr)
z_ds = open_dataset(MOM_input_vgrid_file)
zeta = -reverse(vec(z_ds["zeta"].data[1:2:end]))
z = MutableVerticalDiscretization(zeta) # from surface to bottom
Nz = length(zeta) - 1

println("Building Horizontal grid...")
flush(stdout); flush(stderr)
Hx = parse(Int, get(ENV, "GRID_HX", "7"))
Hy = parse(Int, get(ENV, "GRID_HY", "7"))
Hz = parse(Int, get(ENV, "GRID_HZ", "7"))
@info "Grid halo size: ($Hx, $Hy, $Hz)"
underlying_grid = tripolargrid_from_supergrid(
    arch;
    MOMsupergrid...,
    halosize = (Hx, Hy, Hz),
    radius = Oceananigans.defaults.planet_radius,
    z,
    Nz,
)

@info "Vertical grid"
flush(stdout); flush(stderr)

@show MOM_input_topo_file = find_in_inputs("topog.nc", ocean_input_dirs)
flush(stdout); flush(stderr)
bottom_ds = open_dataset(MOM_input_topo_file)
bottom = -readcubedata(bottom_ds["depth"]).data
# Land cells appear either as the literal sentinel 9999.0 (older topog.nc) or
# as `missing` (newer files where _FillValue is decoded by YAXArrays, e.g.
# 01deg_jra55v140_iaf cycles). Handle both, then strip the Missing union so
# downstream `set!` into Float64 fields succeeds.
bottom = Float64.(replace(bottom, 9999.0 => 0.0, missing => 0.0))

# Cross-check our topography against the ocean_grid.nc that MOM writes at
# runtime. This is *only* a sanity check — `ht`/`kmt` aren't used to build
# the Oceananigans grid, only to assert consistency with what the parent
# simulation actually saw. Some experiments (e.g. 01deg_jra55v140_iaf) don't
# save ocean_grid.nc, so the check is opt-in via CHECK_AGAINST_PARENT_GRID_OUTPUT.
check_parent_grid = lowercase(get(ENV, "CHECK_AGAINST_PARENT_GRID_OUTPUT", "no")) ∈ ("yes", "true", "1")
if check_parent_grid
    @show MOM_output_grid_file = joinpath(dirname(config_path), "ocean", "ocean_grid.nc")
    flush(stdout); flush(stderr)
    isfile(MOM_output_grid_file) || error(
        "CHECK_AGAINST_PARENT_GRID_OUTPUT=yes, but cannot find the parent grid " *
            "output at $MOM_output_grid_file. Either the parent simulation did not " *
            "save it (e.g. 01deg_jra55v140_iaf cycles), or the path is wrong. " *
            "Unset CHECK_AGAINST_PARENT_GRID_OUTPUT (default `no`) to skip this check.",
    )
    MOM_output_grid_ds = open_dataset(MOM_output_grid_file)
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
    @info "z coordinate/grid checks against parent grid output passed."
else
    @info "Skipping z coordinate/grid checks against parent grid output " *
        "(set CHECK_AGAINST_PARENT_GRID_OUTPUT=yes to enable)."
end
flush(stdout); flush(stderr)

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

# TODO turn this into a function and place it in shared_functions.jl,
# and then here just call that function to save the grid.
# TODO: Maybe someone can write a NetCDF writer/reader for OrthogonalSphericalShellGrid

@info "Saving grid to JLD2"
flush(stdout); flush(stderr)

code_to_reconstruct_the_grid = """
    # See src/shared_utils/grid.jl for load_tripolar_grid and build_underlying_grid.
    # Usage:
    #   include("src/shared_functions.jl")
    #   grid = load_tripolar_grid(grid_file, arch)
    # This handles both serial and distributed architectures, using the
    # pre-computed coordinate/metric arrays saved in this JLD2 file.
"""

grid_file = joinpath(experiment_dir, "grid.jld2")

save(
    joinpath(grid_file),
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
        "z_faces" => Array(underlying_grid.z.cᵃᵃᶠ[1:(underlying_grid.Nz + 1)]),
        "bottom" => bottom,
        "radius" => underlying_grid.radius,
        "north_poles_latitude" => underlying_grid.conformal_mapping.north_poles_latitude,
        "first_pole_longitude" => underlying_grid.conformal_mapping.first_pole_longitude,
        "southernmost_latitude" => underlying_grid.conformal_mapping.southernmost_latitude,
        "code_to_reconstruct_the_grid" => code_to_reconstruct_the_grid,
    )
)

@info "Grid saved to $grid_file"
flush(stdout); flush(stderr)
