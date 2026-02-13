"""
Create and save the diffusion closures to JLD2 format.

Requires: create_grid.jl to have been run first.

Usage:
    julia --project create_closures.jl
"""

@info "Loading packages and functions"

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: znode
using Adapt: adapt

using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using JLD2

# Set up architecture
if contains(ENV["HOSTNAME"], "gpu")
    using CUDA
    CUDA.set_runtime_version!(v"12.9.0"; local_toolkit = true)
    @show CUDA.versioninfo()
    arch = GPU()
else
    arch = CPU()
end
@info "Using $arch architecture"

# Configuration
parentmodel = "ACCESS-OM2-1"
# parentmodel = "ACCESS-OM2-025"
# parentmodel = "ACCESS-OM2-01"
outputdir = "/scratch/y99/TMIP/ACCESS-OM2_x_Oceananigans/output/$parentmodel"

include("tripolargrid_reader.jl")

################################################################################
# Load grid from JLD2
################################################################################

error("""
Something is wrong below. I did not check it and I think I might as well create the closures
on the fly until a PrescribedActiveTracer implementation is available
to compute the GM-Redi closure from the outputs.
Or at least a PrescribedTracer that I can use to compute my simplified horizontal/vertical closures.
""")

@info "Loading and reconstructing grid from JLD2 data"
grid_file = joinpath(outputdir, "$(parentmodel)_grid.jld2")
grid = load_tripolar_grid(grid_file)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"

################################################################################
# Create closures
################################################################################

@info "Creating closures"

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

# Vertical diffusivity parameters
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# Load MLD to add strong vertical diffusion in the mixed layer
mld_ds = open_dataset(joinpath(inputdir, "mld.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

implicit_vertical_diffusion = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization();
    κ = κVField
)
explicit_vertical_diffusion = VerticalScalarDiffusivity(
    ExplicitTimeDiscretization();
    κ = κVField
)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)

# Combine them all into a single diffusion closure
closure = (
    horizontal_diffusion,
    implicit_vertical_diffusion,
)
explicit_closure = (
    horizontal_diffusion,
    explicit_vertical_diffusion,
)

################################################################################
# Save closures to JLD2
################################################################################

@info "Saving closures to JLD2"

closures_file = joinpath(outputdir, "$(parentmodel)_closures.jld2")

# Extract κVField data
κVField_cpu = adapt(Array, κVField.data)

save(
    closures_file,
    Dict(
        "Note" => "This file was created by Benoit Pasquier (2026) from work in progress and thus comes with zero guarantees!",
        "κVML" => κVML,
        "κVBG" => κVBG,
        "κVField" => κVField_cpu,
        "κHorizontal" => 300.0,
        "closure_description" => "HorizontalScalarDiffusivity (κ=300.0) + VerticalScalarDiffusivity (κVField)",
        "explicit_closure_description" => "HorizontalScalarDiffusivity (κ=300.0) + VerticalScalarDiffusivity explicit (κVField)",
    )
)

@info "Closures saved to $(closures_file)"
