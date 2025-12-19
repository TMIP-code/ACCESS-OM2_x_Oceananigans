# @info """
# The supergrid for ACCESS-OM2
# is a NetCDF file with 2x the resolution of the model grid
# It contains the following data:
# ncdump -h /g/data/xp65/public/apps/access_moppy_data/grids/mom1deg.nc
# netcdf mom1deg {
# dimensions:
#         string = 255 ;
#         nxp = 721 ;
#         nyp = 601 ;
#         nx = 720 ;
#         ny = 600 ;
# variables:
#         char tile(string) ;
#                 tile:standard_name = "grid_tile_spec" ;
#                 tile:tile_spec_version = "0.2" ;
#                 tile:geometry = "spherical" ;
#                 tile:discretization = "logically_rectangular" ;
#                 tile:conformal = "true" ;
#         double x(nyp, nxp) ;
#                 x:standard_name = "geographic_longitude" ;
#                 x:units = "degree_east" ;
#         double y(nyp, nxp) ;
#                 y:standard_name = "geographic_latitude" ;
#                 y:units = "degree_north" ;
#         double dx(nyp, nx) ;
#                 dx:standard_name = "grid_edge_x_distance" ;
#                 dx:units = "meters" ;
#         double dy(ny, nxp) ;
#                 dy:standard_name = "grid_edge_y_distance" ;
#                 dy:units = "meters" ;
#         double angle_dx(nyp, nxp) ;
#                 angle_dx:standard_name = "grid_vertex_x_angle_WRT_geographic_east" ;
#                 angle_dx:units = "degrees_east" ;
#         double area(ny, nx) ;
#                 area:standard_name = "grid_cell_area" ;
#                 area:units = "m2" ;
# }

# julia> supergrid_ds = open_dataset(supergridfile)
# YAXArray Dataset
# Shared Axes:
# None
# Variables with additional axes:
#   Additional Axes:
#   (↓ nx Sampled{Int64} 1:720 ForwardOrdered Regular Points,
#   → ny Sampled{Int64} 1:600 ForwardOrdered Regular Points)
#   Variables:
#   area

#   Additional Axes:
#   (↓ nxp Sampled{Int64} 1:721 ForwardOrdered Regular Points,
#   → ny Sampled{Int64} 1:600 ForwardOrdered Regular Points)
#   Variables:
#   dy

#   Additional Axes:
#   (↓ nx Sampled{Int64} 1:720 ForwardOrdered Regular Points,
#   → nyp Sampled{Int64} 1:601 ForwardOrdered Regular Points)
#   Variables:
#   dx

#   Additional Axes:
#   (↓ string Sampled{Int64} 1:255 ForwardOrdered Regular Points)
#   Variables:
#   tile

#   Additional Axes:
#   (↓ nxp Sampled{Int64} 1:721 ForwardOrdered Regular Points,
#   → nyp Sampled{Int64} 1:601 ForwardOrdered Regular Points)
#   Variables:
#   angle_dx, x, y

# This file was taken from Oceananigans.jl examples and modified
# to my testing needs.

# To run this on Gadi interactively, use

# ```
# qsub -I -P y99 -l mem=47GB -l walltime=01:00:00 -l ncpus=12 -l storage=gdata/xp65
# cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
# julia
# include("src/ACCESS-OM2_grid.jl")
# ```

# And on the GPU queue, use

# ```
# qsub -I -P y99 -l mem=47GB -q gpuvolta -l walltime=01:00:00 -l ncpus=12 -l ngpus=1 -l storage=gdata/xp65
# cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
# julia
# include("src/ACCESS-OM2_grid.jl")
# ```
# """
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()

# using YAXArrays
# using DimensionalData
# using NetCDF
# using Oceananigans


# # (from YAXArrays + NetCDF):

# model = "ACCESS-OM2-1"
# modelsupergridfile = "mom$(split(model, "-")[end])deg.nc"
# supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", modelsupergridfile)
# supergrid_ds = open_dataset(supergridfile)
# supergrid = readcubedata(supergrid_ds)
@show supergrid

include("tripolargrid_reader.jl")

# Unpack supergrid data
# TODO: I think best to extract the raw data here
# instead of passing YAXArrays
x = supergrid.x.data
y = supergrid.y.data
dx = supergrid.dx.data
dy = supergrid.dy.data
area = supergrid.area.data
# TODO: For dimensions, just get the lengths instead of index ranges
# Not sure this matters but it is a bit more consistent
# with Nx, Ny, etc. used elsewhere where "N" or "n" means number of points
nx = length(supergrid.nx.val)
nxp = length(supergrid.nxp.val)
ny = length(supergrid.ny.val)
nyp = length(supergrid.nyp.val)

grid = tripolargrid_from_supergrid(;
    x, y, dx, dy, area,
    nx, nxp, ny, nyp,
)
