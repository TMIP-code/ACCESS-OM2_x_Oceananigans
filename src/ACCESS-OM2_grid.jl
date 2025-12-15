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


# (from YAXArrays + NetCDF):

model = "ACCESS-OM2-1"
modelsupergridfile = "mom$(split(model, "-")[end])deg.nc"
supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", modelsupergridfile)
supergrid_ds = open_dataset(supergridfile)
superarea = readcubedata(supergrid_ds.area)

"""
    (; lon, lat, areacello, lon_vertices, lat_vertices) = supergrid(model; dims)

returns the longitude, latitude, area, and vertex coordinates from the model's supergrid.
"""
function supergrid(model::String; dims)

    # ACCESS-OM2-1 => mom1deg.nc
    # ACCESS-OM2-025 => mom025deg.nc
    # ACCESS-OM2-01 => mom01deg.nc
    modelsupergridfile = "mom$(split(model, "-")[end])deg.nc"
    supergridfile = joinpath("/g/data/xp65/public/apps/access_moppy_data/grids", modelsupergridfile)

    # Load data
    supergrid_ds = open_dataset(supergridfile)
    superarea = readcubedata(supergrid_ds.area)
    lon = readcubedata(supergrid_ds.x)[2:2:end, 2:2:end]
    lat = readcubedata(supergrid_ds.y)[2:2:end, 2:2:end]
    areacello = YAXArray(
        dims,
        [sum(superarea[i:(i + 1), j:(j + 1)]) for i in 1:2:size(superarea, 1) , j in 1:2:size(superarea, 2)],
        Dict("name" => "areacello", "units" => "m^2"),
    )

    # Build vertices from supergrid
    # Dimensions of vertices ar (vertex, x, y)
    # Note to self: NCO shows it as (y, x, vertex)
    SW(x) = x[1:2:(end - 2), 1:2:(end - 2)]
    SE(x) = x[3:2:end, 1:2:(end - 2)]
    NE(x) = x[3:2:end, 3:2:end]
    NW(x) = x[1:2:(end - 2), 3:2:end]
    (nx, ny) = size(lon)
    vertices(x) = [
        reshape(SW(x), (1, nx, ny))
        reshape(SE(x), (1, nx, ny))
        reshape(NE(x), (1, nx, ny))
        reshape(NW(x), (1, nx, ny))
    ]
    lon_vertices = vertices(supergrid_ds.x)
    lat_vertices = vertices(supergrid_ds.y)

    return (; lon, lat, areacello, lon_vertices, lat_vertices)
end