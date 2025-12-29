using Oceananigans.BoundaryConditions: ZipperBoundaryCondition, NoFluxBoundaryCondition, fill_halo_regions!
using Oceananigans.Fields: set!
using Oceananigans.Grids: Grids, Bounded, Flat, OrthogonalSphericalShellGrid, Periodic, RectilinearGrid, RightConnected,
    architecture, cpu_face_constructor_z, validate_dimension_specification, generate_coordinate, on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!


# @kernel function _compute_coordinates_from_supergrid!(
#         λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC,
#         x, y,
#     )

#     i, j = @index(Global, NTuple)

#     # TODO: Check these are the correct C and F indices
#     # Also, I don;t know what I'm doing here really...
#     # Is that the right way to write this kernel?
#     # Does this make sense?
#     # Does it matter if x/y are on CPU or GPU?
#     λFF[i, j] = x[2i, 2j]
#     φFF[i, j] = y[2i, 2j]
#     λFC[i, j] = x[2i, 2j + 1]
#     φFC[i, j] = y[2i, 2j + 1]
#     λCF[i, j] = x[2i + 1, 2j]
#     φCF[i, j] = y[2i + 1, 2j]
#     λCC[i, j] = x[2i + 1, 2j + 1]
#     φCC[i, j] = y[2i + 1, 2j + 1]
# end
function compute_coordinates_from_supergrid!(
        λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC,
        x, y,
    )

    for i in axes(λFF, 1), j in axes(λFF, 2)
        # TODO: Check these are the correct C and F indices
        λFF[i, j] = x[2i, 2j]
        φFF[i, j] = y[2i, 2j]
        λFC[i, j] = x[2i, 2j + 1]
        φFC[i, j] = y[2i, 2j + 1]
        λCF[i, j] = x[2i + 1, 2j]
        φCF[i, j] = y[2i + 1, 2j]
        λCC[i, j] = x[2i + 1, 2j + 1]
        φCC[i, j] = y[2i + 1, 2j + 1]
    end
    return nothing
end


# @kernel function _compue_metrics_from_supergrid!(
#         Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
#         Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
#         Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
#         dx, dy, area
#     )

#     i, j = @index(Global, NTuple)

#     @inbounds begin
#         Δxᶜᶜᵃ[i, j] = dx[2i - 1, 2j] + dx[2i, 2j]
#         Δxᶠᶜᵃ[i, j] = dx[2i - 2, 2j] + dx[2i - 1, 2j]
#         Δxᶜᶠᵃ[i, j] = dx[2i - 1, 2j - 1] + dx[2i, 2j - 1]
#         Δxᶠᶠᵃ[i, j] = dx[2i - 2, 2j - 1] + dx[2i - 1, 2j - 1]


#     end
# end
function compute_metrics_from_supergrid!(
        Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
        Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
        Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
        nx, ny, dx, dy, area
    )

    for i in axes(Δxᶜᶜᵃ, 1), j in axes(Δxᶜᶜᵃ, 2)
        𝑖, 𝑗 = 2i, 2j
        # For Δx, wrap x indices around with mod1(𝑖 - 2, nx)
        Δxᶜᶜᵃ[i, j] = dx[𝑖 - 1, 𝑗] + dx[𝑖, 𝑗]
        Δxᶠᶜᵃ[i, j] = dx[mod1(𝑖 - 2, nx), 𝑗] + dx[𝑖 - 1, 𝑗]
        Δxᶜᶠᵃ[i, j] = dx[𝑖 - 1, 𝑗 - 1] + dx[𝑖, 𝑗 - 1]
        Δxᶠᶠᵃ[i, j] = dx[mod1(𝑖 - 2, nx), 𝑗 - 1] + dx[𝑖 - 1, 𝑗 - 1]
        # For Δy, repeat last row for south boundary with max(𝑗 - 2, 1)
        Δyᶜᶜᵃ[i, j] = dy[𝑖, 𝑗 - 1] + dy[𝑖, 𝑗]
        Δyᶠᶜᵃ[i, j] = dy[𝑖 - 1, 𝑗 - 1] + dy[𝑖 - 1, 𝑗]
        Δyᶜᶠᵃ[i, j] = dy[𝑖, max(𝑗 - 2, 1)] + dy[𝑖, 𝑗 - 1]
        Δyᶠᶠᵃ[i, j] = dy[𝑖 - 1, max(𝑗 - 2, 1)] + dy[𝑖 - 1, 𝑗 - 1]
        # For area use the same logic as above
        Azᶜᶜᵃ[i, j] = area[𝑖 - 1, 𝑗 - 1] + area[𝑖 - 1, 𝑗] + area[𝑖, 𝑗 - 1] + area[𝑖, 𝑗]
        Azᶠᶜᵃ[i, j] = area[mod1(𝑖 - 2, nx), 𝑗 - 1] + area[mod1(𝑖 - 2, nx), 𝑗] + area[𝑖 - 1, 𝑗 - 1] + area[𝑖 - 1, 𝑗]
        Azᶜᶠᵃ[i, j] = area[𝑖 - 1, max(𝑗 - 2, 1)] + area[𝑖 - 1, 𝑗 - 1] + area[𝑖, max(𝑗 - 2, 1)] + area[𝑖, 𝑗 - 1]
        Azᶠᶠᵃ[i, j] = area[mod1(𝑖 - 2, nx), max(𝑗 - 2, 1)] + area[mod1(𝑖 - 2, nx), 𝑗 - 1] + area[𝑖 - 1, max(𝑗 - 2, 1)] + area[𝑖 - 1, 𝑗 - 1]
    end
    return
end


function tripolargrid_from_supergrid(
        arch = CPU(), FT::DataType = Float64;
        x, y, dx, dy, area,
        nx, nxp, ny, nyp,
        halosize = (4, 4, 4),
        radius = Oceananigans.defaults.planet_radius,
        z = (0, 1), # Maybe z can be 3D array here?
        Nz = 1,
        # north_poles_latitude = 55,
        # first_pole_longitude = 70,
    )  # second pole is at longitude `first_pole_longitude + 180ᵒ`

    @show southernmost_latitude = minimum(y)
    @show latitude = (southernmost_latitude, 90)
    @show longitude = (minimum(x), maximum(x))
    max_latitudes = maximum(y, dims = 2)
    @show north_poles_latitude, i_north_pole = findmin(max_latitudes)
    @show first_pole_longitude = x[i_north_pole, 1]

    # Horizontal grid size
    Nλ, Nφ = nx ÷ 2, ny ÷ 2

    # Halo size
    Hλ, Hφ, Hz = halosize
    gridsize = (Nλ, Nφ, Nz)

    if isodd(Nλ)
        throw(ArgumentError("The number of cells in the longitude dimension should be even!"))
    end

    # For z use the same as Oceananigans TripolarGrid
    topology = (Periodic, RightConnected, Bounded)
    TZ = topology[3]
    z = validate_dimension_specification(TZ, z, :z, Nz, FT)
    Lz, z = generate_coordinate(FT, topology, gridsize, halosize, z, :z, 3, CPU())

    λFF = zeros(Nλ, Nφ)
    φFF = zeros(Nλ, Nφ)
    λFC = zeros(Nλ, Nφ)
    φFC = zeros(Nλ, Nφ)

    λCF = zeros(Nλ, Nφ)
    φCF = zeros(Nλ, Nφ)
    λCC = zeros(Nλ, Nφ)
    φCC = zeros(Nλ, Nφ)

    compute_coordinates_from_supergrid!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC, x, y)
    # If it works switch to Kernel as below?
    # loop! = _compute_tripolar_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    # loop!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC, x, y)

    # Helper grid to fill halosize
    Nx = Nλ
    Ny = Nφ
    grid = RectilinearGrid(;
        size = (Nx, Ny),
        halo = (Hλ, Hφ),
        x = (0, 1), y = (0, 1),
        topology = (Periodic, RightConnected, Flat),
    )

    # Boundary conditions to fill halos of the coordinate and metric terms
    # We need to define them manually because of the convention in the
    # ZipperBoundaryCondition that edge fields need to switch sign (which we definitely do not
    # want for coordinates and metrics)
    default_boundary_conditions = FieldBoundaryConditions(
        north = ZipperBoundaryCondition(),
        south = NoFluxBoundaryCondition(), # The south should be `continued`
        west = Oceananigans.PeriodicBoundaryCondition(),
        east = Oceananigans.PeriodicBoundaryCondition(),
        top = nothing,
        bottom = nothing
    )

    lFF = Field{Face, Face, Center}(grid; boundary_conditions = default_boundary_conditions)
    pFF = Field{Face, Face, Center}(grid; boundary_conditions = default_boundary_conditions)

    lFC = Field{Face, Center, Center}(grid; boundary_conditions = default_boundary_conditions)
    pFC = Field{Face, Center, Center}(grid; boundary_conditions = default_boundary_conditions)

    lCF = Field{Center, Face, Center}(grid; boundary_conditions = default_boundary_conditions)
    pCF = Field{Center, Face, Center}(grid; boundary_conditions = default_boundary_conditions)

    lCC = Field{Center, Center, Center}(grid; boundary_conditions = default_boundary_conditions)
    pCC = Field{Center, Center, Center}(grid; boundary_conditions = default_boundary_conditions)

    set!(lFF, λFF)
    set!(pFF, φFF)

    set!(lFC, λFC)
    set!(pFC, φFC)

    set!(lCF, λCF)
    set!(pCF, φCF)

    set!(lCC, λCC)
    set!(pCC, φCC)

    fill_halo_regions!(lFF)
    fill_halo_regions!(lCF)
    fill_halo_regions!(lFC)
    fill_halo_regions!(lCC)

    fill_halo_regions!(pFF)
    fill_halo_regions!(pCF)
    fill_halo_regions!(pFC)
    fill_halo_regions!(pCC)

    # Coordinates
    λᶠᶠᵃ = dropdims(lFF.data, dims = 3)
    φᶠᶠᵃ = dropdims(pFF.data, dims = 3)

    λᶠᶜᵃ = dropdims(lFC.data, dims = 3)
    φᶠᶜᵃ = dropdims(pFC.data, dims = 3)

    λᶜᶠᵃ = dropdims(lCF.data, dims = 3)
    φᶜᶠᵃ = dropdims(pCF.data, dims = 3)

    λᶜᶜᵃ = dropdims(lCC.data, dims = 3)
    φᶜᶜᵃ = dropdims(pCC.data, dims = 3)

    # Read Metrics
    # TODO: check these are the correct indices
    # dx and dy are the lengths of the edges of the supergrid
    # so need to sum them to get the Δx and Δy
    # Same for area (need to sum 2x2)
    # But I need to add one row and one column to the left.
    dx_west = dx[end, :]
    dx_east = dx[1, :]
    dy_south = dy[:, end]
    area_west = area[end, :]
    area_south = area[:, end]
    area_southwest = area[end, end]

    # TODO: Maybe this can be made faster?
    # TODO: Check if the metrics and area are correct at boundaries
    # TODO: make these on_architecture(arch, zeros(Nx, Ny))
    # to build the grid on GPU
    Δxᶜᶜᵃ = zeros(Nx, Ny)
    Δxᶠᶜᵃ = zeros(Nx, Ny)
    Δxᶜᶠᵃ = zeros(Nx, Ny)
    Δxᶠᶠᵃ = zeros(Nx, Ny)

    Δyᶜᶜᵃ = zeros(Nx, Ny)
    Δyᶠᶜᵃ = zeros(Nx, Ny)
    Δyᶜᶠᵃ = zeros(Nx, Ny)
    Δyᶠᶠᵃ = zeros(Nx, Ny)

    Azᶜᶜᵃ = zeros(Nx, Ny)
    Azᶠᶜᵃ = zeros(Nx, Ny)
    Azᶜᶠᵃ = zeros(Nx, Ny)
    Azᶠᶠᵃ = zeros(Nx, Ny)

    compute_metrics_from_supergrid!(
        Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
        Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
        Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
        nx, ny, dx, dy, area
    )

    # Metrics fields to fill halos
    FF = Field{Face, Face, Center}(grid; boundary_conditions = default_boundary_conditions)
    FC = Field{Face, Center, Center}(grid; boundary_conditions = default_boundary_conditions)
    CF = Field{Center, Face, Center}(grid; boundary_conditions = default_boundary_conditions)
    CC = Field{Center, Center, Center}(grid; boundary_conditions = default_boundary_conditions)

    # Fill all periodic halos
    set!(FF, Δxᶠᶠᵃ)
    set!(CF, Δxᶜᶠᵃ)
    set!(FC, Δxᶠᶜᵃ)
    set!(CC, Δxᶜᶜᵃ)
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Δxᶠᶠᵃ = deepcopy(dropdims(FF.data, dims = 3))
    Δxᶜᶠᵃ = deepcopy(dropdims(CF.data, dims = 3))
    Δxᶠᶜᵃ = deepcopy(dropdims(FC.data, dims = 3))
    Δxᶜᶜᵃ = deepcopy(dropdims(CC.data, dims = 3))

    set!(FF, Δyᶠᶠᵃ)
    set!(CF, Δyᶜᶠᵃ)
    set!(FC, Δyᶠᶜᵃ)
    set!(CC, Δyᶜᶜᵃ)
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Δyᶠᶠᵃ = deepcopy(dropdims(FF.data, dims = 3))
    Δyᶜᶠᵃ = deepcopy(dropdims(CF.data, dims = 3))
    Δyᶠᶜᵃ = deepcopy(dropdims(FC.data, dims = 3))
    Δyᶜᶜᵃ = deepcopy(dropdims(CC.data, dims = 3))

    set!(FF, Azᶠᶠᵃ)
    set!(CF, Azᶜᶠᵃ)
    set!(FC, Azᶠᶜᵃ)
    set!(CC, Azᶜᶜᵃ)
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Azᶠᶠᵃ = deepcopy(dropdims(FF.data, dims = 3))
    Azᶜᶠᵃ = deepcopy(dropdims(CF.data, dims = 3))
    Azᶠᶜᵃ = deepcopy(dropdims(FC.data, dims = 3))
    Azᶜᶜᵃ = deepcopy(dropdims(CC.data, dims = 3))

    Hx, Hy, Hz = halosize

    # TODO: Check if longitude below is correct.
    # I recreated longitude = (-180, 180) by hand here, as it does not seem to be used anywhere else
    # and I assume this is only used to conitnue the Δ metrics south, which should not depend on latitude
    # (unless the South pole is also shifted like in some models?)
    latitude_longitude_grid = LatitudeLongitudeGrid(;
        size = gridsize,
        latitude,
        longitude = (-180, 180),
        halo = halosize,
        z = (0, 1), # z does not really matter here
        radius
    )

    # Continue the metrics to the south with the LatitudeLongitudeGrid
    # metrics (probably we don't even need to do this, since the tripolar grid should
    # terminate below Antartica, but it's better to be safe)
    continue_south!(Δxᶠᶠᵃ, latitude_longitude_grid.Δxᶠᶠᵃ)
    continue_south!(Δxᶠᶜᵃ, latitude_longitude_grid.Δxᶠᶜᵃ)
    continue_south!(Δxᶜᶠᵃ, latitude_longitude_grid.Δxᶜᶠᵃ)
    continue_south!(Δxᶜᶜᵃ, latitude_longitude_grid.Δxᶜᶜᵃ)

    continue_south!(Δyᶠᶠᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶠᶜᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶜᶠᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)
    continue_south!(Δyᶜᶜᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)

    continue_south!(Azᶠᶠᵃ, latitude_longitude_grid.Azᶠᶠᵃ)
    continue_south!(Azᶠᶜᵃ, latitude_longitude_grid.Azᶠᶜᵃ)
    continue_south!(Azᶜᶠᵃ, latitude_longitude_grid.Azᶜᶠᵃ)
    continue_south!(Azᶜᶜᵃ, latitude_longitude_grid.Azᶜᶜᵃ)

    # Final grid with correct metrics
    # TODO: remove `on_architecture(arch, ...)` when we shift grid construction to GPU
    grid = OrthogonalSphericalShellGrid{Periodic, RightConnected, Bounded}(
        arch,
        Nx, Ny, Nz,
        Hx, Hy, Hz,
        convert(FT, Lz),
        on_architecture(arch, map(FT, λᶜᶜᵃ)),
        on_architecture(arch, map(FT, λᶠᶜᵃ)),
        on_architecture(arch, map(FT, λᶜᶠᵃ)),
        on_architecture(arch, map(FT, λᶠᶠᵃ)),
        on_architecture(arch, map(FT, φᶜᶜᵃ)),
        on_architecture(arch, map(FT, φᶠᶜᵃ)),
        on_architecture(arch, map(FT, φᶜᶠᵃ)),
        on_architecture(arch, map(FT, φᶠᶠᵃ)),
        on_architecture(arch, z),
        on_architecture(arch, map(FT, Δxᶜᶜᵃ)),
        on_architecture(arch, map(FT, Δxᶠᶜᵃ)),
        on_architecture(arch, map(FT, Δxᶜᶠᵃ)),
        on_architecture(arch, map(FT, Δxᶠᶠᵃ)),
        on_architecture(arch, map(FT, Δyᶜᶜᵃ)),
        on_architecture(arch, map(FT, Δyᶠᶜᵃ)),
        on_architecture(arch, map(FT, Δyᶜᶠᵃ)),
        on_architecture(arch, map(FT, Δyᶠᶠᵃ)),
        on_architecture(arch, map(FT, Azᶜᶜᵃ)),
        on_architecture(arch, map(FT, Azᶠᶜᵃ)),
        on_architecture(arch, map(FT, Azᶜᶠᵃ)),
        on_architecture(arch, map(FT, Azᶠᶠᵃ)),
        convert(FT, radius),
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude)
    )

    return grid
end


"""
Merge the cells that touch the north fold to make it an T-point pivot fold.

So the last row must be extended by copying values from the opposite side:

P---j---k---l---m---n---o---p---P <- fold
|   |   |   |   |   |   |   |   |
| - C - | - C - | - C - | - C - | <- Centers
|   |   |   |   |   |   |   |   |
a---b---c---d---e---f---g---h---i

becomes

i---h---g---f---e---d---c---b---a <- new coordinates = reversed from south edge
|       |       |       |       |
|   |   |   |   |   |   |   |   |
|       |       |       |       |
P - C - + - C - P - C - + - C - P <- fold = Centers now!
|       |       |       |       |
|   |   |   |   |   |   |   |   |
|       |       |       |       |
a---b---c---d---e---f---g---h---i <- unchanged
"""
function convert_Fpointpivot_to_Tpointpivot(; x, y, dx, dy, area, nx, nxp, ny, nyp)
    for i in 1:nxp
        x[i, nyp - 1] = x[i, nyp]
        x[i, nyp] = x[nxp - i + 1, nyp - 2]
        y[i, nyp - 1] = y[i, nyp]
        y[i, nyp] = y[nxp - i + 1, nyp - 2]
        dy[i, ny - 1] = dy[i, ny - 1] + dy[i, ny]
        dy[i, ny] = dy[nxp - i + 1, ny - 1]
    end
    for i in 1:nx
        dx[i, nyp - 1] = dx[i, nyp]
        dx[i, nyp] = dx[nx - i + 1, nyp - 2]
        area[i, ny - 1] = area[i, ny - 1] + area[i, ny]
        area[i, ny] = area[nx - i + 1, ny - 1]
    end
    return (; x, y, dx, dy, area, nx, nxp, ny, nyp)
end

"""
Places u or v data on the Oceananigans B-grid from MOM output.

It shifts the data from the NE corners (MOM convention)
to the SW corners (Oceananigans convention).
It also flips the vertical coordinate.
j = 1 row is set to zero (both u and v).
i = 1 column is set by wrapping around the data (periodic longitude).
"""
function Bgrid_velocity_from_MOM(grid, data)
    north_bc = Oceananigans.BoundaryCondition(Oceananigans.BoundaryConditions.Zipper(), -1)
    bcs = FieldBoundaryConditions(grid, (Face(), Face(), Center()), north = north_bc)
    x = Field{Face, Face, Center}(grid; boundary_conditions = bcs)
    Nx, Ny, Nz = size(grid)
    x.data[2:Nx, 2:Ny, 1:Nz] .= data[1:end-1, 1:end-1, Nz:-1:1]
    x.data[1:Nx, 1, 1:Nz] .= 0 # TODO Maybe remove if zero is the default on creation
    x.data[1, 2:Ny, 1:Nz] .= data[end, 1:end-1, Nz:-1:1]
    Oceananigans.BoundaryConditions.fill_halo_regions!(x)
    return x
end
