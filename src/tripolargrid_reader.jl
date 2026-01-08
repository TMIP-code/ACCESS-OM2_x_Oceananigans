using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, NoFluxBoundaryCondition, fill_halo_regions!
using Oceananigans.Fields: set!
using Oceananigans.Grids: Grids, Bounded, Flat, OrthogonalSphericalShellGrid, Periodic, RectilinearGrid, RightFaceConnected,
    architecture, cpu_face_constructor_z, validate_dimension_specification, generate_coordinate, on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!


@kernel function compute_coordinates_and_metrics_from_supergrid!(
        λFF, λFC, λCF, λCC,     # TripolarGrid longitude coordinates
        φFF, φFC, φCF, φCC,     # TripolarGrid latitude coordinates
        ΔxFF, ΔxFC, ΔxCF, ΔxCC, # TripolarGrid x distances
        ΔyFF, ΔyFC, ΔyCF, ΔyCC, # TripolarGrid y distances
        AzFF, AzFC, AzCF, AzCC, # TripolarGrid areas
        x, y,   # supergrid coordinates
        dx, dy, # supergrid distances
        area,   # supergrid areas
        nx      # supergrid size in x (nx == 2 * Nx)
    )

    # Note this kernel will fills halos a bit sometimes.
    # That's because size varies with location and topology,
    # e.g., λCC has size (Nx, Ny) but λFF has size (Nx, Ny + 1).
    # But that's OK because we fill halos again later.

    i, j = @index(Global, NTuple)

    # For λ we just copy from the super grid incrementing by 2 in each direction.
    # For λCC, of size (Nx, Ny), we have:
    #
    #                       ┏━━━━━━┯━━━━━━┳━━━━━━┯━━━━━━┓
    #                       ┃      │      ┃      │      ┃
    #                       ┃      │      ┃      │      ┃
    #  j = 2, 𝑗 = 2j = 4 ─▶ ┠──────┼──────╂──────┼──────┨
    #                       ┃      │      ┃      │      ┃
    #                       ┃      │      ┃      │      ┃
    #                       ┣━━━━━━┿━━━━━━╋━━━━━━┿━━━━━━┫
    #                       ┃      │      ┃      │      ┃
    #                       ┃      │      ┃      │      ┃
    #  j = 1, 𝑗 = 2j = 2 ─▶ u ──── c ─────╂──────┼──────┨
    #                       ┃      │      ┃      │      ┃
    #                       ┃      │      ┃      │      ┃
    #                       ┗━━━━━ v ━━━━━┻━━━━━━┷━━━━━━┛
    #                              ▲             ▲
    #                            i = 1         i = 2
    #                         𝑖 = 2i = 2     𝑖 = 2i = 4
    #
    #
    # And for λFF, size (Nx, Ny + 1):
    #
    #  j = 3, 𝑗 = 2j - 1 = 5 ─▶ ┏━━━━━━┯━━━━━━┳━━━━━━┯━━━━━━┓
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #                           ┠──────┼──────╂──────┼──────┨
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #  j = 2, 𝑗 = 2j - 1 = 3 ─▶ ┣━━━━━━┿━━━━━━╋━━━━━━┿━━━━━━┫
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #                           u ──── c ─────╂──────┼──────┨
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #  j = 1, 𝑗 = 2j - 1 = 1 ─▶ ┗━━━━━ v ━━━━━┻━━━━━━┷━━━━━━┛
    #                           ▲             ▲
    #                         i = 1         i = 2
    #                     𝑖 = 2i - 1 = 1    𝑖 = 2i - 1 = 3
    #
    # Note that this kernel will try to fill λCC at index j = Ny + 1 (j = 3) above,
    # which is the halo region. That's OK because the halos will be filled in,
    # but that means the supergrid must be extended by one row for this to work!

    λFF[i, j] = x[2i - 1, 2j - 1]
    λFC[i, j] = x[2i - 1, 2j]
    λCF[i, j] = x[2i, 2j - 1]
    λCC[i, j] = x[2i, 2j]

    # Ditto for φ
    φFF[i, j] = y[2i - 1, 2j - 1]
    φFC[i, j] = y[2i - 1, 2j]
    φCF[i, j] = y[2i, 2j - 1]
    φCC[i, j] = y[2i, 2j]

    # For Δx, I need to sum consecutive dx 2 by 2,
    # and sometimes wrap subgrid 𝑖 indices around with modulo nx.
    # For ΔxCC, of size (Nx, Ny), we have:
    #
    #                       ┏━━━━━━━━━┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┓
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #  j = 2, 𝑗 = 2j = 4 ─▶ ┠─────────┼─────────╂─────────┼─────────┨
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┣━━━━━━━━━┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃◀━━━━━━━━Δx━━━━━━━▶┃         │         ┃
    #  j = 1, 𝑗 = 2j = 2 ─▶ u ─────── c ────────╂─────────┼─────────┨
    #                       ┃◀───dx──▶│◀───dx──▶┃         │         ┃
    #                       ┃    ▲    │    ▲    ┃         │         ┃
    #                       ┃    │    │    │    ┃         │         ┃
    #                       ┗━━━━┿━━━ v ━━━┿━━━━┻━━━━━━━━━┷━━━━━━━━━┛
    #                            │    ▲    │              ▲
    #                            │  i = 1  │            i = 2
    #                            │         𝑖 = 2i = 2
    #                            𝑖 = 2i - 1 = 1
    #
    # For ΔxFF, of size (Nx, Ny + 1), we have:
    #
    #  j = 3, 𝑗 = 2j - 1 = 5 ─▶ ┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┓
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #                           │  ghost  ┃         │         ┃         │         ┃
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #                           ┼─────────╂─────────┼─────────╂─────────┼─────────┨
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #                           │  ghost  ┃         │         ┃         │         ┃
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #  j = 2, 𝑗 = 2j - 1 = 3 ─▶ ┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #                           │  ghost  ┃         │         ┃         │         ┃
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #                           ┼──────── u ─────── c ────────╂─────────┼─────────┨
    #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
    #                           │  ghost  ┃         │         ┃         │         ┃
    #                           │◀━━━━━━━━Δx━━━━━━━▶│         ┃         │         ┃
    #  j = 1, 𝑗 = 2j - 1 = 1 ─▶ ┷━━━━━━━━━┻━━━━━━━━ v ━━━━━━━━┻━━━━━━━━━┷━━━━━━━━━┛
    #                            ◀───dx──▶▲◀───dx──▶          ▲          ◀───dx──▶
    #                                ▲    ┃    ▲              ┃              ▲
    #                                │  i = 1  │            i = 2            │
    #                                │         𝑖 = 2i - 1 = 1                │
    #                                𝑖 = 2i - 2 = 0 ----> wrap it with ----> 𝑖 = mod1(2i - 2, nx)
    #                                                                          = mod1(0, 4) = 4
    ΔxFF[i, j] = dx[mod1(2i - 2, nx), 2j - 1] + dx[2i - 1, 2j - 1]
    ΔxFC[i, j] = dx[mod1(2i - 2, nx), 2j] + dx[2i - 1, 2j]
    ΔxCF[i, j] = dx[2i - 1, 2j - 1] + dx[2i, 2j - 1]
    ΔxCC[i, j] = dx[2i - 1, 2j] + dx[2i, 2j]

    # For Δy, I need to sum consecutive dy 2 by 2,
    # but I need to "extend" the grid north and south.
    # For ΔyCC, of size (Nx, Ny), we have:
    #
    #                       ┏━━━━━━━━━┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┓
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #              j = 2 ─▶ ┠─────────┼─────────╂─────────┼─────────┨
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┣━━━━━━━━━┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
    #                       ┃        ▲│▲        ┃         │         ┃
    #         𝑗 = 2j = 2 ─▶ ┃        ┃││dy      ┃         │         ┃
    #                       ┃        ┃│▼        ┃         │         ┃
    #              j = 1 ─▶ u ───── Δy ─────────╂─────────┼─────────┨
    #                       ┃        ┃│▲        ┃         │         ┃
    #     𝑗 = 2j - 1 = 1 ─▶ ┃        ┃││dy      ┃         │         ┃
    #                       ┃        ▼│▼        ┃         │         ┃
    #                       ┗━━━━━━━━ v ━━━━━━━━┻━━━━━━━━━┷━━━━━━━━━┛
    #                                 ▲                   ▲
    #                               i = 1               i = 2
    #                            𝑖 = 2i = 2           𝑖 = 2i = 4
    #
    #
    # For ΔyFF, of size (Nx, Ny + 1), we clamp the j indices at the boundaries:
    #
    #                       ┠─────────┼─────────╂─────────┼─────────┨
    #    so repeat 𝑗 = 4   ▲┃▲ ╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
    #    𝑗 = 2j - 1 = 5 ─▶ ┃┃│dy ╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
    #                      ┃┃▼ ghost  │  ghost  ┃  ghost  │  ghost  ┃
    #            j = 3 ─▶ Δy┣━━━━━━━━ v ━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
    #                      ┃┃▲        │         ┃         │         ┃
    #    𝑗 = 2j - 2 = 4 ─▶ ┃┃│dy      │         ┃         │         ┃
    #                      ▼┃▼        │         ┃         │         ┃
    #                       ┠─────────┼─────────╂─────────┼─────────┨
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #             j = 2 ─▶  ┣━━━━━━━━━┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       ┃         │         ┃         │         ┃
    #                       u ─────── c ────────╂─────────┼─────────┨
    #                      ▲┃▲        │         ┃         │         ┃
    #    𝑗 = 2j - 1 = 1 ─▶ ┃┃│dy      │         ┃         │         ┃
    #                      ┃┃▼        │         ┃         │         ┃
    #            j = 1 ─▶ Δy┣━━━━━━━━ v ━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
    #                      ┃┃▲ ghost  │  ghost  ┃  ghost  │  ghost  ┃
    #    𝑗 = 2j - 2 = 0 ─▶ ┃┃│dy ╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
    #    so repeat 𝑗 = 1   ▼┃▼ ╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
    #                       ┠─────────┼─────────╂─────────┼─────────┨
    #                       ▲                   ▲
    #                     i = 1               i = 2
    #                𝑖 = 2i - 1 = 1       𝑖 = 2i - 1 = 3
    #
    ΔyFF[i, j] = dy[2i - 1, max(2j - 2, 1)] + dy[2i - 1, min(2j - 1, ny)]
    ΔyFC[i, j] = dy[2i - 1, 2j - 1] + dy[2i - 1, 2j]
    ΔyCF[i, j] = dy[2i, max(2j - 2, 1)] + dy[2i, min(2j - 1, ny)]
    ΔyCC[i, j] = dy[2i, 2j - 1] + dy[2i, 2j]

    # For area use the same logic as above but sum 4 by 4
    AzFF[i, j] = area[mod1(2i - 2, nx), max(2j - 2, 1)] + area[mod1(2i - 2, nx), min(2j - 1, ny)] + area[2i - 1, max(2j - 2, 1)] + area[2i - 1, min(2j - 1, ny)]
    AzFC[i, j] = area[mod1(2i - 2, nx), 2j - 1] + area[mod1(2i - 2, nx), 2j] + area[2i - 1, 2j - 1] + area[2i - 1, 2j]
    AzCF[i, j] = area[2i - 1, max(2j - 2, 1)] + area[2i - 1, min(2j - 1, ny)] + area[2i, max(2j - 2, 1)] + area[2i, min(2j - 1, ny)]
    AzCC[i, j] = area[2i - 1, 2j - 1] + area[2i - 1, 2j] + area[2i, 2j - 1] + area[2i, 2j]

end



function tripolargrid_from_supergrid(
        arch = CPU(), FT::DataType = Float64;
        x, y, dx, dy, area,
        nx, nxp, ny, nyp,
        halosize = (4, 4, 4),
        radius = Oceananigans.defaults.planet_radius,
        z = (0, 1), # Maybe z can be 3D array here?
        Nz = 1,
    )

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

    # Helper grid to fill halo
    Nx = Nλ
    Ny = Nφ
    grid = RectilinearGrid(;
        size = (Nx, Ny),
        halo = (Hλ, Hφ),
        x = (0, 1), y = (0, 1),
        topology = (Periodic, RightFaceConnected, Flat),
    )

    # For z use the same as Oceananigans TripolarGrid
    # while λ and φ will come from supergrid.
    topology = (Periodic, RightFaceConnected, Bounded)
    TZ = topology[3]
    z = validate_dimension_specification(TZ, z, :z, Nz, FT)
    Lz, z = generate_coordinate(FT, topology, gridsize, halosize, z, :z, 3, CPU())

    # To get data of the right size, we create fields at the right locations
    # with the right boundary conditions.
    # We need to define them manually because of the convention in the
    # FPivotZipperBoundaryCondition that edge fields need to switch sign
    # (which we definitely do not want for coordinates and metrics)
    # TODO: Check that, actually... I don't think that's true as
    # I think the sign change only happens for tracers called :u or :v.
    boundary_conditions = FieldBoundaryConditions(
        north = FPivotZipperBoundaryCondition(),
        south = NoFluxBoundaryCondition(), # The south should be `continued`
        west = Oceananigans.PeriodicBoundaryCondition(),
        east = Oceananigans.PeriodicBoundaryCondition(),
        top = nothing,
        bottom = nothing
    )

    λFF = Field{Face, Face, Center}(grid; boundary_conditions)
    λFC = Field{Face, Center, Center}(grid; boundary_conditions)
    λCF = Field{Center, Face, Center}(grid; boundary_conditions)
    λCC = Field{Center, Center, Center}(grid; boundary_conditions)
    φFF = Field{Face, Face, Center}(grid; boundary_conditions)
    φFC = Field{Face, Center, Center}(grid; boundary_conditions)
    φCF = Field{Center, Face, Center}(grid; boundary_conditions)
    φCC = Field{Center, Center, Center}(grid; boundary_conditions)
    ΔxFF = Field{Face, Face, Center}(grid; boundary_conditions)
    ΔxFC = Field{Face, Center, Center}(grid; boundary_conditions)
    ΔxCF = Field{Center, Face, Center}(grid; boundary_conditions)
    ΔxCC = Field{Center, Center, Center}(grid; boundary_conditions)
    ΔyFF = Field{Face, Face, Center}(grid; boundary_conditions)
    ΔyFC = Field{Face, Center, Center}(grid; boundary_conditions)
    ΔyCF = Field{Center, Face, Center}(grid; boundary_conditions)
    ΔyCC = Field{Center, Center, Center}(grid; boundary_conditions)
    AzFF = Field{Face, Face, Center}(grid; boundary_conditions)
    AzFC = Field{Face, Center, Center}(grid; boundary_conditions)
    AzCF = Field{Center, Face, Center}(grid; boundary_conditions)
    AzCC = Field{Center, Center, Center}(grid; boundary_conditions)

    # Compute coordinates and metrics from supergrid
    # but run the kernel up to (Nλ, Nφ + 1) instead of (Nλ, Nφ)!
    # (We extend the indices to make sure to fill interior points for all locations.)
    loop! = compute_coordinates_and_metrics_from_supergrid!(device(CPU()), (16, 16), (Nλ, Nφ + 1))
    loop!(
        λFF, λFC, λCF, λCC,     # TripolarGrid longitude coordinates
        φFF, φFC, φCF, φCC,     # TripolarGrid latitude coordinates
        ΔxFF, ΔxFC, ΔxCF, ΔxCC, # TripolarGrid x distances
        ΔyFF, ΔyFC, ΔyCF, ΔyCC, # TripolarGrid y distances
        AzFF, AzFC, AzCF, AzCC, # TripolarGrid areas
        x, y,   # supergrid coordinates
        dx, dy, # supergrid distances
        area,   # supergrid areas
        nx      # supergrid size in x (nx == 2 * Nx)
    )

    # Fill halos (important as we overwrote some halo regions above)
    for x in (
            λFF, λFC, λCF, λCC,     # TripolarGrid longitude coordinates
            φFF, φFC, φCF, φCC,     # TripolarGrid latitude coordinates
            ΔxFF, ΔxFC, ΔxCF, ΔxCC, # TripolarGrid x distances
            ΔyFF, ΔyFC, ΔyCF, ΔyCC, # TripolarGrid y distances
            AzFF, AzFC, AzCF, AzCC, # TripolarGrid areas
        )
        fill_halo_regions!(x)
    end

    # and only keep interior data + drop z dimension
    λᶠᶠᵃ = dropdims(λFF.data, dims = 3)
    λᶠᶜᵃ = dropdims(λFC.data, dims = 3)
    λᶜᶠᵃ = dropdims(λCF.data, dims = 3)
    λᶜᶜᵃ = dropdims(λCC.data, dims = 3)
    φᶠᶠᵃ = dropdims(φFF.data, dims = 3)
    φᶠᶜᵃ = dropdims(φFC.data, dims = 3)
    φᶜᶠᵃ = dropdims(φCF.data, dims = 3)
    φᶜᶜᵃ = dropdims(φCC.data, dims = 3)
    Δxᶠᶠᵃ = dropdims(FF.data, dims = 3)
    Δxᶜᶠᵃ = dropdims(CF.data, dims = 3)
    Δxᶠᶜᵃ = dropdims(FC.data, dims = 3)
    Δxᶜᶜᵃ = dropdims(CC.data, dims = 3)
    Δyᶠᶠᵃ = dropdims(FF.data, dims = 3)
    Δyᶜᶠᵃ = dropdims(CF.data, dims = 3)
    Δyᶠᶜᵃ = dropdims(FC.data, dims = 3)
    Δyᶜᶜᵃ = dropdims(CC.data, dims = 3)
    Azᶠᶠᵃ = dropdims(FF.data, dims = 3)
    Azᶜᶠᵃ = dropdims(CF.data, dims = 3)
    Azᶠᶜᵃ = dropdims(FC.data, dims = 3)
    Azᶜᶜᵃ = dropdims(CC.data, dims = 3)

    Hx, Hy, Hz = halosize

    # TODO: Check if longitude below is correct.
    # I recreated longitude = (-180, 180) by hand here, as it does not seem to be used anywhere else
    # and I assume this is only used to conitnue the Δ metrics south, which should not depend on longitude
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
    grid = OrthogonalSphericalShellGrid{Periodic, RightFaceConnected, Bounded}(
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
        # TODO: this mapping to Tripolar should be replaced with a custom one
        Tripolar(north_poles_latitude, first_pole_longitude, southernmost_latitude)
    )

    @warn "This grid uses a Tripolar mapping but it should have its own custom one I think."

    return grid
end

WIP WIP WIP

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
    x.data[2:Nx, 2:Ny, 1:Nz] .= data[1:(Nx - 1), 1:(Ny - 1), Nz:-1:1]
    x.data[1, 2:Ny, 1:Nz] .= data[Nx, 1:(Ny - 1), Nz:-1:1]
    Oceananigans.BoundaryConditions.fill_halo_regions!(x)
    return x
end

"""
I think I need to make my own BC first on the B-grid velocities,
then interpolate to C-grid,
then merge cells across the fold,
and only then fill halo regions with the Oceananigans machinery
(because it can only deal with the fold at XFace points).
"""
function Bgrid_OffsetArray_velocity_from_MOM_with_foldᵃᶠᵃ(grid, data)
    # I only use the grid here to create the same offsetarray
    x = Field{Face, Face, Center}(grid).data
    Nx, Ny, Nz = size(grid)
    # Shift everything from NE to SW and flip vertical
    x[2:(Nx + 1), 2:(Ny + 1), 1:Nz] .= data[1:Nx, 1:Ny, Nz:-1:1]
    # Fill i = 1 column by wrapping around in longitude
    x[1, 2:(Ny + 1), 1:Nz] .= data[Nx, 1:Ny, Nz:-1:1]
    return x
end

function interpolate_u_from_Bgrid_to_Cgrid!(uc, ubdata)
    for i in 1:(Nx + 1), j in 1:Ny, k in 1:Nz
        uc.data[i, j, k] = (ubdata[i, j, k] + ubdata[i, j + 1, k]) / 2
    end
    return uc
end
function interpolate_v_from_Bgrid_to_Cgrid!(vc, vbdata)
    for i in 1:Nx, j in 1:(Ny + 1), k in 1:Nz
        vc.data[i, j, k] = (vbdata[i, j, k] + vbdata[i + 1, j, k]) / 2
    end
    return vc
end


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

function plot_surface_field(grid, xstr; prefix = "")
    xdata = getproperty(grid, xstr)
    x = Field{celllocation(xstr)...}(grid)
    x .= xdata
    # mask_immersed_field!(x, NaN)
    # fill_halo_regions!(x)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "i", ylabel = "j", aspect = DataAspect())
    (; Hx, Hy, Nx, Ny, Nz) = grid
    hm = heatmap!(ax, (1 - Hx):(Nx + Hx), (1 - Hy):(Ny + Hy), x.data[:, :, Nz].parent; nan_color = :black)
    ax.title = "$xstr at surface"
    # translate!(hm, (0, 0, -100))
    Colorbar(fig[2, 1], hm; vertical = false, tellwidth = false)
    filepath = joinpath(outputdir, "$(prefix)$(xstr)_map.png")
    save(filepath, fig)
    return filepath
end
