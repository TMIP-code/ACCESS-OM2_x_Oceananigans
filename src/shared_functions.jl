using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, NoFluxBoundaryCondition, fill_halo_regions!
using Oceananigans.Grids: Grids, Bounded, Flat, OrthogonalSphericalShellGrid, Periodic, RectilinearGrid, RightFaceFolded,
    validate_dimension_specification, generate_coordinate, on_architecture, znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, continue_south!
using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using KernelAbstractions: @kernel, @index
using GPUArraysCore: @allowscalar
using Adapt: adapt
using Statistics: mean, median
using Printf: @sprintf


@kernel function compute_coordinates_and_metrics_from_supergrid!(
        λFF, λFC, λCF, λCC,     # TripolarGrid longitude coordinates
        φFF, φFC, φCF, φCC,     # TripolarGrid latitude coordinates
        ΔxFF, ΔxFC, ΔxCF, ΔxCC, # TripolarGrid x distances
        ΔyFF, ΔyFC, ΔyCF, ΔyCC, # TripolarGrid y distances
        AzFF, AzFC, AzCF, AzCC, # TripolarGrid areas
        x, y,   # supergrid coordinates
        dx, dy, # supergrid distances
        area,   # supergrid areas
        nx, ny  # supergrid size in x (nx = 2 * Nx, ny = 2 * (Ny - 1))
    )

    # Note this kernel will fill "interior halos" sometimes.
    # ("interior halo" here refers to those partly or fully diagnostic slices
    # that are "inside" the default grid size, (Nx, Ny).)
    # That's OK because we fill halos after.
    # But that means we sometimes go out of supergrid index bounds.

    i, j = @index(Global, NTuple)

    # For λ we just copy from the super grid incrementing by 2 in each direction.
    # Remember the RightFaceFolded grid has an extra row, so we have:
    #
    #                                                         halo
    #                           ┏━━━━━━┯━━━━━━┳━━━━━━┯━━━━━━┓ ────
    #                           ┃ ╱╱╱╱ │ ╱╱╱╱ ┃ ╱╱╱╱ │ ╱╱╱╱ ┃
    #                           ┃ ╱╱╱╱ │ ╱╱╱╱ ┃ ╱╱╱╱ │ ╱╱╱╱ ┃ half-halo
    #  j = 3,     𝑗 = 2j = 6 ─▶ ┠──────┼──────╂──────┼──────┨ half-interior
    #                           ┃ ╱╱╱╱ │ ╱╱╱╱ ┃ ╱╱╱╱ │ ╱╱╱╱ ┃
    #                           ┃ ╱╱╱╱ │ ╱╱╱╱ ┃ ╱╱╱╱ │ ╱╱╱╱ ┃
    #  j = 3, 𝑗 = 2j - 1 = 5 ─▶ ┣━━━━━━┿━━━━━━╋━━━━━━┿━━━━━━┫ ────────
    #                           ┃      │      ┃      │      ┃ interior
    #                           ┃      │      ┃      │      ┃
    #  j = 2,     𝑗 = 2j = 4 ─▶ ┠──────┼──────╂──────┼──────┨
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #  j = 2, 𝑗 = 2j - 1 = 3 ─▶ ┣━━━━━━┿━━━━━━╋━━━━━━┿━━━━━━┫
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #  j = 1,     𝑗 = 2j = 2 ─▶ FC ─── CC ────╂──────┼──────┨
    #                           ┃      │      ┃      │      ┃
    #                           ┃      │      ┃      │      ┃
    #  j = 1, 𝑗 = 2j - 1 = 1 ─▶ FF ━━━ CF ━━━━┻━━━━━━┷━━━━━━┛
    #                           ▲      ▲      ▲      ▲
    #                       i = 1      1      2      2
    #              𝑖 =   2i   =        2             4
    #              𝑖 = 2i - 1 = 1             3
    #
    #
    # Note that this kernel will try to fill CC and FC at index j = Ny (j = 3).
    # That's OK for the grid we are building because the halos will be filled in,
    # but it's not OK for the input grid, for which 𝑗 = 2j = 6 is out of bounds.
    # So I clamp 𝑗 to valid indices.
    @inbounds begin
        λFF[i, j] = x[2i - 1, clamp(2j - 1, 1, ny + 1)]
        φFF[i, j] = y[2i - 1, clamp(2j - 1, 1, ny + 1)]
        λFC[i, j] = x[2i - 1, clamp(2j, 1, ny + 1)]
        φFC[i, j] = y[2i - 1, clamp(2j, 1, ny + 1)]
        λCF[i, j] = x[2i, clamp(2j - 1, 1, ny + 1)]
        φCF[i, j] = y[2i, clamp(2j - 1, 1, ny + 1)]
        λCC[i, j] = x[2i, clamp(2j, 1, ny + 1)]
        φCC[i, j] = y[2i, clamp(2j, 1, ny + 1)]

        # For Δx, I need to sum consecutive dx 2 by 2,
        # and sometimes wrap subgrid 𝑖 indices around with modulo nx.
        # For ΔxCC, we have:
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
        # For ΔxFF, we have:
        #
        #  j = 3, 𝑗 = 2j - 1 = 5 ─▶ ┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┳━━━━━━━━━┯━━━━━━━━━┓
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #                           │  halo   ┃         │         ┃         │         ┃
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #                           ┼─────────╂─────────┼─────────╂─────────┼─────────┨
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #                           │  halo   ┃         │         ┃         │         ┃
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #  j = 2, 𝑗 = 2j - 1 = 3 ─▶ ┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #                           │  halo   ┃         │         ┃         │         ┃
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #                           ┼──────── u ─────── c ────────╂─────────┼─────────┨
        #                           │ ╱╱╱╱╱╱╱ ┃         │         ┃         │         ┃
        #                           │  halo   ┃         │         ┃         │         ┃
        #                           │◀━━━━━━━━Δx━━━━━━━▶│         ┃         │         ┃
        #  j = 1, 𝑗 = 2j - 1 = 1 ─▶ ┷━━━━━━━━━┻━━━━━━━━ v ━━━━━━━━┻━━━━━━━━━┷━━━━━━━━━┛
        #                            ◀───dx──▶▲◀───dx──▶          ▲          ◀───dx──▶
        #                                ▲    ┃    ▲              ┃              ▲
        #                                │  i = 1  │            i = 2            │
        #                                │         𝑖 = 2i - 1 = 1                │
        #                                𝑖 = 2i - 2 = 0 ----> wrap it with ----> 𝑖 = mod1(2i - 2, nx)
        #                                                                          = mod1(0, 4) = 4
        ΔxFF[i, j] = dx[mod1(2i - 2, nx), clamp(2j - 1, 1, ny + 1)] + dx[2i - 1, clamp(2j - 1, 1, ny + 1)]
        ΔxFC[i, j] = dx[mod1(2i - 2, nx), clamp(2j, 1, ny + 1)] + dx[2i - 1, clamp(2j, 1, ny + 1)]
        ΔxCF[i, j] = dx[2i - 1, clamp(2j - 1, 1, ny + 1)] + dx[2i, clamp(2j - 1, 1, ny + 1)]
        ΔxCC[i, j] = dx[2i - 1, clamp(2j, 1, ny + 1)] + dx[2i, clamp(2j, 1, ny + 1)]

        # For Δy, I need to sum consecutive dy 2 by 2.
        # For ΔyCC, we have:
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
        # For ΔyFF:
        #
        #                       ┠─────────┼─────────╂─────────┼─────────┨
        #     clamp at 𝑗 = 4   ▲┃▲ ╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #    𝑗 = 2j - 1 = 7 ─▶ ┃┃│dy ╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #                      ┃┃▼ halo   │  halo   ┃  halo   │  halo   ┃
        #            j = 4 ─▶ Δy┣━━━━━━━━ v ━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
        #                      ┃┃▲ ╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #    𝑗 = 2j - 2 = 6 ─▶ ┃┃│dy ╱╱╱╱ │ inthalo ┃ inthalo │ inthalo ┃
        #                      ▼┃▼ ╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #                       ┠─────────┼─────────╂─────────┼─────────┨
        #                       ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #                       ┃ inthalo │ inthalo ┃ inthalo │ inthalo ┃
        #                       ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #             j = 3 ─▶  ┣━━━━━━━━━┿━━━━━━━━━╋━━━━━━━━━┿━━━━━━━━━┫
        #                       ┃         │         ┃         │         ┃
        #                       ┃         │         ┃         │         ┃
        #                       ┃         │         ┃         │         ┃
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
        #                      ┃┃▲ halo   │  halo   ┃  halo   │  halo   ┃
        #    𝑗 = 2j - 2 = 0 ─▶ ┃┃│dy ╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #    so repeat 𝑗 = 1   ▼┃▼ ╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃ ╱╱╱╱╱╱╱ │ ╱╱╱╱╱╱╱ ┃
        #                       ┠─────────┼─────────╂─────────┼─────────┨
        #                       ▲                   ▲
        #                     i = 1               i = 2
        #                𝑖 = 2i - 1 = 1       𝑖 = 2i - 1 = 3
        #
        ΔyFF[i, j] = dy[2i - 1, clamp(2j - 2, 1, ny)] + dy[2i - 1, clamp(2j - 1, 1, ny)]
        ΔyFC[i, j] = dy[2i - 1, clamp(2j - 1, 1, ny)] + dy[2i - 1, clamp(2j, 1, ny)]
        ΔyCF[i, j] = dy[2i, clamp(2j - 2, 1, ny)] + dy[2i, clamp(2j - 1, 1, ny)]
        ΔyCC[i, j] = dy[2i, clamp(2j - 1, 1, ny)] + dy[2i, clamp(2j, 1, ny)]

        # For area use the same logic as above but sum 4 by 4
        AzFF[i, j] = area[mod1(2i - 2, nx), clamp(2j - 2, 1, ny)] + area[mod1(2i - 2, nx), clamp(2j - 1, 1, ny)] + area[2i - 1, clamp(2j - 2, 1, ny)] + area[2i - 1, clamp(2j - 1, 1, ny)]
        AzFC[i, j] = area[mod1(2i - 2, nx), clamp(2j - 1, 1, ny)] + area[mod1(2i - 2, nx), clamp(2j, 1, ny)] + area[2i - 1, clamp(2j - 1, 1, ny)] + area[2i - 1, clamp(2j, 1, ny)]
        AzCF[i, j] = area[2i - 1, clamp(2j - 2, 1, ny)] + area[2i - 1, clamp(2j - 1, 1, ny)] + area[2i, clamp(2j - 2, 1, ny)] + area[2i, clamp(2j - 1, 1, ny)]
        AzCC[i, j] = area[2i - 1, clamp(2j - 1, 1, ny)] + area[2i - 1, clamp(2j, 1, ny)] + area[2i, clamp(2j - 1, 1, ny)] + area[2i, clamp(2j, 1, ny)]
    end

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

    southernmost_latitude = minimum(y)
    latitude = (southernmost_latitude, 90)
    longitude = (minimum(x), maximum(x))
    max_latitudes = maximum(y, dims = 2)
    north_poles_latitude, i_north_pole = findmin(max_latitudes)
    first_pole_longitude = @allowscalar x[i_north_pole, 1]

    # Horizontal grid size
    # Note the RightFaceFolded topology requires an extra row on the north since my FPivot PR
    Nx = nx ÷ 2
    Ny = ny ÷ 2 + 1
    @assert nx == 2Nx
    @assert ny == 2(Ny - 1)

    # Halo size
    Hx, Hy, Hz = halosize
    gridsize = (Nx, Ny, Nz)

    # Helper grid to fill halo
    Nx = Nx
    Ny = Ny
    grid = RectilinearGrid(
        CPU(), FT;
        size = (Nx, Ny),
        halo = (Hx, Hy),
        x = (0, 1), y = (0, 1),
        topology = (Periodic, RightFaceFolded, Flat),
    )

    # For z use the same as Oceananigans TripolarGrid
    # while λ and φ will come from supergrid.
    topology = (Periodic, RightFaceFolded, Bounded)
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
    kp = KernelParameters(1:Nx, 1:Ny)
    launch!(
        CPU(), grid, kp,
        compute_coordinates_and_metrics_from_supergrid!,
        λFF, λFC, λCF, λCC,     # TripolarGrid longitude coordinates
        φFF, φFC, φCF, φCC,     # TripolarGrid latitude coordinates
        ΔxFF, ΔxFC, ΔxCF, ΔxCC, # TripolarGrid x distances
        ΔyFF, ΔyFC, ΔyCF, ΔyCC, # TripolarGrid y distances
        AzFF, AzFC, AzCF, AzCC, # TripolarGrid areas
        x, y,   # supergrid coordinates
        dx, dy, # supergrid distances
        area,   # supergrid areas
        nx, ny  # supergrid size in x (nx = 2Nx, ny = 2(Ny - 1))
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
    Δxᶠᶠᵃ = dropdims(ΔxFF.data, dims = 3)
    Δxᶜᶠᵃ = dropdims(ΔxCF.data, dims = 3)
    Δxᶠᶜᵃ = dropdims(ΔxFC.data, dims = 3)
    Δxᶜᶜᵃ = dropdims(ΔxCC.data, dims = 3)
    Δyᶠᶠᵃ = dropdims(ΔyFF.data, dims = 3)
    Δyᶜᶠᵃ = dropdims(ΔyCF.data, dims = 3)
    Δyᶠᶜᵃ = dropdims(ΔyFC.data, dims = 3)
    Δyᶜᶜᵃ = dropdims(ΔyCC.data, dims = 3)
    Azᶠᶠᵃ = dropdims(AzFF.data, dims = 3)
    Azᶜᶠᵃ = dropdims(AzCF.data, dims = 3)
    Azᶠᶜᵃ = dropdims(AzFC.data, dims = 3)
    Azᶜᶜᵃ = dropdims(AzCC.data, dims = 3)

    # Final grid with correct metrics
    # TODO: remove `on_architecture(arch, ...)` when we shift grid construction to GPU
    grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
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

    @warn "This grid uses a `Tripolar` mapping but it should have its own custom one I think."

    return grid
end


function load_tripolar_grid(grid_file, arch = CPU(), FT = Float64)
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
        underlying_grid,
        PartialCellBottom(gd["bottom"]);
        active_cells_map = true,
        active_z_columns = true,
    )


    return grid
end


#taken from Makie ext (not sure how to load these)
function drop_singleton_indices(N)
    if N == 1
        return 1
    else
        return Colon()
    end
end
function make_plottable_array(f)
    compute!(f)
    mask_immersed_field!(f, NaN)

    Nx, Ny, Nz = size(f)

    ii = drop_singleton_indices(Nx)
    jj = drop_singleton_indices(Ny)
    kk = drop_singleton_indices(Nz)

    fi = interior(f, ii, jj, kk)
    fi_cpu = on_architecture(CPU(), fi)

    if architecture(f) isa CPU
        fi_cpu = deepcopy(fi_cpu) # so we can re-zero peripheral nodes
    end

    mask_immersed_field!(f)

    return fi_cpu
end


################################################################################
# Wet cell mask and indexing
################################################################################

"""
    compute_wet_mask(grid) -> (; wet3D, idx, Nidx)

Create the 3D wet cell boolean mask, linear index vector, and count.
Returns interior-sized arrays (excludes fold point for tripolar grids).
"""
function compute_wet_mask(grid)
    fNaN = CenterField(grid)
    mask_immersed_field!(fNaN, NaN)
    wet3D = .!isnan.(interior(on_architecture(CPU(), fNaN)))
    idx = findall(wet3D)
    Nidx = length(idx)
    return (; wet3D, idx, Nidx)
end


################################################################################
# Cell volume computation
################################################################################

@kernel function compute_volume!(vol, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds vol[i, j, k] = volume(i, j, k, grid, Center(), Center(), Center())
end

"""
    compute_volume(grid) -> CenterField

Compute cell volumes as a CenterField on the same architecture as grid.
"""
function compute_volume(grid)
    vol = CenterField(grid)
    (Nxv, Nyv, Nzv) = size(vol)
    kp = KernelParameters(1:Nxv, 1:Nyv, 1:Nzv)
    launch!(CPU(), grid, kp, compute_volume!, vol, grid)
    return vol
end


################################################################################
# Hydrostatic free surface tendency kernel
################################################################################

@kernel function compute_hydrostatic_free_surface_GADc!(GADc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GADc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end


################################################################################
# Progress message callback for simulations
#
# NOTE: run_1year.jl defines its own version with a different format string
# (divides max_age by elapsed time). That version is NOT centralized here.
################################################################################

function progress_message(sim)
    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year)
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout)
    return @info @sprintf(
        "  sim iter: %04d, time: %1.3f, Δt: %.2e, max(age) = %.1e at (%d, %d, %d), mean(age) = %.1e, wall: %s\n",
        iteration(sim), time(sim), sim.Δt, max_age, idx_max.I..., mean_age, walltime
    )
end


################################################################################
# Analysis utilities: zonal averages and horizontal slices
################################################################################

"""
    compute_ocean_basin_masks(grid) -> (; ATL, PAC, IND)

Compute Atlantic, Pacific, and Indian ocean basin masks using OceanBasins.jl.
Returns a named tuple of 2D Bool arrays sized (Nx', Ny').

Requires `OCEANS, isatlantic, ispacific, isindian` from OceanBasins in scope.
"""
function compute_ocean_basin_masks(grid)
    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′ = size(ug)[1:2]
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    lon = Array(ug.λᶜᶜᵃ[1:Nx′, 1:Ny′])

    flat_lat = vec(lat)
    flat_lon = vec(lon)
    ATL = reshape(isatlantic(flat_lat, flat_lon, OCEANS), size(lat))
    PAC = reshape(ispacific(flat_lat, flat_lon, OCEANS), size(lat))
    IND = reshape(isindian(flat_lat, flat_lon, OCEANS), size(lat))

    return (; ATL, PAC, IND)
end

"""
    zonalaverage(x3D, v3D, mask)

Volume-weighted zonal average (average along dimension 1).
`mask` is a 2D or 3D boolean array; 2D masks broadcast over depth.
NaN values in `x3D` are excluded. Returns a (Ny, Nz) matrix.
"""
function zonalaverage(x3D, v3D, mask)
    m = ndims(mask) == 2 ? reshape(mask, size(mask, 1), size(mask, 2), 1) : mask
    xw = @. ifelse(isnan(x3D) | !m, 0.0, x3D * v3D)
    w = @. ifelse(isnan(x3D) | !m, 0.0, v3D)
    num = dropdims(sum(xw; dims = 1); dims = 1)
    den = dropdims(sum(w; dims = 1); dims = 1)
    return @. ifelse(den > 0, num / den, NaN)
end

"""
    find_nearest_depth_index(grid, target_depth)

Return the k-index of the vertical level nearest to `target_depth` (m, positive downward).
"""
function find_nearest_depth_index(grid, target_depth)
    z = znodes(grid, Center(), Center(), Center())
    _, k = findmin(abs.(z .+ target_depth))
    return k
end


################################################################################
# Age diagnostic plots (10 figures)
#
# Requires CairoMakie and OceanBasins symbols in the calling script's scope.
################################################################################

"""
    plot_age_diagnostics(age_3D, grid, wet3D, vol_3D, output_dir, label;
                         colorrange=nothing, colormap=:viridis)

Generate 10 diagnostic figures and save as PNG:
  1-4: Zonal average (global, Atlantic, Pacific, Indian) — contourf (lat vs depth)
  5-10: Horizontal slices at 100, 200, 500, 1000, 2000, 3000 m — heatmap

Arguments:
- `age_3D`:     (Nx', Ny', Nz') array (years) with 0 for dry cells
- `grid`:       ImmersedBoundaryGrid (tripolar)
- `wet3D`:      (Nx', Ny', Nz') Bool mask
- `vol_3D`:     (Nx', Ny', Nz') volume array (m^3)
- `output_dir`: directory for saving PNGs
- `label`:      filename prefix (e.g. "steady_age_full")
"""
function plot_age_diagnostics(
        age_3D, grid, wet3D, vol_3D, output_dir, label;
        colorrange = nothing,
        colormap = :viridis,
    )
    mkpath(output_dir)

    # Replace dry cells with NaN for plotting
    age_plot = copy(age_3D)
    age_plot[.!wet3D] .= NaN

    # Extract grid coordinates
    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′, Nz′ = size(wet3D)
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z  # positive downward

    # Representative latitude for y-axis of zonal plots (mean along i)
    lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

    # Compute basin masks
    basins = compute_ocean_basin_masks(grid)
    global_mask = trues(Nx′, Ny′)

    # ── Zonal averages (figures 1-4) ──────────────────────────────────────

    basin_configs = [
        ("global", global_mask),
        ("atlantic", basins.ATL),
        ("pacific", basins.PAC),
        ("indian", basins.IND),
    ]

    for (basin_name, basin_mask) in basin_configs
        za = zonalaverage(age_plot, vol_3D, basin_mask)

        fig = Figure(; size = (800, 500))
        ax = Axis(
            fig[1, 1];
            title = "$label — $basin_name zonal average",
            xlabel = "Latitude",
            ylabel = "Depth (m)",
            backgroundcolor = :lightgray,
            xgridvisible = false,
            ygridvisible = false,
        )

        cf_kwargs = (; colormap, nan_color = :lightgray)
        if colorrange !== nothing
            cf_kwargs = (; cf_kwargs..., colorrange)
        end
        cf = contourf!(ax, lat_repr, depth_vals, za; cf_kwargs...)
        translate!(cf, 0, 0, -100)
        ylims!(ax, maximum(depth_vals), 0)
        Colorbar(fig[1, 2], cf; label = "Age (years)")

        outputfile = joinpath(output_dir, "$(label)_zonal_avg_$(basin_name).png")
        @info "Saving $outputfile"
        save(outputfile, fig)
    end

    # ── Horizontal slices (figures 5-10) ──────────────────────────────────

    target_depths = [100, 200, 500, 1000, 2000, 3000]

    for depth in target_depths
        k = find_nearest_depth_index(grid, depth)
        actual_depth = round(depth_vals[k]; digits = 1)
        slice = age_plot[:, :, k]

        fig = Figure(; size = (1000, 500))
        ax = Axis(
            fig[1, 1];
            title = "$label at $depth m (k=$k, z=$actual_depth m)",
        )

        hm_kwargs = (; colormap, nan_color = :black)
        if colorrange !== nothing
            hm_kwargs = (; hm_kwargs..., colorrange)
        end
        hm = heatmap!(ax, slice; hm_kwargs...)
        Colorbar(fig[1, 2], hm; label = "Age (years)")

        outputfile = joinpath(output_dir, "$(label)_slice_$(depth)m.png")
        @info "Saving $outputfile"
        save(outputfile, fig)
    end

    @info "Age diagnostic plots saved to $output_dir"
    flush(stdout)
    return nothing
end
