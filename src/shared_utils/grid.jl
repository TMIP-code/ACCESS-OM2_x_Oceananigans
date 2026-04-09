################################################################################
# Grid creation, loading, wet masks, and volume computation
#
# Extracted from shared_functions.jl — tripolar grid construction from
# supergrid, grid loading from JLD2, wet cell mask, and cell volumes.
################################################################################

using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, NoFluxBoundaryCondition, fill_halo_regions!
using Oceananigans.DistributedComputations: Distributed, local_size, concatenate_local_sizes, ranks
using Oceananigans.Grids: FullyConnected, LeftConnectedRightFaceFolded, LeftConnectedRightFaceConnected, RightConnected
using Oceananigans.DistributedComputations: insert_connected_topology
using Oceananigans.OrthogonalSphericalShellGrids: receiving_rank
using OffsetArrays: OffsetArray
using Oceananigans.Grids: Grids, Bounded, Flat, MutableVerticalDiscretization, OrthogonalSphericalShellGrid, Periodic,
    RectilinearGrid, RightFaceFolded, topology, validate_dimension_specification, generate_coordinate, on_architecture, znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, TripolarGrid, continue_south!, fold_set!, partition_tripolar_metric
using Oceananigans.Architectures: CPU, GPU, architecture, child_architecture
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using KernelAbstractions: @kernel, @index
using GPUArraysCore: @allowscalar
using LinearAlgebra: dot, Diagonal

@kernel function compute_coordinates_and_metrics_from_supergrid!(
        λFF, λFC, λCF, λCC,     # TripolarGrid longitude coordinates
        φFF, φFC, φCF, φCC,     # TripolarGrid latitude coordinates
        ΔxFF, ΔxFC, ΔxCF, ΔxCC, # TripolarGrid x distances
        ΔyFF, ΔyFC, ΔyCF, ΔyCC, # TripolarGrid y distances
        AzFF, AzFC, AzCF, AzCC, # TripolarGrid areas
        x, y,   # supergrid coordinates
        dx, dy, # supergrid distances
        area,   # supergrid areas
        nx, ny  # supergrid size in x (nx = 2 * Nx, ny = 2 * Ny)
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
        halosize = (7, 7, 7),
        radius = Oceananigans.defaults.planet_radius,
        z = (0, 1), # z can be a MutableVerticalDiscretization or a tuple of (zmin, zmax)
        Nz = 1,
    )

    southernmost_latitude = minimum(y)
    latitude = (southernmost_latitude, 90)
    longitude = (minimum(x), maximum(x))
    max_latitudes = maximum(y, dims = 2)
    north_poles_latitude, i_north_pole = findmin(max_latitudes)
    first_pole_longitude = @allowscalar x[i_north_pole, 1]

    # Horizontal grid size
    Nx = nx ÷ 2
    Ny = ny ÷ 2
    @assert nx == 2Nx
    @assert ny == 2Ny

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
    # Note: the helper grid is a RectilinearGrid, not an RFTRG, so worksize(grid)
    # would return (Nx, Ny) instead of (Nx, Ny+1). Set the work range explicitly.
    Wx, Wy = Nx, Ny + 1
    kp = KernelParameters(1:Wx, 1:Wy)
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
        nx, ny  # supergrid size in x (nx = 2Nx, ny = 2Ny)
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
    underlying_grid = build_underlying_grid(gd, arch, FT)
    return ImmersedBoundaryGrid(
        underlying_grid,
        PartialCellBottom(gd["bottom"]);
        active_cells_map = true,
        active_z_columns = true,
    )
end

# Single-process (CPU or single GPU): use low-level positional constructor with saved arrays
function build_underlying_grid(gd, arch, FT = Float64)
    # Reconstruct MutableVerticalDiscretization from plain z_faces via generate_coordinate
    z_faces = gd["z_faces"]
    _, z_mvd = Oceananigans.Grids.generate_coordinate(
        FT,
        (Periodic, RightFaceFolded, Bounded),
        (gd["Nx"], gd["Ny"], gd["Nz"]),
        (gd["Hx"], gd["Hy"], gd["Hz"]),
        MutableVerticalDiscretization(z_faces),
        :z, 3, CPU(),
    )
    grid = OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
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
        on_architecture(arch, z_mvd),
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
        Tripolar(gd["north_poles_latitude"], gd["first_pole_longitude"], gd["southernmost_latitude"], RightFaceFolded),
    )
    @assert grid.z isa MutableVerticalDiscretization "Serial grid: expected MutableVerticalDiscretization, got $(typeof(grid.z))"
    return grid
end

# Distributed: partition the pre-computed global grid arrays from JLD2.
# Mirrors the logic of Oceananigans' TripolarGrid(arch::Distributed, ...) in
# distributed_tripolar_grid.jl lines 117-270, but loads arrays from JLD2 instead
# of recomputing them from a conformal mapping formula.
function build_underlying_grid(gd, arch::Distributed, FT = Float64)
    # 1. Validate partition (mirror Oceananigans checks)
    workers = ranks(arch.partition)
    px = ifelse(isnothing(arch.partition.x), 1, arch.partition.x)
    py = ifelse(isnothing(arch.partition.y), 1, arch.partition.y)
    if isodd(px) && px != 1
        throw(ArgumentError("Only even partitioning in x is supported with TripolarGrid."))
    end
    if px != 1 && py == 1
        throw(ArgumentError("An x-only partitioning is not supported for TripolarGrid."))
    end

    # 2. Build global grid from JLD2 arrays (THE FIX: uses pre-computed supergrid metrics)
    global_grid = build_underlying_grid(gd, CPU(), FT)
    Nx, Ny, Nz = size(global_grid)
    Hx, Hy, Hz = global_grid.Hx, global_grid.Hy, global_grid.Hz

    # 3. Compute local sizes and index ranges
    lsize = local_size(arch, (Nx, Ny, Nz))
    nxlocal = concatenate_local_sizes(lsize, arch, 1)
    nylocal = concatenate_local_sizes(lsize, arch, 2)

    ny_top = last(nylocal)
    if ny_top < Hy + 2
        @warn "Distributed TripolarGrid: the y-partition size on the top rank " *
            "($ny_top) is smaller than Hy + 2 = $(Hy + 2). Fold halo " *
            "communication may produce incorrect corner values."
    end

    xrank = ifelse(isnothing(arch.partition.x), 0, arch.local_index[1] - 1)
    yrank = ifelse(isnothing(arch.partition.y), 0, arch.local_index[2] - 1)

    jstart = 1 + sum(nylocal[1:yrank])
    jend = yrank == workers[2] - 1 ? Ny : sum(nylocal[1:(yrank + 1)])
    jrange = (jstart - Hy):(jend + Hy)

    istart = 1 + sum(nxlocal[1:xrank])
    iend = xrank == workers[1] - 1 ? Nx : sum(nxlocal[1:(xrank + 1)])
    irange = (istart - Hx):(iend + Hx)

    # 4. Partition coordinate and metric arrays (using upstream partition_tripolar_metric)
    λᶜᶜᵃ = partition_tripolar_metric(global_grid, :λᶜᶜᵃ, irange, jrange)
    λᶠᶜᵃ = partition_tripolar_metric(global_grid, :λᶠᶜᵃ, irange, jrange)
    λᶜᶠᵃ = partition_tripolar_metric(global_grid, :λᶜᶠᵃ, irange, jrange)
    λᶠᶠᵃ = partition_tripolar_metric(global_grid, :λᶠᶠᵃ, irange, jrange)
    φᶜᶜᵃ = partition_tripolar_metric(global_grid, :φᶜᶜᵃ, irange, jrange)
    φᶠᶜᵃ = partition_tripolar_metric(global_grid, :φᶠᶜᵃ, irange, jrange)
    φᶜᶠᵃ = partition_tripolar_metric(global_grid, :φᶜᶠᵃ, irange, jrange)
    φᶠᶠᵃ = partition_tripolar_metric(global_grid, :φᶠᶠᵃ, irange, jrange)
    Δxᶜᶜᵃ = partition_tripolar_metric(global_grid, :Δxᶜᶜᵃ, irange, jrange)
    Δxᶠᶜᵃ = partition_tripolar_metric(global_grid, :Δxᶠᶜᵃ, irange, jrange)
    Δxᶜᶠᵃ = partition_tripolar_metric(global_grid, :Δxᶜᶠᵃ, irange, jrange)
    Δxᶠᶠᵃ = partition_tripolar_metric(global_grid, :Δxᶠᶠᵃ, irange, jrange)
    Δyᶜᶜᵃ = partition_tripolar_metric(global_grid, :Δyᶜᶜᵃ, irange, jrange)
    Δyᶠᶜᵃ = partition_tripolar_metric(global_grid, :Δyᶠᶜᵃ, irange, jrange)
    Δyᶜᶠᵃ = partition_tripolar_metric(global_grid, :Δyᶜᶠᵃ, irange, jrange)
    Δyᶠᶠᵃ = partition_tripolar_metric(global_grid, :Δyᶠᶠᵃ, irange, jrange)
    Azᶜᶜᵃ = partition_tripolar_metric(global_grid, :Azᶜᶜᵃ, irange, jrange)
    Azᶠᶜᵃ = partition_tripolar_metric(global_grid, :Azᶠᶜᵃ, irange, jrange)
    Azᶜᶠᵃ = partition_tripolar_metric(global_grid, :Azᶜᶠᵃ, irange, jrange)
    Azᶠᶠᵃ = partition_tripolar_metric(global_grid, :Azᶠᶠᵃ, irange, jrange)

    # 5. Determine local topologies
    Rx, Ry = workers[1], workers[2]
    rx, ry = xrank + 1, yrank + 1
    global_fold_topology = topology(global_grid, 2)  # RightFaceFolded
    LY = insert_connected_topology(global_fold_topology, Ry, ry, Rx, rx)
    LX = workers[1] == 1 ? Periodic : FullyConnected
    ny = nylocal[yrank + 1]
    nx = nxlocal[xrank + 1]

    # 6. z and radius from global grid
    z = on_architecture(arch, global_grid.z)
    radius = global_grid.radius

    # 7. Fix fold connectivity (mirror Oceananigans distributed_tripolar_grid.jl lines 216-238)
    if workers[1] != 1
        northwest_idx_x = ranks(arch)[1] - arch.local_index[1] + 2
        northeast_idx_x = ranks(arch)[1] - arch.local_index[1]
        if northwest_idx_x > workers[1]
            northwest_idx_x = arch.local_index[1]
        end
        if northeast_idx_x < 1
            northeast_idx_x = arch.local_index[1]
        end
        northwest_recv_rank = receiving_rank(arch; receive_idx_x = northwest_idx_x)
        northeast_recv_rank = receiving_rank(arch; receive_idx_x = northeast_idx_x)
        north_recv_rank = receiving_rank(arch)
        if yrank == workers[2] - 1
            arch.connectivity.northwest = northwest_recv_rank
            arch.connectivity.northeast = northeast_recv_rank
            arch.connectivity.north = north_recv_rank
        end
    end

    # 8. Construct local grid
    grid = OrthogonalSphericalShellGrid{LX, LY, Bounded}(
        arch,
        nx, ny, Nz,
        Hx, Hy, Hz,
        convert(FT, global_grid.Lz),
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
        global_grid.conformal_mapping,
    )
    @assert grid.z isa MutableVerticalDiscretization "Distributed grid: expected MutableVerticalDiscretization, got $(typeof(grid.z))"
    return grid
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
    Nx, Ny, Nz = size(grid)
    kp = KernelParameters(1:Nx, 1:Ny, 1:Nz)
    launch!(CPU(), grid, kp, compute_volume!, vol, grid)
    return vol
end


"""
    make_vol_norm(v1D, year)

Return a volume-weighted RMS norm function in units of years:
  vol_norm(x) = sqrt(∑ vᵢ xᵢ² / ∑ vᵢ) / year
where `v1D` is a 1D vector of cell volumes (wet cells only) and `year` is
the number of seconds in a year.
"""
function make_vol_norm(v1D, year)
    inv_sumv = 1 / sum(v1D)
    return x -> sqrt(dot(x, Diagonal(v1D), x) * inv_sumv) / year
end
