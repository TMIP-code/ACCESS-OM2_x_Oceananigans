using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, NoFluxBoundaryCondition, fill_halo_regions!
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Fields: location, instantiated_location
using Oceananigans.Grids: Grids, Bounded, Flat, OrthogonalSphericalShellGrid, Periodic, RectilinearGrid, RightFaceFolded,
    validate_dimension_specification, generate_coordinate, on_architecture, znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.OrthogonalSphericalShellGrids: Tripolar, TripolarGrid, continue_south!, fold_set!
using Oceananigans.OutputReaders: FieldTimeSeries, InMemory
using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using KernelAbstractions: @kernel, @index
using GPUArraysCore: @allowscalar
using Adapt: adapt
using LinearAlgebra: dot, Diagonal
using Statistics: mean, median
using Printf: @sprintf
using TOML

################################################################################
# Config env var helpers
################################################################################

"""Parse and validate the 4 core config env vars."""
function parse_config_env()
    VS = get(ENV, "VELOCITY_SOURCE", "cgridtransports")
    WF = get(ENV, "W_FORMULATION", "wdiagnosed")
    AS = get(ENV, "ADVECTION_SCHEME", "centered2")
    TS = get(ENV, "TIMESTEPPER", "AB2")
    VS ∈ ("bgridvelocities", "cgridtransports") || error("VELOCITY_SOURCE must be bgridvelocities or cgridtransports (got: $VS)")
    WF ∈ ("wdiagnosed", "wprescribed") || error("W_FORMULATION must be wdiagnosed or wprescribed (got: $WF)")
    AS ∈ ("centered2", "weno3", "weno5") || error("ADVECTION_SCHEME must be centered2, weno3, or weno5 (got: $AS)")
    TS ∈ ("AB2", "SRK2", "SRK3", "SRK4", "SRK5") || error("TIMESTEPPER must be AB2, SRK2, SRK3, SRK4, or SRK5 (got: $TS)")
    return (; VELOCITY_SOURCE = VS, W_FORMULATION = WF, ADVECTION_SCHEME = AS, TIMESTEPPER = TS)
end

"""Convert ADVECTION_SCHEME string to Oceananigans advection object."""
function advection_from_scheme(s::AbstractString)
    return s == "centered2" ? Centered(order = 2) :
        s == "weno3" ? WENO(order = 3) :
        s == "weno5" ? WENO(order = 5) :
        error("Unknown ADVECTION_SCHEME: $s")
end

"""Convert TIMESTEPPER string to Oceananigans timestepper Symbol."""
function timestepper_from_string(s::AbstractString)
    return s == "AB2" ? :QuasiAdamsBashforth2 :
        s == "SRK2" ? :SplitRungeKutta2 :
        s == "SRK3" ? :SplitRungeKutta3 :
        s == "SRK4" ? :SplitRungeKutta4 :
        s == "SRK5" ? :SplitRungeKutta5 :
        error("Unknown TIMESTEPPER: $s")
end

"""
    load_project_config(; parentmodel_arg_index = 1) -> (; parentmodel, outputdir, Δt_seconds, profile)

Load project configuration from LocalPreferences.toml, ARGS, or ENV.
Priority: ARGS[parentmodel_arg_index] > ENV["PARENT_MODEL"] > TOML defaults > "ACCESS-OM2-1".

Returns plain Float64 for Δt_seconds; callers needing Oceananigans units multiply by `second`.
"""
function load_project_config(; parentmodel_arg_index = 1)
    cfg_file = "LocalPreferences.toml"
    cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

    parentmodel = if length(ARGS) >= parentmodel_arg_index && !isempty(ARGS[parentmodel_arg_index])
        ARGS[parentmodel_arg_index]
    elseif haskey(ENV, "PARENT_MODEL")
        ENV["PARENT_MODEL"]
    else
        get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
    end

    profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
    if profile === nothing
        @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
        outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
        Δt = parentmodel == "ACCESS-OM2-1" ? 5400.0 :
            parentmodel == "ACCESS-OM2-025" ? 1800.0 : 400.0
    else
        outputdir = profile["outputdir"]
        Δt = Float64(profile["dt_seconds"])
    end

    @info "GIT_COMMIT = $(get(ENV, "GIT_COMMIT", "unknown"))"

    return (; parentmodel, outputdir, Δt_seconds = Δt, profile)
end

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
        halosize = (7, 7, 7),
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
    return OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(
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
        Tripolar(gd["north_poles_latitude"], gd["first_pole_longitude"], gd["southernmost_latitude"]),
    )
end

# Distributed: re-create via TripolarGrid which handles partitioning, topology, and connectivity
function build_underlying_grid(gd, arch::Distributed, FT = Float64)
    # Extract z face positions from saved MutableVerticalDiscretization (strip halos)
    Nz = gd["Nz"]
    z_faces = gd["z"].cᵃᵃᶠ[1:(Nz + 1)]
    return TripolarGrid(
        arch, FT;
        size = (gd["Nx"], gd["Ny"], Nz),
        z = z_faces,
        halo = (gd["Hx"], gd["Hy"], gd["Hz"]),
        north_poles_latitude = gd["north_poles_latitude"],
        first_pole_longitude = gd["first_pole_longitude"],
        southernmost_latitude = gd["southernmost_latitude"],
        radius = gd["radius"],
        fold_topology = RightFaceFolded,
    )
end

################################################################################
# Distributed-aware loading helpers
################################################################################

"""
    load_fts(arch, file, name, grid; backend, time_indexing, cpu_grid=nothing)

Load a `FieldTimeSeries` from a JLD2 file. For `Distributed` architectures, loads on CPU
first then copies to distributed fields via `fold_set!` to work around Oceananigans bug
where `offset_data` clips global data to local size before `fold_set!` can partition it.
"""
function load_fts(arch, file, name, grid; backend, time_indexing, cpu_grid = nothing)
    return FieldTimeSeries(file, name; architecture = arch, grid, backend, time_indexing)
end

function load_fts(arch::Distributed, file, name, grid; cpu_grid, backend, time_indexing)
    @info "Loading FTS '$name' via CPU grid for distributed partitioning"
    cpu_fts = FieldTimeSeries(
        file, name; architecture = CPU(), grid = cpu_grid,
        backend = InMemory(), time_indexing
    )
    dist_fts = FieldTimeSeries(
        instantiated_location(cpu_fts), grid, cpu_fts.times;
        backend = InMemory(), time_indexing
    )
    # Use fold_set! directly because DistributedTripolarField dispatch doesn't match
    # ImmersedBoundaryGrid-wrapped fields (Oceananigans bug: DistributedTripolarField
    # uses DistributedTripolarGrid but not DistributedTripolarGridOfSomeKind).
    conformal_mapping = grid.underlying_grid.conformal_mapping
    y_loc = instantiated_location(cpu_fts)[2]
    for n in eachindex(cpu_fts.times)
        fold_set!(dist_fts[n], Array(interior(cpu_fts[n])), conformal_mapping, typeof(y_loc))
    end
    fill_halo_regions!(dist_fts)
    return dist_fts
end

"""
    load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)

Load MLD data and create a vertical diffusivity field. For `Distributed` architectures,
keeps data on CPU so `set!` dispatches to `fold_set!` with global arrays.
"""
function load_mld_diffusivity(arch, grid, mld_file, κVML, κVBG, Nz)
    mld_ds = open_dataset(mld_file)
    mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
    z_center = znodes(grid, Center(), Center(), Center())
    is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
    κVField = CenterField(grid)
    set!(κVField, κVML * is_mld + κVBG * .!is_mld)
    return κVField
end

function load_mld_diffusivity(arch::Distributed, grid, mld_file, κVML, κVBG, Nz)
    mld_ds = open_dataset(mld_file)
    mld_data = -replace(readcubedata(mld_ds.mld).data, NaN => 0.0)
    z_center = collect(znodes(grid, Center(), Center(), Center()))
    is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
    κVField = CenterField(grid)
    # Use fold_set! directly (DistributedTripolarField dispatch doesn't match ImmersedBoundaryGrid)
    conformal_mapping = grid.underlying_grid.conformal_mapping
    fold_set!(κVField, Array(κVML * is_mld + κVBG * .!is_mld), conformal_mapping, Center)
    fill_halo_regions!(κVField)
    return κVField
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


################################################################################
# Part-file loading helpers for split JLD2 output
################################################################################

# Part files are monthly snapshots produced by file_splitting:
#   part 1 = snapshot at t=0
#   part 2 = snapshot at t=1 month
#   ...
#   part 13 = snapshot at t=1 year

"""
    load_serial_part(dir, field_name, duration_tag, part) -> (data, time)

Load a single serial part file (one monthly snapshot).
Returns the 3D data array and the snapshot time.
"""
function load_serial_part(dir, field_name, duration_tag, part)
    filepath = joinpath(dir, "$(field_name)_$(duration_tag)_part$(part).jld2")
    isfile(filepath) || error("Part file not found: $filepath")
    return jldopen(filepath, "r") do f
        iters = keys(f["timeseries/$(field_name)"])
        iter = first(filter(k -> k != "serialized", iters))
        data = f["timeseries/$(field_name)/$iter"]
        t = f["timeseries/t/$iter"]
        return data, t
    end
end

"""
    load_distributed_part(dir, field_name, duration_tag, part, px, py, Nx, Ny, Nz) -> (data, time)

Load and stitch distributed rank files for a single part into a global array.
Partition layout is px × py (e.g., 2×2 for 4 GPUs).
`Nx, Ny, Nz` is the interior size (e.g., from `size(wet3D)`).
The distributed grid may include a fold row (Ny_full = Ny + 1 for tripolar grids);
the stitched result is trimmed to `(Nx, Ny, Nz)` to match serial interior output.
"""
function load_distributed_part(dir, field_name, duration_tag, part, px, py, Nx, Ny, Nz)
    nranks = px * py

    # Oceananigans rank2index uses column-major ordering:
    #   rank = i * Ry + j  (0-indexed, Rz=1)
    #   i = div(rank, py),  j = mod(rank, py)
    # So x-column i contains ranks {i*py, i*py+1, ..., i*py+py-1}
    # and y-row j contains ranks {j, j+py, j+2*py, ...}

    # Determine per-rank sizes from one rank per x-column and one per y-row
    rank_sizes_x = zeros(Int, px)
    rank_sizes_y = zeros(Int, py)
    for i in 1:px
        rank = (i - 1) * py  # first rank in x-column i
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank)_part$(part).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")
        jldopen(filepath, "r") do f
            iters = keys(f["timeseries/$(field_name)"])
            iter = first(filter(k -> k != "serialized", iters))
            rank_sizes_x[i] = size(f["timeseries/$(field_name)/$iter"], 1)
        end
    end
    for j in 1:py
        rank = j - 1  # first rank in y-row j
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank)_part$(part).jld2")
        jldopen(filepath, "r") do f
            iters = keys(f["timeseries/$(field_name)"])
            iter = first(filter(k -> k != "serialized", iters))
            rank_sizes_y[j] = size(f["timeseries/$(field_name)/$iter"], 2)
        end
    end

    Nx_full = sum(rank_sizes_x)
    Ny_full = sum(rank_sizes_y)
    x_offsets = cumsum([0; rank_sizes_x[1:(end - 1)]])
    y_offsets = cumsum([0; rank_sizes_y[1:(end - 1)]])

    global_data = zeros(Float64, Nx_full, Ny_full, Nz)
    t = nothing

    for rank in 0:(nranks - 1)
        filepath = joinpath(dir, "$(field_name)_$(duration_tag)_rank$(rank)_part$(part).jld2")
        isfile(filepath) || error("Rank file not found: $filepath")

        # Oceananigans column-major: i = div(rank, py), j = mod(rank, py)
        i_rank = div(rank, py) + 1
        j_rank = mod(rank, py) + 1

        jldopen(filepath, "r") do f
            iters = keys(f["timeseries/$(field_name)"])
            iter = first(filter(k -> k != "serialized", iters))
            local_data = f["timeseries/$(field_name)/$iter"]
            if t === nothing
                t = f["timeseries/t/$iter"]
            end

            x_start = x_offsets[i_rank] + 1
            x_end = x_offsets[i_rank] + rank_sizes_x[i_rank]
            y_start = y_offsets[j_rank] + 1
            y_end = y_offsets[j_rank] + rank_sizes_y[j_rank]

            global_data[x_start:x_end, y_start:y_end, :] .= local_data
        end
    end

    # Trim to interior size (exclude fold row for tripolar grids)
    return global_data[1:Nx, 1:Ny, :], t
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
################################################################################

function progress_message(sim)
    if architecture(sim.model.grid) isa Distributed
        # Lightweight progress for distributed runs: avoid GPU→CPU transfer and
        # MPI-synchronous findmax/mean which can deadlock across ranks.
        flush(stdout); flush(stderr)
        return @info @sprintf(
            "  sim iter: %04d, time: %1.3f yr, Δt: %.2e yr, wall: %s\n",
            iteration(sim), time(sim) / year, sim.Δt / year, prettytime(sim.run_wall_time)
        )
    end

    max_age, idx_max = findmax(adapt(Array, sim.model.tracers.age) / year)
    mean_age = mean(adapt(Array, sim.model.tracers.age)) / year
    walltime = prettytime(sim.run_wall_time)

    flush(stdout); flush(stderr)
    return @info @sprintf(
        "  sim iter: %04d, time: %1.3f yr, Δt: %.2e yr, max(age) = %.1e yr at (%d, %d, %d), mean(age) = %.1e yr, wall: %s\n",
        iteration(sim), time(sim) / year, sim.Δt / year, max_age, idx_max.I..., mean_age, walltime
    )
end


################################################################################
# Simulation setup for standard age runs
################################################################################

"""
    setup_age_simulation(model, Δt, stop_time, outputdir, model_config, duration_tag;
                          output_interval, progress_interval)

Create a Simulation with progress callback and JLD2 output writer for a standard
age simulation. Returns `(simulation, age_output_dir)`.
"""
function setup_age_simulation(
        model, Δt, stop_time, outputdir, model_config, duration_tag;
        output_interval, progress_interval
    )
    simulation = Simulation(model; Δt, stop_time)
    add_callback!(simulation, progress_message, TimeInterval(progress_interval))

    px = parse(Int, get(ENV, "GPU_PARTITION_X", "1"))
    py = parse(Int, get(ENV, "GPU_PARTITION_Y", "1"))
    gpu_tag = (px == 1 && py == 1) ? "" : "$(px)x$(py)"
    age_output_dir = isempty(gpu_tag) ?
        joinpath(outputdir, "standardrun", model_config) :
        joinpath(outputdir, "standardrun", model_config, gpu_tag)
    mkpath(age_output_dir)

    # One output writer per field (separate files for each).
    # Distributed grids: only output age (Center,Center,Center) — Face-located
    # fields (u,v,w,eta) trigger a BoundsError in _fill_north_send_buffer!
    # on distributed tripolar grids (upstream Oceananigans bug).
    is_distributed = !isempty(gpu_tag)
    output_defs = if is_distributed
        Dict("age" => model.tracers.age)
    else
        Dict(
            "age" => model.tracers.age,
            "u" => model.velocities.u,
            "v" => model.velocities.v,
            "w" => model.velocities.w,
            "eta" => model.free_surface.displacement,
        )
    end
    for (name, field) in output_defs
        output_prefix = joinpath(age_output_dir, "$(name)_$(duration_tag)")
        simulation.output_writers[Symbol(name)] = JLD2Writer(
            model, Dict(name => field);
            schedule = TimeInterval(output_interval),
            filename = output_prefix,
            file_splitting = TimeInterval(output_interval),
            with_halos = false,
            overwrite_existing = true,
            including = [],
        )
    end

    @info "Simulation configured: stop_time=$(stop_time / year) yr, output_dir=$age_output_dir"
    flush(stdout); flush(stderr)

    return simulation, age_output_dir
end


################################################################################
# Age field validation (post-simulation diagnostics)
################################################################################

"""
    validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label="simulation")

Run 5 diagnostic tests on the age field after a simulation:
1. Max age bound check (should not exceed elapsed time by >10%)
2. Surface age near zero (surface relaxation working)
3. Non-negativity (advection scheme oscillations)
4. Depth-averaged profile (per-level statistics)
5. Hotspot inspection (neighbors of max age cell)
"""
function validate_age_field(model, grid, simulation, ADVECTION_SCHEME; label = "simulation")
    @info "Validating age field after $label"
    flush(stdout); flush(stderr)

    age_data = Array(interior(model.tracers.age))
    elapsed_time = time(simulation)

    (; wet3D, idx, Nidx) = compute_wet_mask(grid)
    Nx′, Ny′, Nz′ = size(wet3D)
    age_wet = age_data[idx]

    # ── Test 1: Max age bound ────────────────────────────────────────────────
    max_age_val = maximum(age_wet)
    max_age_ratio = max_age_val / elapsed_time
    @info "Max age bound check:" max_age_years = max_age_val / year ratio_to_elapsed = max_age_ratio
    if max_age_ratio > 1.1
        @warn "Max age exceeds 1.1× elapsed time — possible numerical overshoot or bug"
    end

    # ── Test 2: Surface age should be near zero ──────────────────────────────
    surface_mask = wet3D[:, :, end]
    surface_ages = age_data[:, :, end][surface_mask]
    max_surface_age = maximum(abs, surface_ages)
    mean_surface_age = mean(surface_ages)
    @info "Surface age:" max_days = max_surface_age / day mean_days = mean_surface_age / day
    if max_surface_age > 1day
        @warn "Surface age exceeds 1 day — surface relaxation may not be working correctly"
    end

    # ── Test 3: Non-negativity ───────────────────────────────────────────────
    n_negative = count(x -> x < 0, age_wet)
    min_age_val = minimum(age_wet)
    @info "Non-negativity:" n_negative fraction = n_negative / Nidx min_age_days = min_age_val / day
    if n_negative > 0
        @warn "Found $n_negative negative age values (min = $(min_age_val / day) days) — advection scheme ($ADVECTION_SCHEME) may produce oscillations"
    end

    # ── Test 4: Depth-averaged profile ───────────────────────────────────────
    z_centers = Array(znodes(grid, Center(), Center(), Center()))
    @info "Volume-weighted mean age by depth level:"
    flush(stdout); flush(stderr)
    for k in Nz′:-1:1
        level_mask = wet3D[:, :, k]
        n_wet = count(level_mask)
        if n_wet == 0
            continue
        end
        level_ages = age_data[:, :, k][level_mask]
        level_mean = mean(level_ages)
        level_max = maximum(level_ages)
        level_min = minimum(level_ages)
        z_val = z_centers[k]
        @info @sprintf(
            "  k=%3d  z=%7.0fm  mean=%6.2f yr  max=%6.2f yr  min=%6.2f yr  n_wet=%d",
            k, z_val, level_mean / year, level_max / year, level_min / year, n_wet
        )
    end
    flush(stdout); flush(stderr)

    # ── Test 5: Hotspot inspection ───────────────────────────────────────────
    max_idx = argmax(age_data)
    mi, mj, mk = Tuple(max_idx)
    @info "Max age hotspot at (i=$mi, j=$mj, k=$mk):" age_years = age_data[mi, mj, mk] / year z_meters = z_centers[mk]
    for dk in -1:1, dj in -1:1, di in -1:1
        ni, nj, nk = mi + di, mj + dj, mk + dk
        if 1 ≤ ni ≤ Nx′ && 1 ≤ nj ≤ Ny′ && 1 ≤ nk ≤ Nz′ && wet3D[ni, nj, nk]
            @info @sprintf("  neighbor (%+d,%+d,%+d): age = %6.2f yr", di, dj, dk, age_data[ni, nj, nk] / year)
        end
    end
    flush(stdout); flush(stderr)

    # ── Summary ──────────────────────────────────────────────────────────────
    @info "Validation summary:" max_age_years = max_age_val / year mean_wet_age_years = mean(age_wet) / year max_surface_days = max_surface_age / day n_negative n_wet_cells = Nidx
    flush(stdout); flush(stderr)

    return nothing
end


################################################################################
# Matrix processing helpers
################################################################################

"""
    process_sparse_matrix(M, MATRIX_PROCESSING) -> SparseMatrixCSC

Apply the requested matrix processing to M. Valid values for MATRIX_PROCESSING:
- "raw": no processing
- "symfill": add zero entries at (j,i) for every existing (i,j)
- "dropzeros": remove stored zeros
- "symdrop": keep (i,j) only if (j,i) also exists, then drop zeros
"""
function process_sparse_matrix(M, MATRIX_PROCESSING)
    if MATRIX_PROCESSING == "symfill"
        I_idx, J_idx, V = findnz(M)
        M = sparse([I_idx; J_idx], [J_idx; I_idx], [V; zeros(length(V))], size(M)...)
        @info "After symfill: nnz=$(nnz(M))"
    elseif MATRIX_PROCESSING == "dropzeros"
        nnz_before = nnz(M)
        dropzeros!(M)
        @info "After dropzeros: nnz $nnz_before → $(nnz(M))"
    elseif MATRIX_PROCESSING == "symdrop"
        dropzeros!(M)
        M_t = copy(M')
        nnz_before = nnz(M)
        M = M .* (M_t .!= 0)
        dropzeros!(M)
        @info "After symdrop: nnz $nnz_before → $(nnz(M))"
    else
        @info "No matrix processing applied (raw)"
    end
    flush(stdout); flush(stderr)
    return M
end

"""
    compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; LUMP_AND_SPRAY=false)

If LUMP_AND_SPRAY is true, compute LUMP, SPRAY, Mc and save to matrices_dir.
Returns (; Mc, LUMP, SPRAY) — all `nothing` if LUMP_AND_SPRAY is false.
"""
function compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; LUMP_AND_SPRAY = false)
    if LUMP_AND_SPRAY
        @info "Computing LUMP and SPRAY matrices"
        flush(stdout); flush(stderr)
        LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
        @info "LUMP ($(size(LUMP, 1))×$(size(LUMP, 2)), nnz=$(nnz(LUMP)))"
        @info "SPRAY ($(size(SPRAY, 1))×$(size(SPRAY, 2)), nnz=$(nnz(SPRAY)))"
        Mc = LUMP * M * SPRAY
        @info "Coarsened matrix Mc ($(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc)))"

        jldsave(joinpath(matrices_dir, "LUMP.jld2"); LUMP)
        jldsave(joinpath(matrices_dir, "SPRAY.jld2"); SPRAY)
        jldsave(joinpath(matrices_dir, "Mc.jld2"); Mc)
        flush(stdout); flush(stderr)
        return (; Mc, LUMP, SPRAY)
    else
        @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no)"
        flush(stdout); flush(stderr)
        return (; Mc = nothing, LUMP = nothing, SPRAY = nothing)
    end
end

################################################################################
# Analysis utilities: zonal averages and horizontal slices
################################################################################

"""
    compute_ocean_basin_masks(grid, wet3D) -> (; ATL, PAC, IND)

Compute Atlantic, Pacific, and Indian ocean basin masks using OceanBasins.jl.
Returns a named tuple of 2D Bool arrays sized (Nx', Ny') where (Nx', Ny')
are the interior dimensions from `wet3D` (excludes tripolar fold point).

Requires `OCEANS, isatlantic, ispacific, isindian` from OceanBasins in scope.
"""
function compute_ocean_basin_masks(grid, wet3D)
    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′ = size(wet3D)[1:2]
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
    zonalaverage!(za, xw, w, x3D, v3D, mask3D)

In-place volume-weighted zonal average using preallocated buffers.
`mask3D` must be 3D (reshape 2D masks before calling).
Writes result into `za` (Ny, Nz) and uses `xw`, `w` as (Nx, Ny, Nz) scratch space.
"""
function zonalaverage!(za, xw, w, x3D, v3D, mask3D)
    @. xw = ifelse(isnan(x3D) | !mask3D, 0.0, x3D * v3D)
    @. w = ifelse(isnan(x3D) | !mask3D, 0.0, v3D)
    @views for j in axes(za, 1), k in axes(za, 2)
        num = 0.0
        den = 0.0
        for i in axes(xw, 1)
            num += xw[i, j, k]
            den += w[i, j, k]
        end
        za[j, k] = den > 0 ? num / den : NaN
    end
    return za
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
                         colorrange=(0, 1500), levels=0:100:1500, colormap=:viridis)

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
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        lowclip = colormap[1],
        highclip = colormap[end],
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
    basins = compute_ocean_basin_masks(grid, wet3D)
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

        cf = contourf!(ax, lat_repr, depth_vals, za; levels, colormap, nan_color = :lightgray, extendhigh = :auto, extendlow = :auto)
        translate!(cf, 0, 0, -100)
        xlims!(ax, -90, 90)
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

        hm = heatmap!(ax, slice; colorrange, colormap, nan_color = :black, lowclip, highclip)
        Colorbar(fig[1, 2], hm; label = "Age (years)")

        outputfile = joinpath(output_dir, "$(label)_slice_$(depth)m.png")
        @info "Saving $outputfile"
        save(outputfile, fig)
    end

    @info "Age diagnostic plots saved to $output_dir"
    flush(stdout); flush(stderr)
    return nothing
end


################################################################################
# Age animation helpers (zonal averages + depth slices)
#
# Requires CairoMakie and OceanBasins symbols in the calling script's scope,
# as well as Oceananigans (Units, Grids, ImmersedBoundaries).
################################################################################

"""
    animate_zonal_averages(age_fts, grid, wet3D, vol_3D, output_dir, prefix;
                           colorrange, levels, colormap, n_frames=144, framerate=24)

Animate 4 zonal-average MP4s (global, Atlantic, Pacific, Indian) from a
`FieldTimeSeries`.  Each frame interpolates to the corresponding time, converts
to years, and computes volume-weighted zonal averages per basin.
"""
function animate_zonal_averages(
        age_fts, grid, wet3D, vol_3D, output_dir, prefix;
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        n_frames = 144,
        framerate = 24,
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    ug = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx′, Ny′, Nz′ = size(wet3D)
    lat = Array(ug.φᶜᶜᵃ[1:Nx′, 1:Ny′])
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z
    lat_repr = dropdims(mean(lat; dims = 1); dims = 1)

    basins = compute_ocean_basin_masks(grid, wet3D)
    global_mask = trues(Nx′, Ny′)
    basin_configs = [
        ("global", global_mask),
        ("atlantic", basins.ATL),
        ("pacific", basins.PAC),
        ("indian", basins.IND),
    ]

    stop_time = age_fts.times[end]
    frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

    age_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)
    xw_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)
    w_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)
    za_buf = Array{Float64}(undef, Ny′, Nz′)

    # Build figure once; update observables per basin
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    first_mask = reshape(basin_configs[1][2], size(basin_configs[1][2], 1), size(basin_configs[1][2], 2), 1)
    zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, first_mask)
    za_obs = Observable(copy(za_buf))
    title_obs = Observable("")

    fig = Figure(; size = (800, 500))
    ax = Axis(
        fig[1, 1];
        title = title_obs,
        xlabel = "Latitude",
        ylabel = "Depth (m)",
        backgroundcolor = :lightgray,
        xgridvisible = false,
        ygridvisible = false,
    )

    cf = contourf!(
        ax, lat_repr, depth_vals, za_obs;
        levels, colormap, nan_color = :lightgray, extendhigh = :auto, extendlow = :auto
    )
    translate!(cf, 0, 0, -100)
    xlims!(ax, -90, 90)
    ylims!(ax, maximum(depth_vals), 0)
    Colorbar(fig[1, 2], cf; label = "Age (years)")

    for (basin_name, basin_mask) in basin_configs
        @info "Animating zonal average — $basin_name"
        flush(stdout); flush(stderr)

        mask3D = reshape(basin_mask, size(basin_mask, 1), size(basin_mask, 2), 1)

        # Reset to first frame for this basin
        age_raw = interior(age_fts[Time(frame_times[1])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
        za_obs.val .= za_buf
        notify(za_obs)
        title_obs[] = @sprintf("age — %s zonal avg (t = 0.0 months)", basin_name)

        filepath = joinpath(output_dir, "$(prefix)_zonal_avg_$(basin_name).mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            age_raw = interior(age_fts[Time(frame_times[i])])
            @. age_buf = ifelse(wet3D, age_raw / year, NaN)
            zonalaverage!(za_buf, xw_buf, w_buf, age_buf, vol_3D, mask3D)
            za_obs.val .= za_buf
            notify(za_obs)
            title_obs[] = @sprintf("age — %s zonal avg (t = %.1f months)", basin_name, frame_times[i] / (year / 12))
        end

        @info "Saved $filepath"
    end

    return nothing
end

"""
    animate_depth_slices(age_fts, grid, wet3D, output_dir, prefix;
                         colorrange, levels, colormap, n_frames=144, framerate=24)

Animate 6 depth-slice MP4s (100, 200, 500, 1000, 2000, 3000 m) from a
`FieldTimeSeries`.  Each frame interpolates to the corresponding time and
extracts the nearest depth level.
"""
function animate_depth_slices(
        age_fts, grid, wet3D, output_dir, prefix;
        colorrange = (0, 1500),
        levels = 0:100:1500,
        colormap = cgrad(:viridis, length(levels) - 1, categorical = true),
        lowclip = colormap[1],
        highclip = colormap[end],
        n_frames = 144,
        framerate = 24,
    )
    mkpath(output_dir)

    year = 365.25 * 86400  # seconds

    Nx′, Ny′, Nz′ = size(wet3D)
    z = znodes(grid, Center(), Center(), Center())
    depth_vals = -z

    target_depths = [100, 200, 500, 1000, 2000, 3000]
    depth_k_indices = [(d, find_nearest_depth_index(grid, d)) for d in target_depths]

    stop_time = age_fts.times[end]
    frame_times = range(0, stop_time; length = n_frames + 1)[1:n_frames]

    age_buf = Array{Float64}(undef, Nx′, Ny′, Nz′)

    # Build figure once; update observables per depth
    age_raw = interior(age_fts[Time(frame_times[1])])
    @. age_buf = ifelse(wet3D, age_raw / year, NaN)
    slice_obs = Observable(age_buf[:, :, depth_k_indices[1][2]])
    title_obs = Observable("")

    fig = Figure(; size = (1000, 500))
    ax = Axis(fig[1, 1]; title = title_obs)

    hm = heatmap!(
        ax, slice_obs; colorrange, colormap, nan_color = :black,
        lowclip, highclip
    )
    Colorbar(fig[1, 2], hm; label = "Age (years)")

    for (depth, k) in depth_k_indices
        @info "Animating depth slice — $(depth) m"
        flush(stdout); flush(stderr)

        actual_depth = round(depth_vals[k]; digits = 1)

        # Reset to first frame for this depth
        age_raw = interior(age_fts[Time(frame_times[1])])
        @. age_buf = ifelse(wet3D, age_raw / year, NaN)
        slice_obs[] = @view(age_buf[:, :, k])
        title_obs[] = @sprintf("age at %d m (k=%d, z=%.1f m, t = 0.0 months)", depth, k, actual_depth)

        filepath = joinpath(output_dir, "$(prefix)_slice_$(depth)m.mp4")
        record(fig, filepath, 1:n_frames; framerate) do i
            age_raw = interior(age_fts[Time(frame_times[i])])
            @. age_buf = ifelse(wet3D, age_raw / year, NaN)
            slice_obs[] = @view(age_buf[:, :, k])
            title_obs[] = @sprintf("age at %d m (k=%d, z=%.1f m, t = %.1f months)", depth, k, actual_depth, frame_times[i] / (year / 12))
        end

        @info "Saved $filepath"
    end

    return nothing
end
