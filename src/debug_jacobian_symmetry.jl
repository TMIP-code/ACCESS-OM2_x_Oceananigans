"""
Minimal 2-D (x-z) script to reproduce and inspect the structural asymmetry in the
Jacobian produced by `create_matrix.jl`.

Grid: Nx=10, Ny=1, Nz=8  (Periodic x, Periodic y, Bounded z)
- MutableVerticalDiscretization (z-star)
- PartialCellBottom with a linear bottom depth (shallow on the left, full depth on the right)
- Prescribed u=1, v=0, η=1 (constant)
- Same explicit closures and linear surface forcing as create_matrix.jl
- Same autodiff Jacobian pipeline

Usage:
```
julia --project src/debug_jacobian_symmetry.jl
```
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    hydrostatic_free_surface_tracer_tendency, _update_zstar_scaling!, surface_kernel_parameters
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Grids: on_architecture, MutableVerticalDiscretization
using Oceananigans.Utils: KernelParameters, launch!, get_active_cells_map
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Units: second, seconds, day, days

using KernelAbstractions: @kernel, @index
using LinearAlgebra
using SparseArrays
using DifferentiationInterface
using DifferentiationInterface: overloaded_input_type
using SparseConnectivityTracer
using ForwardDiff: ForwardDiff
using SparseMatrixColorings

include("shared_functions.jl")

################################################################################
# Grid
################################################################################

arch = CPU()

Nx = 4
Ny = 1
Nz = 4
H = 100.0  # total depth (m)
Lx = 1.0e6  # domain extent in x (m)
Ly = 1.0e5  # domain extent in y (m)
Δt = 5400.0seconds

@info "Building grid: Nx=$Nx, Ny=$Ny, Nz=$Nz, H=$H m"
flush(stdout); flush(stderr)

# MutableVerticalDiscretization makes the grid a MutableGridOfSomeKind,
# enabling _update_zstar_scaling! to work exactly as in create_matrix.jl.
z_faces = collect(range(-H, 0, length = Nz + 1))
underlying_grid = RectilinearGrid(
    arch;
    size = (Nx, Nz),
    x = (0, Lx),
    # y = (0, Ly),
    z = MutableVerticalDiscretization(z_faces),
    topology = (Periodic, Flat, Bounded),
    halo = (4, 4),
)

# Linear bottom: -0.5H at i=1 (shallow/left) to -H at i=Nx (full depth/right).
# This creates non-trivial partial cells that vary across the domain.
bottom_height = [-H * (0.5 + 0.5 * (i - 1) / (Nx - 1)) for i in 1:Nx, j in 1:Ny]

grid = ImmersedBoundaryGrid(
    underlying_grid,
    PartialCellBottom(bottom_height);
    active_cells_map = true,
    active_z_columns = true,
)
@info "Grid built"
@show grid
flush(stdout); flush(stderr)

################################################################################
# Prescribed velocity and free-surface fields
################################################################################

@info "Setting up prescribed fields"
flush(stdout); flush(stderr)

u_const = XFaceField(grid)
v_const = YFaceField(grid)
w_const = ZFaceField(grid)
η_const = Field{Center, Center, Nothing}(grid)

# set!(u_const, 1.0)
set!(u_const, 0.0)
fill_halo_regions!(u_const)
set!(v_const, 0.0)
fill_halo_regions!(v_const)
set!(w_const, 0.0)
fill_halo_regions!(w_const)
# set!(η_const, 1.0)  # 1 m free-surface displacement → non-trivial σ scaling
set!(η_const, 0.0)  # 1 m free-surface displacement → non-trivial σ scaling
fill_halo_regions!(η_const)
@show u_const
@show v_const
@show w_const
@show η_const

# w is diagnosed from u and v via the continuity equation
# velocities = PrescribedVelocityFields(u = u_const, v = v_const, formulation = DiagnosticVerticalVelocity())
velocities = PrescribedVelocityFields(u = u_const, v = v_const, w = w_const)
free_surface = PrescribedFreeSurface(displacement = η_const)

################################################################################
# Closures (explicit only — required for Jacobian via ForwardDiff)
################################################################################

@info "Creating closures"
flush(stdout); flush(stderr)

mld_depth = -0.2 * H   # top 20 % of the column is the "mixed layer"
κVML = 0.1              # m²/s in the mixed layer
κVBG = 3.0e-5           # m²/s in the interior

κVField = CenterField(grid)
# set!(κVField, (x, y, z) -> z > mld_depth ? κVML : κVBG)
set!(κVField, (x, z) -> z > mld_depth ? κVML : κVBG)

explicit_vertical_diffusion = VerticalScalarDiffusivity(ExplicitTimeDiscretization(); κ = κVField)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
# horizontal_diffusion = HorizontalScalarDiffusivity(κ = 0.0)
# explicit_closure = (horizontal_diffusion, explicit_vertical_diffusion)
explicit_closure = (horizontal_diffusion,)
# explicit_closure = (explicit_vertical_diffusion,)

################################################################################
# Jacobian model (same structure as create_matrix.jl)
################################################################################

@info "Building Jacobian model"
flush(stdout); flush(stderr)

age_parameters = (; relaxation_timescale = 3Δt, source_rate = 1.0)

@inline linear_source_sink(i, j, k, grid, clock, fields, params) =
    ifelse(k ≥ grid.Nz, -fields.ADc[i, j, k] / params.relaxation_timescale, 0.0)

linear_dynamics = Forcing(linear_source_sink; parameters = age_parameters, discrete_form = true)
linear_forcing = (; ADc = linear_dynamics)

ADc0 = CenterField(grid)

jacobian_model = HydrostaticFreeSurfaceModel(
    grid;
    tracer_advection = Centered(order = 2),
    velocities = velocities,
    free_surface = free_surface,
    tracers = (; ADc = ADc0),
    closure = explicit_closure,
    forcing = linear_forcing,
)

################################################################################
# Initialise model state (update z-star scaling from constant η)
################################################################################

@info "Initialising z-star scaling from constant η"
flush(stdout); flush(stderr)

launch!(CPU(), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η_const, grid)
fill_halo_regions!(jacobian_model.tracers.ADc)

@info "Model state initialised"
flush(stdout); flush(stderr)

################################################################################
# Autodiff setup (mirrored from create_matrix.jl)
################################################################################

@info "Setting up autodiff for Jacobian computation"
flush(stdout); flush(stderr)

@warn "Adding newton_div method to allow sparsity tracer to pass through WENO"
autodifftypes = Union{SparseConnectivityTracer.AbstractTracer, SparseConnectivityTracer.Dual, ForwardDiff.Dual}
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(::Type{FT}, a::FT, b) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a, b::FT) where {FT <: autodifftypes} = a / b
@inline Oceananigans.Utils.newton_div(inv_FT, a::FT, b) where {FT <: autodifftypes} = a / b

Nx′, Ny′, Nz′ = size(ADc0)
N′ = Nx′ * Ny′ * Nz′
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
@info "Number of wet cells: Nidx = $Nidx"
flush(stdout); flush(stderr)

kernel_parameters = KernelParameters(1:Nx′, 1:Ny′, 1:Nz′)
active_cells_map = get_active_cells_map(grid, Val(:interior))

function mytendency!(GADcvec::Vector{T}, ADcvec::Vector{T}, clock) where {T}
    ADc3D = zeros(T, Nx′, Ny′, Nz′)
    ADc3D[idx] .= ADcvec

    ADc = CenterField(grid, T)
    set!(ADc, ADc3D)

    GADc = CenterField(grid, T)

    c_advection = jacobian_model.advection[:ADc]
    c_forcing = jacobian_model.forcing[:ADc]
    c_immersed_bc = immersed_boundary_condition(jacobian_model.tracers[:ADc])

    args = tuple(
        Val(1),
        Val(:ADc),
        c_advection,
        jacobian_model.closure,
        c_immersed_bc,
        jacobian_model.buoyancy,
        jacobian_model.biogeochemistry,
        jacobian_model.transport_velocities,
        jacobian_model.free_surface,
        (; ADc = ADc),
        jacobian_model.closure_fields,
        jacobian_model.auxiliary_fields,
        clock,
        c_forcing,
    )

    launch!(
        arch, grid, kernel_parameters,
        compute_hydrostatic_free_surface_GADc!,
        GADc, grid, args;
        active_cells_map,
    )

    GADcvec .= view(interior(on_architecture(CPU(), GADc)), idx)
    return GADcvec
end

@info "Benchmarking tendency function"
flush(stdout); flush(stderr)
ADcvec = ones(Nidx)
GADcvec = ones(Nidx)
@time mytendency!(GADcvec, ADcvec, 0.0)
@time mytendency!(GADcvec, ADcvec, 0.0)

sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(; gradient_pattern_type = Set{UInt}),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

@time "Prepare Jacobian sparsity pattern" jac_prep_sparse = prepare_jacobian(
    mytendency!,
    GADcvec,
    sparse_forward_backend,
    ADcvec,
    Constant(0.0);
    strict = Val(false),
)

DualType = eltype(overloaded_input_type(jac_prep_sparse))
ADc3D_dual = zeros(DualType, Nx′, Ny′, Nz′)
ADc_dual = CenterField(grid, DualType)
Gc_dual = CenterField(grid, DualType)

function mytendency_preallocated!(GADcvec::Vector{DualType}, ADcvec::Vector{DualType}, clock)
    ADc3D_dual[idx] .= ADcvec
    set!(ADc_dual, ADc3D_dual)

    c_advection = jacobian_model.advection[:ADc]
    c_forcing = jacobian_model.forcing[:ADc]
    c_immersed_bc = immersed_boundary_condition(jacobian_model.tracers[:ADc])

    args = tuple(
        Val(1),
        Val(:ADc),
        c_advection,
        jacobian_model.closure,
        c_immersed_bc,
        jacobian_model.buoyancy,
        jacobian_model.biogeochemistry,
        jacobian_model.transport_velocities,
        jacobian_model.free_surface,
        (; ADc = ADc_dual),
        jacobian_model.closure_fields,
        jacobian_model.auxiliary_fields,
        clock,
        c_forcing,
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_GADc!,
        Gc_dual, grid, args;
        active_cells_map,
    )

    GADcvec .= view(interior(Gc_dual), idx)
    return GADcvec
end

@time "Prepare buffer for Jacobian" jac_buffer = similar(sparsity_pattern(jac_prep_sparse), eltype(ADcvec))

################################################################################
# Compute Jacobian
################################################################################

@info "Computing Jacobian"
flush(stdout); flush(stderr)
@time "Compute Jacobian" jacobian!(
    mytendency_preallocated!,
    GADcvec,
    jac_buffer,
    jac_prep_sparse,
    sparse_forward_backend,
    ADcvec,
    Constant(0.0),
)

M = copy(jac_buffer)  # units: 1/s
display(M)

################################################################################
# Structural symmetry diagnostics
################################################################################

@info "Checking structural symmetry of M"
flush(stdout); flush(stderr)

i_nz, j_nz, _ = findnz(M)
M1 = sparse(i_nz, j_nz, true)
asym = M1 - M1' .> 0

@info "Structural symmetry check:"
@info "  nnz(M)          = $(nnz(M))"
@info "  nnz(asym = M1 - M1') = $(nnz(asym))"
display(asym)

if nnz(asym) > 0
    @warn "Matrix is NOT structurally symmetric! Listing asymmetric pairs:"
    rows_a, cols_a, _ = findnz(asym)
    for (r, c) in zip(rows_a, cols_a)
        ri = idx[r]
        ci = idx[c]
        @info "  M[$r,$c] exists but M[$c,$r] doesn't  →  3D: $(Tuple(ri)) <- $(Tuple(ci) .- Tuple(ri))"
    end
else
    @info "Matrix is structurally symmetric."
end

@info "debug_jacobian_symmetry.jl complete."
flush(stdout); flush(stderr)
