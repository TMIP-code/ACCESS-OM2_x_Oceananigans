"""
Common setup for transport-matrix Jacobian computation.

Included by `create_matrix.jl` and `test_linearity.jl`.  After including this
file the caller has access to:

  grid, idx, Nidx, wet3D, Nx′, Ny′, Nz′,
  jac_prep, jac_buffer, sparse_forward_backend,
  ADcvec, GADcvec, ADc_buf, GADc_buf,
  mytendency!, matrices_dir, matrix_plots_dir,
  parentmodel, model_config, outputdir, year, month
"""

@info "Loading packages and functions"
flush(stdout); flush(stderr)

using Oceananigans

# Matrix build always runs on the CPU: sparsity detection and graph colouring
# cannot be performed on the GPU.
arch = CPU()
arch_str = "CPU"
@info "Using $arch architecture"
flush(stdout); flush(stderr)

using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency,
    _update_zstar_scaling!, surface_kernel_parameters
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, FieldBoundaryConditions,
    fill_halo_regions!
using Oceananigans.Grids: znode, get_active_cells_map
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
month = months = year / 12

using Adapt: adapt
using Statistics
using LinearAlgebra
using SparseArrays
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using TOML
using JLD2
using Printf
using CairoMakie
using KernelAbstractions: @kernel, @index
using DifferentiationInterface
using DifferentiationInterface: Cache, jacobian_sparsity_with_contexts
using SparseConnectivityTracer
using ADTypes: KnownJacobianSparsityDetector
using ForwardDiff: ForwardDiff
using SparseMatrixColorings

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

(; parentmodel, experiment_dir, monthly_dir, yearly_dir, outputdir, Δt_seconds) = load_project_config()
Δt = Δt_seconds * second

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

@info "Run configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- model_config     = $model_config"
flush(stdout); flush(stderr)
matrices_dir = joinpath(outputdir, "TM", model_config)
matrix_plots_dir = joinpath(matrices_dir, "plots")
const_dir = joinpath(matrices_dir, "const")
const_plots_dir = joinpath(const_dir, "plots")
mkpath(matrices_dir)
mkpath(matrix_plots_dir)
mkpath(const_dir)
mkpath(const_plots_dir)
@show outputdir
@show matrices_dir
flush(stdout); flush(stderr)

################################################################################
# Load grid
################################################################################

@info "Reconstructing grid (loading data from JLD2)"
flush(stdout); flush(stderr)
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout); flush(stderr)

################################################################################
# Load time-averaged (constant) velocity fields
################################################################################

@info "Loading time-averaged (yearly) velocity and η fields"
flush(stdout); flush(stderr)

if VELOCITY_SOURCE == "cgridtransports"
    u_constant_file = joinpath(yearly_dir, "u_from_mass_transport_yearly.jld2")
    v_constant_file = joinpath(yearly_dir, "v_from_mass_transport_yearly.jld2")
    @info """Loading yearly velocities from mass-transport files:
    - $(u_constant_file)
    - $(v_constant_file)
    """
elseif VELOCITY_SOURCE == "bgridvelocities"
    u_constant_file = joinpath(yearly_dir, "u_interpolated_yearly.jld2")
    v_constant_file = joinpath(yearly_dir, "v_interpolated_yearly.jld2")
    @info """Loading yearly velocities from B-grid interpolated files:
    - $(u_constant_file)
    - $(v_constant_file)
    """
end
η_constant_file = joinpath(yearly_dir, "eta_yearly.jld2")
flush(stdout); flush(stderr)

# Re-use the same boundary conditions as create_velocities.jl
ubcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = FPivotZipperBoundaryCondition(-1))
vbcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = FPivotZipperBoundaryCondition(-1))

u_constant = XFaceField(grid; boundary_conditions = ubcs)
set!(u_constant, load(u_constant_file, "u"))
fill_halo_regions!(u_constant)
@show u_constant

v_constant = YFaceField(grid; boundary_conditions = vbcs)
set!(v_constant, load(v_constant_file, "v"))
fill_halo_regions!(v_constant)
@show v_constant

η_constant = Field{Center, Center, Nothing}(grid)
set!(η_constant, load(η_constant_file, "η"))
fill_halo_regions!(η_constant)
@show η_constant

@info "Constant velocities and η loaded"
flush(stdout); flush(stderr)

################################################################################
# Prescribed velocities and free surface
################################################################################

if W_FORMULATION == "wprescribed"
    w_constant_file = VELOCITY_SOURCE == "cgridtransports" ?
        joinpath(yearly_dir, "w_from_mass_transport_yearly.jld2") :
        joinpath(yearly_dir, "w_yearly.jld2")
    @info "Using prescribed w field from: $w_constant_file"
    flush(stdout); flush(stderr)
    w_constant = CenterField(grid)
    set!(w_constant, load(w_constant_file, "w"))
    fill_halo_regions!(w_constant)
    velocities = PrescribedVelocityFields(u = u_constant, v = v_constant, w = w_constant)
elseif W_FORMULATION == "wdiagnosed"
    @info "Prescribing u, v (constant); diagnosing w via continuity"
    flush(stdout); flush(stderr)
    velocities = PrescribedVelocityFields(u = u_constant, v = v_constant, formulation = DiagnosticVerticalVelocity())
end
free_surface = PrescribedFreeSurface(displacement = η_constant)

################################################################################
# Closures (explicit only — required for Jacobian via ForwardDiff)
################################################################################

@info "Creating closures"
flush(stdout); flush(stderr)

# Vertical diffusivity parameters (match run_ACCESS-OM2.jl)
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# TODO: replace with monthly MLD (time-dependent κ) once implemented
mld_ds = open_dataset(joinpath(yearly_dir, "mld_yearly.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

explicit_vertical_diffusion = VerticalScalarDiffusivity(ExplicitTimeDiscretization(); κ = κVField)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)

explicit_closure = (horizontal_diffusion, explicit_vertical_diffusion)

@info "Closures created"
flush(stdout); flush(stderr)

################################################################################
# Jacobian model
################################################################################

@info "Building Jacobian model"
flush(stdout); flush(stderr)

age_parameters = (;
    relaxation_timescale = 3Δt,
    source_rate = 1.0,
)

@inline linear_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.ADc[i, j, k] / params.relaxation_timescale, 0.0)

linear_dynamics = Forcing(linear_source_sink; parameters = age_parameters, discrete_form = true)
linear_forcing = (; ADc = linear_dynamics)

ADc0 = CenterField(grid)

jacobian_model_kwargs = (
    timestepper = timestepper_from_string(TIMESTEPPER),
    tracer_advection = advection_from_scheme(ADVECTION_SCHEME),
    velocities = velocities,
    free_surface = free_surface,
    tracers = (; ADc = ADc0),
    closure = explicit_closure,
    forcing = linear_forcing,
)

jacobian_model = HydrostaticFreeSurfaceModel(grid; jacobian_model_kwargs...)

################################################################################
# Initialise model state (update zstar and halo regions)
################################################################################

@info "Initialising model state (zstar scaling from constant η)"
flush(stdout); flush(stderr)

launch!(CPU(), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η_constant, grid)
fill_halo_regions!(jacobian_model.tracers.ADc)

@info "Model state initialised"
flush(stdout); flush(stderr)

################################################################################
# Autodiff setup
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

function mytendency!(GADcvec, ADcvec, ADc_field, GADc_field)
    # Fill the field's interior directly from the vector
    interior(ADc_field) .= 0
    for (n, ijk) in enumerate(idx)
        interior(ADc_field)[ijk] = ADcvec[n]
    end
    fill_halo_regions!(ADc_field)

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
        (; ADc = ADc_field),
        jacobian_model.closure_fields,
        jacobian_model.auxiliary_fields,
        jacobian_model.clock,
        c_forcing,
    )

    launch!(
        CPU(), grid, kernel_parameters,
        compute_hydrostatic_free_surface_GADc!,
        GADc_field, grid, args;
        active_cells_map,
    )

    # Fill output vector with interior wet values
    GADcvec .= view(interior(GADc_field), idx)
    return GADcvec
end

# Preallocate field buffers for Cache
ADc_buf = CenterField(grid)
GADc_buf = CenterField(grid)

@info "Benchmarking tendency function"
flush(stdout); flush(stderr)
ADcvec = ones(Nidx)
GADcvec = similar(ADcvec)
mytendency!(GADcvec, ADcvec, ADc_buf, GADc_buf)
@time "Tendency evaluation" mytendency!(GADcvec, ADcvec, ADc_buf, GADc_buf)

# Step 1: Detect sparsity pattern (expensive tracing pass)
@info "Detecting sparsity pattern..."
flush(stdout); flush(stderr)
@time "Detect sparsity" S = jacobian_sparsity_with_contexts(
    mytendency!, GADcvec, TracerSparsityDetector(; gradient_pattern_type = Set{UInt}), ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)

# Step 2: Symmetrize sparsity pattern (S[i,j] ↔ S[j,i])
S_sym = S .| S'
@info "Sparsity: nnz(S) = $(nnz(S)), nnz(S_sym) = $(nnz(S_sym))"
flush(stdout); flush(stderr)

# Step 3: Prepare Jacobian with known (symmetric) sparsity pattern
sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = KnownJacobianSparsityDetector(S_sym),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

@info "Preparing Jacobian..."
flush(stdout); flush(stderr)
@time "Prepare Jacobian" jac_prep = prepare_jacobian(
    mytendency!, GADcvec, sparse_forward_backend, ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)
S_final = sparsity_pattern(jac_prep)
@info "Sparsity pattern: $(size(S_final, 1))×$(size(S_final, 2)), nnz=$(nnz(S_final)), $(maximum(column_colors(jac_prep))) colors"
flush(stdout); flush(stderr)
jac_buffer = similar(S_final, eltype(ADcvec))
