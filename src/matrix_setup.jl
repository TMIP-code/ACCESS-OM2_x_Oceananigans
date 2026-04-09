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
using Oceananigans.TurbulenceClosures: AdvectiveFormulation, DiffusiveFormulation
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
using Oceananigans.BuoyancyFormulations: SeawaterBuoyancy, LinearEquationOfState
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
GM_REDI_STR = lowercase(get(ENV, "GM_REDI", "no"))
GM_REDI_STR == "yes" && (GM_REDI_STR = "diff")  # backward compat
GM_REDI = GM_REDI_STR in ("diff", "adv")
GM_ADVECTIVE = GM_REDI_STR == "adv"
MONTHLY_KAPPAV = lowercase(get(ENV, "MONTHLY_KAPPAV", "no")) == "yes"
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

@info "Run configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- GM_REDI          = $GM_REDI (GM_REDI_STR=$GM_REDI_STR)"
@info "- MONTHLY_KAPPAV   = $MONTHLY_KAPPAV"
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
@show grid

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout); flush(stderr)

################################################################################
# Load time-averaged (constant) velocity fields
################################################################################

@info "Loading time-averaged (yearly) velocity and η fields"
flush(stdout); flush(stderr)

if VELOCITY_SOURCE ∈ ("cgridtransports", "totaltransport")
    vs_prefix = VELOCITY_SOURCE == "totaltransport" ? "total_transport" : "mass_transport"
    u_constant_file = joinpath(yearly_dir, "u_from_$(vs_prefix)_yearly.jld2")
    v_constant_file = joinpath(yearly_dir, "v_from_$(vs_prefix)_yearly.jld2")
    @info """Loading yearly velocities from $(vs_prefix) files:
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

if GM_REDI
    @info "Loading yearly T and S fields for GM-Redi buoyancy"
    flush(stdout); flush(stderr)
    T_constant_file = joinpath(yearly_dir, "temp_yearly.jld2")
    S_constant_file = joinpath(yearly_dir, "salt_yearly.jld2")
    T_constant = CenterField(grid)
    set!(T_constant, load(T_constant_file, "T"))
    fill_halo_regions!(T_constant)
    S_constant = CenterField(grid)
    set!(S_constant, load(S_constant_file, "S"))
    fill_halo_regions!(S_constant)
    @show T_constant
    @show S_constant
end

################################################################################
# Prescribed velocities and free surface
################################################################################

if W_FORMULATION == "wprescribed"
    w_constant_file = if VELOCITY_SOURCE == "totaltransport"
        joinpath(yearly_dir, "w_from_total_transport_yearly.jld2")
    elseif VELOCITY_SOURCE == "cgridtransports"
        joinpath(yearly_dir, "w_from_mass_transport_yearly.jld2")
    else
        joinpath(yearly_dir, "w_yearly.jld2")
    end
    @info "Using prescribed w field from: $w_constant_file"
    flush(stdout); flush(stderr)
    wbcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); north = FPivotZipperBoundaryCondition(1))
    w_constant = ZFaceField(grid; boundary_conditions = wbcs)
    set!(w_constant, load(w_constant_file, "w"))
    fill_halo_regions!(w_constant)
    @show w_constant
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
@show κVField

if GM_REDI
    # No HorizontalScalarDiffusivity — isopycnal κ_symmetric in GM-Redi handles horizontal mixing
    explicit_vertical_diffusion = VerticalScalarDiffusivity(
        ExplicitTimeDiscretization();
        κ = (; T = 0.0, S = 0.0, ADc = κVField)
    )
    gm_formulation = GM_ADVECTIVE ? AdvectiveFormulation() : DiffusiveFormulation()
    # AdvectiveFormulation requires scalar κ_skew (Oceananigans limitation);
    # this is fine since T/S have tracer_advection=nothing and are prescribed.
    gm_κ_skew = GM_ADVECTIVE ? 300.0 : (; T = 0.0, S = 0.0, ADc = 300.0)
    gm_κ_symmetric = GM_ADVECTIVE ? 300.0 : (; T = 0.0, S = 0.0, ADc = 300.0)
    gm_redi = IsopycnalSkewSymmetricDiffusivity(
        skew_flux_formulation = gm_formulation,
        κ_skew = gm_κ_skew,
        κ_symmetric = gm_κ_symmetric,
    )
    explicit_closure = (explicit_vertical_diffusion, gm_redi)
    @info "Closures: vertical + GM-Redi ($gm_formulation) — no horizontal scalar diffusion"
else
    explicit_vertical_diffusion = VerticalScalarDiffusivity(ExplicitTimeDiscretization(); κ = κVField)
    horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)
    explicit_closure = (horizontal_diffusion, explicit_vertical_diffusion)
end

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

if GM_REDI
    tracers = (; ADc = ADc0, T = T_constant, S = S_constant)
    tracer_advection = (; ADc = advection_from_scheme(ADVECTION_SCHEME), T = nothing, S = nothing)
    buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState())
    buoyancy_kw = (; buoyancy)
else
    tracers = (; ADc = ADc0)
    tracer_advection = advection_from_scheme(ADVECTION_SCHEME)
    buoyancy_kw = (;)
end

jacobian_model_kwargs = (
    timestepper = timestepper_from_string(TIMESTEPPER),
    tracer_advection = tracer_advection,
    velocities = velocities,
    free_surface = free_surface,
    tracers = tracers,
    closure = explicit_closure,
    forcing = linear_forcing,
)

jacobian_model = HydrostaticFreeSurfaceModel(grid; jacobian_model_kwargs..., buoyancy_kw...)

################################################################################
# Initialise model state (update zstar and halo regions)
################################################################################

@info "Initialising model state (zstar scaling from constant η)"
flush(stdout); flush(stderr)

if GM_REDI
    set!(jacobian_model.tracers.T, T_constant)
    set!(jacobian_model.tracers.S, S_constant)
end

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

    # Build tracers NamedTuple: must include T/S for GM-Redi buoyancy gradient computation
    if GM_REDI
        tracers_for_tendency = (; ADc = ADc_field, T = jacobian_model.tracers.T, S = jacobian_model.tracers.S)
    else
        tracers_for_tendency = (; ADc = ADc_field)
    end

    iADc = findfirst(==(:ADc), keys(tracers_for_tendency))
    args = tuple(
        Val(iADc),
        Val(:ADc),
        c_advection,
        jacobian_model.closure,
        c_immersed_bc,
        jacobian_model.buoyancy,
        jacobian_model.biogeochemistry,
        jacobian_model.transport_velocities,
        jacobian_model.free_surface,
        tracers_for_tendency,
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
