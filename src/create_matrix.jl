"""
Build the transport matrix (Jacobian of the tracer tendency) from time-averaged
(constant) velocity and free-surface fields produced by `create_velocities.jl`.

Unlike `run_ACCESS-OM2.jl`, no simulation is run: the model is initialised
with constant prescribed fields and the Jacobian is computed in a single pass.
The matrix build always uses the CPU (sparsity detection and coloring require it).

Usage — interactive:
```
qsub -I -P y99 -l mem=190GB -q normal -l walltime=04:00:00 -l ncpus=48 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 \\
     -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/create_matrix.jl")
```

Environment variables:
  PARENT_MODEL      – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE   – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION     – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME  – centered2 | weno3 | weno5  (default: centered2)
  ENABLE_AGE_SOLVE  – true | false  (default: false)
                      When true, solves the linear age equation (coarsened + full)
                      and saves the 3-D steady-state age fields.
"""

@info "Loading packages and functions"
flush(stdout)

using Oceananigans

# Matrix build always runs on the CPU: sparsity detection and graph colouring
# cannot be performed on the GPU.
arch = CPU()
arch_str = "CPU"
@info "Using $arch architecture"
flush(stdout)

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
using NonlinearSolve
using KernelAbstractions: @kernel, @index
using DifferentiationInterface
using DifferentiationInterface: Cache, jacobian_sparsity_with_contexts
using SparseConnectivityTracer
using ADTypes: KnownJacobianSparsityDetector
using ForwardDiff: ForwardDiff
using SparseMatrixColorings
using OceanTransportMatrixBuilder
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
import Pardiso
const nprocs = 48

################################################################################
# Configuration
################################################################################

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
    Δt = parentmodel == "ACCESS-OM2-1" ? 5400seconds : parentmodel == "ACCESS-OM2-025" ? 1800seconds : 400seconds
else
    outputdir = profile["outputdir"]
    Δt = profile["dt_seconds"] * second
end

include("shared_functions.jl")

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)"

parse_env_bool(name, default) = lowercase(strip(get(ENV, name, string(default)))) ∈ ("1", "true", "yes", "on")
ENABLE_AGE_SOLVE = parse_env_bool("ENABLE_AGE_SOLVE", false)

@info "Run configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- ENABLE_AGE_SOLVE = $ENABLE_AGE_SOLVE"
@info "- model_config     = $model_config"
flush(stdout)

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
matrices_dir = joinpath(outputdir, "matrices", model_config)
matrix_plots_dir = joinpath(matrices_dir, "plots")
mkpath(matrices_dir)
mkpath(matrix_plots_dir)
@show outputdir
@show matrices_dir
flush(stdout)

################################################################################
# Load grid
################################################################################

@info "Reconstructing grid (loading data from JLD2)"
flush(stdout)
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout)

################################################################################
# Load time-averaged (constant) velocity fields
################################################################################

@info "Loading time-averaged (constant) velocity and η fields"
flush(stdout)

if VELOCITY_SOURCE == "cgridtransports"
    u_constant_file = joinpath(preprocessed_inputs_dir, "u_from_mass_transport_constant.jld2")
    v_constant_file = joinpath(preprocessed_inputs_dir, "v_from_mass_transport_constant.jld2")
    @info """Loading constant velocities from mass-transport files:
    - $(u_constant_file)
    - $(v_constant_file)
    """
elseif VELOCITY_SOURCE == "bgridvelocities"
    u_constant_file = joinpath(preprocessed_inputs_dir, "u_interpolated_constant.jld2")
    v_constant_file = joinpath(preprocessed_inputs_dir, "v_interpolated_constant.jld2")
    @info """Loading constant velocities from B-grid interpolated files:
    - $(u_constant_file)
    - $(v_constant_file)
    """
end
η_constant_file = joinpath(preprocessed_inputs_dir, "eta_constant.jld2")
flush(stdout)

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
flush(stdout)

################################################################################
# Prescribed velocities and free surface
################################################################################

if W_FORMULATION == "wprescribed"
    w_constant_file = VELOCITY_SOURCE == "cgridtransports" ?
                      joinpath(preprocessed_inputs_dir, "w_from_mass_transport_constant.jld2") :
                      joinpath(preprocessed_inputs_dir, "w_constant.jld2")
    @info "Using prescribed w field from: $w_constant_file"
    flush(stdout)
    w_constant = CenterField(grid)
    set!(w_constant, load(w_constant_file, "w"))
    fill_halo_regions!(w_constant)
    velocities = PrescribedVelocityFields(u = u_constant, v = v_constant, w = w_constant)
elseif W_FORMULATION == "wdiagnosed"
    @info "Prescribing u, v (constant); diagnosing w via continuity"
    flush(stdout)
    velocities = PrescribedVelocityFields(u = u_constant, v = v_constant, formulation = DiagnosticVerticalVelocity())
end
free_surface = PrescribedFreeSurface(displacement = η_constant)

################################################################################
# Closures (explicit only — required for Jacobian via ForwardDiff)
################################################################################

@info "Creating closures"
flush(stdout)

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

# Vertical diffusivity parameters (match run_ACCESS-OM2.jl)
κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

mld_ds = open_dataset(joinpath(inputdir, "mld.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

explicit_vertical_diffusion = VerticalScalarDiffusivity(ExplicitTimeDiscretization(); κ = κVField)
horizontal_diffusion = HorizontalScalarDiffusivity(κ = 300.0)

explicit_closure = (horizontal_diffusion, explicit_vertical_diffusion)

@info "Closures created"
flush(stdout)

################################################################################
# Jacobian model
################################################################################

@info "Building Jacobian model"
flush(stdout)

age_parameters = (;
    relaxation_timescale = 3Δt,
    source_rate = 1.0,
)

@inline linear_source_sink(i, j, k, grid, clock, fields, params) = ifelse(k ≥ grid.Nz, -fields.ADc[i, j, k] / params.relaxation_timescale, 0.0)

linear_dynamics = Forcing(linear_source_sink; parameters = age_parameters, discrete_form = true)
linear_forcing = (; ADc = linear_dynamics)

ADc0 = CenterField(grid)

jacobian_model_kwargs = (
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
flush(stdout)

launch!(CPU(), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η_constant, grid)
fill_halo_regions!(jacobian_model.tracers.ADc)

@info "Model state initialised"
flush(stdout)

################################################################################
# Autodiff setup
################################################################################

@info "Setting up autodiff for Jacobian computation"
flush(stdout)

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
flush(stdout)

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
flush(stdout)
ADcvec = ones(Nidx)
GADcvec = similar(ADcvec)
mytendency!(GADcvec, ADcvec, ADc_buf, GADc_buf)
@time "Tendency evaluation" mytendency!(GADcvec, ADcvec, ADc_buf, GADc_buf)

# Step 1: Detect sparsity pattern (expensive tracing pass)
@info "Detecting sparsity pattern..."
flush(stdout)
@time "Detect sparsity" S = jacobian_sparsity_with_contexts(
    mytendency!, GADcvec, TracerSparsityDetector(; gradient_pattern_type = Set{UInt}), ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)

# Step 2: Symmetrize sparsity pattern (S[i,j] ↔ S[j,i])
S_sym = S .| S'
@info "Sparsity: nnz(S) = $(nnz(S)), nnz(S_sym) = $(nnz(S_sym))"
flush(stdout)

# Step 3: Prepare Jacobian with known (symmetric) sparsity pattern
sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = KnownJacobianSparsityDetector(S_sym),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

@info "Preparing Jacobian..."
flush(stdout)
@time "Prepare Jacobian" jac_prep = prepare_jacobian(
    mytendency!, GADcvec, sparse_forward_backend, ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)
S_final = sparsity_pattern(jac_prep)
@info "Sparsity pattern: $(size(S_final, 1))×$(size(S_final, 2)), nnz=$(nnz(S_final)), $(maximum(column_colors(jac_prep))) colors"
flush(stdout)
jac_buffer = similar(S_final, eltype(ADcvec))

################################################################################
# Compute Jacobian
#
# Single computation — constant (time-averaged) fields mean no monthly loop.
################################################################################

@info "Computing Jacobian (single pass — time-averaged constant fields)"
flush(stdout)
@time "Compute Jacobian" jacobian!(
    mytendency!, GADcvec, jac_buffer, jac_prep,
    sparse_forward_backend, ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)

M = copy(jac_buffer)  # units: 1/s
@info "Jacobian M ($(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M)), density=$(@sprintf("%.2e", nnz(M) / length(M))))"
flush(stdout)
@info "Sparsity pattern of M:"
display(M)

@info "Saving Jacobian to $(matrices_dir)"
flush(stdout)
jldsave(joinpath(matrices_dir, "M.jld2"); M)

fig = Figure()
ax = Axis(fig[1, 1])
plt = spy!(
    0.5 .. size(M, 1) + 0.5,
    0.5 .. size(M, 2) + 0.5,
    M;
    colormap = :coolwarm,
    colorrange = maximum(abs.(M)) .* (-1, 1),
    markersize = size(M, 2) / 1000,
)
ylims!(ax, size(M, 2) + 0.5, 0.5)
Colorbar(fig[1, 2], plt)
save(joinpath(matrix_plots_dir, "M_spy.png"), fig)


################################################################################
# Optional: age solve
################################################################################

if ENABLE_AGE_SOLVE
    # Check structural symmetry of M
    i, j, v = findnz(M)
    M1 = sparse(i, j, true)
    asymmetric_entries = M1 - M1' .> 0
    @info "Non-structurally-symmetric part of M ($(nnz(asymmetric_entries)) asymmetric entries):"
    display(asymmetric_entries)
    ISSTRUCTURALLYSYMMETRIC = Pardiso.isstructurallysymmetric(M)
    @info "M is structurally symmetric: $ISSTRUCTURALLYSYMMETRIC"
    flush(stdout)

    @info "LUMP and SPRAY matrices"
    flush(stdout)

    v1D = interior(compute_volume(grid))[idx]

    LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
    @info "LUMP ($(size(LUMP, 1))×$(size(LUMP, 2)), nnz=$(nnz(LUMP))):"
    display(LUMP)
    @info "SPRAY ($(size(SPRAY, 1))×$(size(SPRAY, 2)), nnz=$(nnz(SPRAY))):"
    display(SPRAY)

    @info "Coarsened Jacobian"
    flush(stdout)
    Mc = LUMP * M * SPRAY
    @info "Coarsened Jacobian Mc ($(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc))):"
    display(Mc)

    jldsave(joinpath(matrices_dir, "LUMP.jld2"); LUMP)
    jldsave(joinpath(matrices_dir, "SPRAY.jld2"); SPRAY)
    jldsave(joinpath(matrices_dir, "Mc.jld2"); Mc)


    # TODO: Keep preconditioner code below for later when doing iterative solves.
    # @info "Setting up preconditioner"
    # stop_time = 12 * month  # reference time for preconditioner scaling
    # flush(stdout)
    # struct MyPreconditioner
    #     prob
    # end
    # Qc = stop_time * Mc
    # Plprob = LinearProblem(Qc, ones(size(Qc, 1)))
    # Plprob = init(Plprob, solver, rtol = 1.0e-12)
    # Pl = MyPreconditioner(Plprob)
    # Base.eltype(::MyPreconditioner) = Float64
    # function LinearAlgebra.ldiv!(Pl::MyPreconditioner, x::AbstractVector)
    #     Pl.prob.b = LUMP * x
    #     solve!(Pl.prob)
    #     x .= SPRAY * Pl.prob.u .- x
    #     return x
    # end
    # function LinearAlgebra.ldiv!(y::AbstractVector, Pl::MyPreconditioner, x::AbstractVector)
    #     Pl.prob.b = LUMP * x
    #     solve!(Pl.prob)
    #     y .= SPRAY * Pl.prob.u .- x
    #     return y
    # end
    # Pr = I
    # precs = Returns((Pl, Pr))

    if ISSTRUCTURALLYSYMMETRIC
        @info "Setting up Pardiso solver"
        flush(stdout)
        matrix_type = Pardiso.REAL_SYM
        @show solver = MKLPardisoIterate(; nprocs, matrix_type)
    else
        @info "Matrix is not structurally symmetric; using UMFPACKFactorization()"
        flush(stdout)
        @show solver = UMFPACKFactorization()
    end

    # ── Coarsened linear solve ──
    @info "Solving coarsened linear system (Mc \\ -1)"
    flush(stdout)
    init_prob_coarsened = LinearProblem(Mc, -ones(size(Mc, 1)))
    init_prob_coarsened = init(init_prob_coarsened, solver, rtol = 1.0e-12)
    @time "solve coarsened age" age_coarse_vec = SPRAY * solve!(init_prob_coarsened).u / year

    Nwet = size(wet3D)
    age_coarse_3D = zeros(Float64, Nwet)
    age_coarse_3D[idx] .= age_coarse_vec

    vol_mean_coarse = sum(age_coarse_vec .* v1D) / sum(v1D)
    @info "Volume-weighted mean coarsened steady age: $(vol_mean_coarse) years"

    fig, ax, plt = hist(age_coarse_vec)
    save(joinpath(matrix_plots_dir, "steady_age_coarsened_histogram.png"), fig)

    jldsave(joinpath(matrices_dir, "steady_age_coarsened.jld2"); age = age_coarse_3D, wet3D, idx)
    @info "saved coarsened steady age to $(joinpath(matrices_dir, "steady_age_coarsened.jld2"))"

    # ── Full linear solve ──
    @info "Solving full linear system (M \\ -1)"
    flush(stdout)
    init_prob_full = LinearProblem(M, -ones(size(M, 1)))
    init_prob_full = init(init_prob_full, solver, rtol = 1.0e-12)
    @time "solve full age" age_full_vec = solve!(init_prob_full).u / year

    age_full_3D = zeros(Float64, Nwet)
    age_full_3D[idx] .= age_full_vec

    vol_mean_full = sum(age_full_vec .* v1D) / sum(v1D)
    @info "Volume-weighted mean full steady age: $(vol_mean_full) years"

    fig, ax, plt = hist(age_full_vec)
    save(joinpath(matrix_plots_dir, "steady_age_full_histogram.png"), fig)

    jldsave(joinpath(matrices_dir, "steady_age_full.jld2"); age = age_full_3D, wet3D, idx)
    @info "saved full steady age to $(joinpath(matrices_dir, "steady_age_full.jld2"))"

    # ── Age diagnostic plots (zonal averages + horizontal slices) ──
    @info "Plotting age diagnostic figures"
    flush(stdout)
    vol_3D = zeros(Float64, Nwet)
    vol_3D[idx] .= v1D
    const OCEANS = oceanpolygons()
    plot_age_diagnostics(age_coarse_3D, grid, wet3D, vol_3D, matrix_plots_dir, "steady_age_coarsened")
    plot_age_diagnostics(age_full_3D, grid, wet3D, vol_3D, matrix_plots_dir, "steady_age_full")
else
    @info "Skipping age solve (ENABLE_AGE_SOLVE = false)"
    flush(stdout)
end

@info "create_matrix.jl complete. Outputs in $(matrices_dir)"
flush(stdout)
