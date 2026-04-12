"""
Build transport matrices from velocity snapshots saved by a 1-year simulation.

Loads u, v, w, η snapshots from the split JLD2Writer part files produced by
`run_1year.jl` (one file per monthly output) and computes a Jacobian at each
snapshot time.  The saved w field already includes the ∂η/∂t contribution
(diagnosed during the simulation), so this script always uses `wprescribed`
regardless of the W_FORMULATION env var.

The sparsity pattern and graph colouring are computed once and reused for all
snapshots.  Each Jacobian takes ~15 s on 48 CPUs.

Usage — interactive:
```
qsub -I -P y99 -l mem=192GB -q normal -l walltime=04:00:00 -l ncpus=48 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 \\
     -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/create_snapshot_matrices.jl")
```

Environment variables: same as create_matrix.jl (PARENT_MODEL, VELOCITY_SOURCE, etc.)
"""

@info "Loading packages and functions"
flush(stdout); flush(stderr)

using Oceananigans

# Matrix build always runs on the CPU
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
using Oceananigans.OutputReaders: InMemory
using Oceananigans.Units: minute, minutes, hour, hours, day, days, second, seconds
year = years = 365.25days
month = months = year / 12

using Adapt: adapt
using Statistics
using LinearAlgebra
using SparseArrays
import Pardiso
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
@info "- W_FORMULATION    = $W_FORMULATION (NOTE: snapshots always use wprescribed)"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- GM_REDI          = $GM_REDI (GM_REDI_STR=$GM_REDI_STR)"
@info "- MONTHLY_KAPPAV   = $MONTHLY_KAPPAV"
@info "- model_config     = $model_config"
flush(stdout); flush(stderr)

matrices_dir = joinpath(outputdir, "TM", model_config)
snapshot_matrices_dir = joinpath(matrices_dir, "snapshots")
mkpath(snapshot_matrices_dir)
@show outputdir
@show matrices_dir
@show snapshot_matrices_dir
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
# Discover snapshot part files from run_1year output
################################################################################

age_output_dir = joinpath(outputdir, "standardrun", model_config)

# With file_splitting = TimeInterval(prescribed_Δt), run_1year produces:
#   age_1year_part1.jld2  (t=0, skip)
#   age_1year_part2.jld2  (t=1mo)
#   ...
#   age_1year_part13.jld2 (t=12mo)
# Parts 2–13 are the usable monthly snapshots.
n_parts = 13
part_files = [joinpath(age_output_dir, "age_1year_part$(i).jld2") for i in 2:n_parts]
n_usable = length(part_files)

@info "Snapshot part files (skipping part1 = t=0):"
for (i, f) in enumerate(part_files)
    isfile(f) || error("Part file not found: $f")
    @info "  snapshot $(@sprintf("%02d", i)): $f"
end
@info "$n_usable usable snapshot part files"
flush(stdout); flush(stderr)

################################################################################
# Create constant velocity fields (mutated in-place for each snapshot)
################################################################################

@info "Creating constant velocity field containers"
flush(stdout); flush(stderr)

ubcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = FPivotZipperBoundaryCondition(-1))
vbcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = FPivotZipperBoundaryCondition(-1))
wbcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); north = FPivotZipperBoundaryCondition(1))


u_constant = XFaceField(grid; boundary_conditions = ubcs)
v_constant = YFaceField(grid; boundary_conditions = vbcs)
w_constant = ZFaceField(grid; boundary_conditions = wbcs)
η_constant = Field{Center, Center, Nothing}(grid)

# Initialise with first usable snapshot (part 2) for sparsity detection
@info "Loading first snapshot for sparsity detection: $(part_files[1])"
flush(stdout); flush(stderr)
u_snap = FieldTimeSeries(part_files[1], "u"; architecture = arch, grid, backend = InMemory())
v_snap = FieldTimeSeries(part_files[1], "v"; architecture = arch, grid, backend = InMemory())
w_snap = FieldTimeSeries(part_files[1], "w"; architecture = arch, grid, backend = InMemory())
η_snap = FieldTimeSeries(part_files[1], "η"; architecture = arch, grid, backend = InMemory())
set!(u_constant, interior(u_snap[end]))
fill_halo_regions!(u_constant)
set!(v_constant, interior(v_snap[end]))
fill_halo_regions!(v_constant)
set!(w_constant, interior(w_snap[end]))
fill_halo_regions!(w_constant)
set!(η_constant, interior(η_snap[end]))
fill_halo_regions!(η_constant)
u_snap = v_snap = w_snap = η_snap = nothing  # free memory

@info "Constant velocity containers initialised"
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
end

################################################################################
# Prescribed velocities (always wprescribed) and free surface
################################################################################

@info "Using wprescribed (w from simulation snapshots includes ∂η/∂t)"
flush(stdout); flush(stderr)

velocities = PrescribedVelocityFields(u = u_constant, v = v_constant, w = w_constant)
free_surface = PrescribedFreeSurface(displacement = η_constant)

################################################################################
# Closures (explicit only — required for Jacobian via ForwardDiff)
################################################################################

@info "Creating closures"
flush(stdout); flush(stderr)

κVML = 0.1    # m^2/s in the mixed layer
κVBG = 3.0e-5 # m^2/s in the ocean interior (background)

# TODO: replace with monthly MLD (time-dependent κ) once implemented
mld_ds = open_dataset(joinpath(yearly_dir, "mld_yearly.nc"))
mld_data = on_architecture(arch, -replace(readcubedata(mld_ds.mld).data, NaN => 0.0))
z_center = znodes(grid, Center(), Center(), Center())
is_mld = reshape(z_center, 1, 1, Nz) .> mld_data
κVField = CenterField(grid)
set!(κVField, κVML * is_mld + κVBG * .!is_mld)

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
# Initialise model state (update zstar from first snapshot η)
################################################################################

@info "Initialising model state (zstar scaling from first snapshot η)"
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
# Autodiff setup (sparsity detection + coloring — ONCE for all snapshots)
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

    GADcvec .= view(interior(GADc_field), idx)
    return GADcvec
end

ADc_buf = CenterField(grid)
GADc_buf = CenterField(grid)

@info "Benchmarking tendency function"
flush(stdout); flush(stderr)
ADcvec = ones(Nidx)
GADcvec = similar(ADcvec)
mytendency!(GADcvec, ADcvec, ADc_buf, GADc_buf)
@time "Tendency evaluation" mytendency!(GADcvec, ADcvec, ADc_buf, GADc_buf)

@info "Detecting sparsity pattern..."
flush(stdout); flush(stderr)
@time "Detect sparsity" S = jacobian_sparsity_with_contexts(
    mytendency!, GADcvec, TracerSparsityDetector(; gradient_pattern_type = Set{UInt}), ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)

# No symmetrization — coloring matches the raw Jacobian pattern.
# Structural symmetry for Pardiso is enforced downstream on Q_precond.
@info "Sparsity: nnz(S) = $(nnz(S))"
flush(stdout); flush(stderr)

sparse_forward_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = KnownJacobianSparsityDetector(S),
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

# Free sparsity intermediates no longer needed (pattern is captured in jac_prep)
S = nothing
S_sym = nothing
GC.gc()
@info "Freed sparsity detection intermediates"
flush(stdout); flush(stderr)

################################################################################
# Allocate accumulator for inline averaging
################################################################################

nnz_count = nnz(jac_buffer)
nzval_avg = zeros(Float64, nnz_count)

################################################################################
# Compute Jacobian at each snapshot (load one part file at a time)
################################################################################

@info "="^72
@info "Computing Jacobians at $n_usable snapshot times"
@info "="^72
flush(stdout); flush(stderr)

for i_snap in 1:n_usable
    snap_label = @sprintf("%02d", i_snap)
    file = part_files[i_snap]

    @info "Snapshot $snap_label/$n_usable: loading $file"
    flush(stdout); flush(stderr)

    # Load velocity fields from this part file
    u_snap = FieldTimeSeries(file, "u"; architecture = arch, grid, backend = InMemory())
    v_snap = FieldTimeSeries(file, "v"; architecture = arch, grid, backend = InMemory())
    w_snap = FieldTimeSeries(file, "w"; architecture = arch, grid, backend = InMemory())
    η_snap = FieldTimeSeries(file, "η"; architecture = arch, grid, backend = InMemory())
    t_snap = u_snap.times[end]

    @info "  t = $(@sprintf("%.4f", t_snap / year)) yr"

    # Update velocity fields from snapshot
    set!(u_constant, interior(u_snap[end]))
    fill_halo_regions!(u_constant)
    set!(v_constant, interior(v_snap[end]))
    fill_halo_regions!(v_constant)
    set!(w_constant, interior(w_snap[end]))
    fill_halo_regions!(w_constant)
    set!(η_constant, interior(η_snap[end]))
    fill_halo_regions!(η_constant)
    u_snap = v_snap = w_snap = η_snap = nothing  # free memory

    # Update zstar scaling from this snapshot's η
    launch!(CPU(), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η_constant, grid)

    # Compute Jacobian
    @time "Jacobian at snapshot $snap_label" jacobian!(
        mytendency!, GADcvec, jac_buffer, jac_prep,
        sparse_forward_backend, ADcvec,
        Cache(ADc_buf), Cache(GADc_buf),
    )

    @info "  M(snapshot $snap_label): nnz=$(nnz(jac_buffer)), max|M|=$(maximum(abs, nonzeros(jac_buffer)))"

    # Accumulate into average (all share the same sparsity pattern)
    nzval_avg .+= jac_buffer.nzval

    # Save snapshot matrix directly (no copy — JLD2 serializes immediately)
    outfile = joinpath(snapshot_matrices_dir, "M_snapshot_$(snap_label).jld2")
    jldsave(outfile; M = jac_buffer, t = t_snap)
    @info "  Saved $outfile"
    flush(stdout); flush(stderr)
end

################################################################################
# Finalise and save averaged matrix
################################################################################

@info "="^72
@info "Finalising averaged matrix"
@info "="^72
flush(stdout); flush(stderr)

nzval_avg ./= n_usable

M_avg = SparseMatrixCSC(
    size(jac_buffer, 1), size(jac_buffer, 2),
    jac_buffer.colptr, jac_buffer.rowval, copy(nzval_avg),
)

@info "M_avg: $(size(M_avg, 1))×$(size(M_avg, 2)), nnz=$(nnz(M_avg))"

avg_dir = joinpath(matrices_dir, "avg")
mkpath(avg_dir)
outfile = joinpath(avg_dir, "M.jld2")
isfile(outfile) && rm(outfile)
jldsave(outfile; M = M_avg)
@info "Saved $outfile (nnz=$(nnz(M_avg)))"
flush(stdout); flush(stderr)

@info "="^72
@info "create_snapshot_matrices.jl complete"
@info "Saved $n_usable snapshot matrices to $snapshot_matrices_dir"
@info "Saved averaged matrix to $avg_dir"
flush(stdout); flush(stderr)
