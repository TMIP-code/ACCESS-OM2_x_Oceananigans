"""
Build transport matrices from velocity snapshots saved by a 1-year simulation.

Loads u, v, w, η snapshots from the JLD2Writer output of `run_1year.jl` and
computes a Jacobian at each snapshot time.  The saved w field already includes
the ∂η/∂t contribution (diagnosed during the simulation), so this script always
uses `wprescribed` regardless of the W_FORMULATION env var.

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
  SNAPSHOT_FILE – path to run_1year JLD2 output (default: auto-detected)
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

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

@info "Run configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION (NOTE: snapshots always use wprescribed)"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- model_config     = $model_config"
flush(stdout); flush(stderr)

preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
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
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, arch)

Nx, Ny, Nz = size(grid)
@info "Grid loaded: Nx=$Nx, Ny=$Ny, Nz=$Nz"
flush(stdout); flush(stderr)

################################################################################
# Load velocity snapshots from run_1year output
################################################################################

snapshot_file = get(ENV, "SNAPSHOT_FILE", "")
if isempty(snapshot_file)
    age_output_dir = joinpath(outputdir, "standardrun", model_config)
    snapshot_file = joinpath(age_output_dir, "age_1year.jld2")
end
@info "Loading velocity snapshots from: $snapshot_file"
isfile(snapshot_file) || error("Snapshot file not found: $snapshot_file")
flush(stdout); flush(stderr)

u_fts = FieldTimeSeries(snapshot_file, "u"; architecture = arch, grid, backend = InMemory(2))
v_fts = FieldTimeSeries(snapshot_file, "v"; architecture = arch, grid, backend = InMemory(2))
w_fts = FieldTimeSeries(snapshot_file, "w"; architecture = arch, grid, backend = InMemory(2))
η_fts = FieldTimeSeries(snapshot_file, "η"; architecture = arch, grid, backend = InMemory(2))

snapshot_times = u_fts.times
n_snapshots = length(snapshot_times)

# Handle t=0 offset: JLD2Writer may include an initial snapshot at t≈0
t0_offset = snapshot_times[1] ≈ 0.0 ? 1 : 0
if t0_offset > 0
    @info "Detected t=0 snapshot — skipping it ($(n_snapshots - t0_offset) usable snapshots)"
else
    @info "No t=0 snapshot detected ($n_snapshots usable snapshots)"
end
n_usable = n_snapshots - t0_offset
flush(stdout); flush(stderr)

@info "Snapshot times (years):"
for i in (1 + t0_offset):n_snapshots
    @info "  snapshot $(@sprintf("%02d", i - t0_offset)): t = $(@sprintf("%.4f", snapshot_times[i] / year)) yr"
end
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

# Initialise with first usable snapshot for sparsity detection
first_snap = 1 + t0_offset
set!(u_constant, interior(u_fts[first_snap]))
fill_halo_regions!(u_constant)
set!(v_constant, interior(v_fts[first_snap]))
fill_halo_regions!(v_constant)
set!(w_constant, interior(w_fts[first_snap]))
fill_halo_regions!(w_constant)
set!(η_constant, interior(η_fts[first_snap]))
fill_halo_regions!(η_constant)

@info "Constant velocity containers initialised"
flush(stdout); flush(stderr)

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

resolution_str = split(parentmodel, "-")[end]
experiment = "$(resolution_str)deg_jra55_iaf_omip2_cycle6"
time_window = "Jan1960-Dec1979"
@show inputdir = "/scratch/y99/TMIP/data/$parentmodel/$experiment/$time_window"

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
# Initialise model state (update zstar from first snapshot η)
################################################################################

@info "Initialising model state (zstar scaling from first snapshot η)"
flush(stdout); flush(stderr)

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

S_sym = S .| S'
@info "Sparsity: nnz(S) = $(nnz(S)), nnz(S_sym) = $(nnz(S_sym))"
flush(stdout); flush(stderr)

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

# Free sparsity intermediates no longer needed (pattern is captured in jac_prep)
S = nothing
S_sym = nothing
GC.gc()
@info "Freed sparsity detection intermediates"
flush(stdout); flush(stderr)

################################################################################
# Classify snapshots for inline averaging (start-of-month vs mid-month)
################################################################################

@info "Classifying snapshots for inline averaging"
flush(stdout); flush(stderr)

is_start_of_month = falses(n_usable)
for i_snap in 1:n_usable
    i_fts = i_snap + t0_offset
    t_months = snapshot_times[i_fts] / month
    dist_to_int = abs(t_months - round(t_months))
    dist_to_half = abs(t_months - (floor(t_months) + 0.5))
    is_start_of_month[i_snap] = dist_to_int < dist_to_half
end
n_start = count(is_start_of_month)
n_mid = n_usable - n_start
@info "  Start-of-month: $n_start, mid-month: $n_mid"
flush(stdout); flush(stderr)

# Allocate nzval accumulators (reuse sparsity pattern from jac_buffer)
nnz_count = nnz(jac_buffer)
nzval_avg24 = zeros(Float64, nnz_count)
nzval_avg12a = zeros(Float64, nnz_count)   # start-of-month
nzval_avg12b = zeros(Float64, nnz_count)   # mid-month

################################################################################
# Compute Jacobian at each snapshot
################################################################################

@info "="^72
@info "Computing Jacobians at $n_usable snapshot times"
@info "="^72
flush(stdout); flush(stderr)

for i_snap in 1:n_usable
    i_fts = i_snap + t0_offset
    t_snap = snapshot_times[i_fts]
    snap_label = @sprintf("%02d", i_snap)

    @info "Snapshot $snap_label/$n_usable: t = $(@sprintf("%.4f", t_snap / year)) yr"
    flush(stdout); flush(stderr)

    # Update velocity fields from snapshot
    set!(u_constant, interior(u_fts[i_fts]))
    fill_halo_regions!(u_constant)
    set!(v_constant, interior(v_fts[i_fts]))
    fill_halo_regions!(v_constant)
    set!(w_constant, interior(w_fts[i_fts]))
    fill_halo_regions!(w_constant)
    set!(η_constant, interior(η_fts[i_fts]))
    fill_halo_regions!(η_constant)

    # Update zstar scaling from this snapshot's η
    launch!(CPU(), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η_constant, grid)

    # Compute Jacobian
    @time "Jacobian at snapshot $snap_label" jacobian!(
        mytendency!, GADcvec, jac_buffer, jac_prep,
        sparse_forward_backend, ADcvec,
        Cache(ADc_buf), Cache(GADc_buf),
    )

    @info "  M(snapshot $snap_label): nnz=$(nnz(jac_buffer)), max|M|=$(maximum(abs, nonzeros(jac_buffer)))"

    # Accumulate into averages (all share the same sparsity pattern)
    nzval_avg24 .+= jac_buffer.nzval
    if is_start_of_month[i_snap]
        nzval_avg12a .+= jac_buffer.nzval
    else
        nzval_avg12b .+= jac_buffer.nzval
    end

    # Save snapshot matrix directly (no copy — JLD2 serializes immediately)
    outfile = joinpath(snapshot_matrices_dir, "M_snapshot_$(snap_label).jld2")
    jldsave(outfile; M = jac_buffer, t = t_snap)
    @info "  Saved $outfile"
    flush(stdout); flush(stderr)
end

################################################################################
# Finalise and save averaged matrices
################################################################################

@info "="^72
@info "Finalising averaged matrices"
@info "="^72
flush(stdout); flush(stderr)

nzval_avg24 ./= n_usable
nzval_avg12a ./= n_start
nzval_avg12b ./= n_mid

for (label, nzval) in [("avg24", nzval_avg24), ("avg12a", nzval_avg12a), ("avg12b", nzval_avg12b)]
    M_avg = SparseMatrixCSC(
        size(jac_buffer, 1), size(jac_buffer, 2),
        jac_buffer.colptr, jac_buffer.rowval, copy(nzval),
    )

    if !Pardiso.isstructurallysymmetric(M_avg)
        error("[$label] Averaged matrix is NOT structurally symmetric (nnz=$(nnz(M_avg)))")
    end
    @info "  [$label] Structural symmetry check passed"

    out_dir = joinpath(matrices_dir, label)
    mkpath(out_dir)
    outfile = joinpath(out_dir, "M.jld2")
    isfile(outfile) && rm(outfile)
    jldsave(outfile; M = M_avg)
    @info "  Saved $outfile (nnz=$(nnz(M_avg)))"
    flush(stdout); flush(stderr)
end

@info "="^72
@info "create_snapshot_matrices.jl complete"
@info "Saved $n_usable snapshot matrices to $snapshot_matrices_dir"
@info "Saved averaged matrices (avg24, avg12a, avg12b) to $matrices_dir"
flush(stdout); flush(stderr)
