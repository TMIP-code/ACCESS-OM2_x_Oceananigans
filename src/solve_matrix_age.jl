"""
Solve for the steady-state age using the pre-built transport matrix M.

Loads M from `outputs/{parentmodel}/TM/{model_config}/M.jld2` (produced by
`create_matrix.jl`) and solves the linear system M x = -1 for the steady-state age.
When LUMP_AND_SPRAY=yes, solves the coarsened system instead.

This script runs on CPU only (no GPU required).

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q normal -l walltime=02:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/solve_matrix_age.jl")
```

Environment variables:
  PARENT_MODEL      – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE   – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION     – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME  – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER       – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  LINEAR_SOLVER     – Pardiso | ParU | UMFPACK  (default: Pardiso)
  LUMP_AND_SPRAY    – yes | no  (default: no)
  MATRIX_PROCESSING – raw | symfill | dropzeros | symdrop  (default: raw)
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Grids: on_architecture, znodes
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.AbstractOperations: volume
using KernelAbstractions: @kernel, @index
using LinearAlgebra
using SparseArrays
using Statistics
using JLD2
using Printf
using CairoMakie
using OceanTransportMatrixBuilder
using OceanBasins: oceanpolygons, isatlantic, ispacific, isindian
import Pardiso
import ParU_jll
using LinearSolve
using TOML

const nprocs = 12

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

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
else
    outputdir = profile["outputdir"]
end

year = years = 365.25 * 86400  # seconds

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
(LINEAR_SOLVER ∈ ("Pardiso", "ParU", "UMFPACK")) || error("LINEAR_SOLVER must be one of: Pardiso, ParU, UMFPACK (got: $LINEAR_SOLVER)")

MATRIX_PROCESSING = get(ENV, "MATRIX_PROCESSING", "raw")
(MATRIX_PROCESSING ∈ ("raw", "symfill", "dropzeros", "symdrop")) || error("MATRIX_PROCESSING must be one of: raw, symfill, dropzeros, symdrop (got: $MATRIX_PROCESSING)")

LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
coarse_tag = LUMP_AND_SPRAY ? "coarse" : "full"
output_tag = "steady_age_$(coarse_tag)_$(LINEAR_SOLVER)_$(MATRIX_PROCESSING)"

matrices_dir = joinpath(outputdir, "TM", model_config)
matrix_plots_dir = joinpath(matrices_dir, "plots")
mkpath(matrix_plots_dir)

@info "Run configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- VELOCITY_SOURCE   = $VELOCITY_SOURCE"
@info "- W_FORMULATION     = $W_FORMULATION"
@info "- ADVECTION_SCHEME  = $ADVECTION_SCHEME"
@info "- TIMESTEPPER       = $TIMESTEPPER"
@info "- LINEAR_SOLVER     = $LINEAR_SOLVER"
@info "- MATRIX_PROCESSING = $MATRIX_PROCESSING"
@info "- LUMP_AND_SPRAY    = $LUMP_AND_SPRAY (tag: $coarse_tag)"
@info "- output_tag        = $output_tag"
@info "- model_config      = $model_config"
@info "- matrices_dir      = $matrices_dir"
flush(stdout); flush(stderr)

################################################################################
# Load grid
################################################################################

@info "Loading grid"
flush(stdout); flush(stderr)
preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
@info "Grid loaded: $(size(grid))"
flush(stdout); flush(stderr)

################################################################################
# Load transport matrix M
################################################################################

M_file = joinpath(matrices_dir, "M.jld2")
@info "Loading transport matrix from $M_file"
flush(stdout); flush(stderr)
M = load(M_file, "M")
@info "Loaded M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"
flush(stdout); flush(stderr)

################################################################################
# Wet mask and cell volumes
################################################################################

@info "Computing wet cell mask and cell volumes"
flush(stdout); flush(stderr)
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
@info "Number of wet cells: $Nidx"
@assert Nidx == size(M, 1) "Mismatch: wet cells ($Nidx) != matrix rows ($(size(M, 1)))"

v1D = interior(compute_volume(grid))[idx]
flush(stdout); flush(stderr)

################################################################################
# Matrix processing
################################################################################

@info "Applying matrix processing: $MATRIX_PROCESSING"
flush(stdout); flush(stderr)

if MATRIX_PROCESSING == "symfill"
    # Fill sparsity structure to be structurally symmetric by adding zero-valued
    # entries at (j, i) for every existing (i, j) entry
    I_idx, J_idx, V = findnz(M)
    M = sparse([I_idx; J_idx], [J_idx; I_idx], [V; zeros(length(V))], size(M)...)
    @info "After symfill: nnz=$(nnz(M))"
elseif MATRIX_PROCESSING == "dropzeros"
    nnz_before = nnz(M)
    dropzeros!(M)
    @info "After dropzeros: nnz $nnz_before → $(nnz(M))"
elseif MATRIX_PROCESSING == "symdrop"
    # Drop non-symmetric structural entries: keep (i,j) only if (j,i) also exists
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

################################################################################
# LUMP/SPRAY coarsening (if requested)
################################################################################

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
else
    @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no)"
end
flush(stdout); flush(stderr)

################################################################################
# Linear solver setup
################################################################################

# Set up linear solver
if LINEAR_SOLVER == "Pardiso"
    # Determine matrix type based on structural symmetry of the solve matrix
    solve_matrix = LUMP_AND_SPRAY ? Mc : M
    ISSTRUCTURALLYSYMMETRIC = Pardiso.isstructurallysymmetric(solve_matrix)
    @info "Solve matrix is structurally symmetric: $ISSTRUCTURALLYSYMMETRIC"
    matrix_type = ISSTRUCTURALLYSYMMETRIC ? Pardiso.REAL_SYM : Pardiso.REAL_NONSYM
    @info "Using Pardiso $(ISSTRUCTURALLYSYMMETRIC ? "REAL_SYM (mtype=1)" : "REAL_NONSYM (mtype=11)")"
    @show solver = MKLPardisoIterate(; nprocs, matrix_type)
elseif LINEAR_SOLVER == "ParU"
    @info "Using ParUFactorization (parallel sparse LU)"
    @show solver = ParUFactorization(; reuse_symbolic = true)
elseif LINEAR_SOLVER == "UMFPACK"
    @info "Using UMFPACKFactorization (serial sparse LU)"
    @show solver = UMFPACKFactorization(; reuse_symbolic = true)
end
flush(stdout); flush(stderr)

################################################################################
# Solve for steady-state age
################################################################################

Nwet = size(wet3D)

if LUMP_AND_SPRAY
    # ── Coarsened linear solve ──
    n_solve = size(Mc, 1)
    @info "Solving coarsened linear system (Mc \\ -1)"
    flush(stdout); flush(stderr)
    cache = init(LinearProblem(Mc, -ones(n_solve)), solver, rtol = 1.0e-12)
    @time "1st solve" age_vec = SPRAY * solve!(cache).u / year

    # 2nd solve (reuses cached factorization)
    @info "2nd solve with random RHS"
    flush(stdout); flush(stderr)
    cache.b = randn(n_solve)
    @time "2nd solve" solve!(cache)

    age_3D = zeros(Float64, Nwet)
    age_3D[idx] .= age_vec

    vol_mean = sum(age_vec .* v1D) / sum(v1D)
    @info "Volume-weighted mean coarsened steady age: $(vol_mean) years"
else
    # ── Full linear solve ──
    n_solve = size(M, 1)
    @info "Solving full linear system (M \\ -1)"
    flush(stdout); flush(stderr)
    cache = init(LinearProblem(M, -ones(n_solve)), solver, rtol = 1.0e-12)
    @time "1st solve" age_vec = solve!(cache).u / year

    # 2nd solve (reuses cached factorization)
    @info "2nd solve with random RHS"
    flush(stdout); flush(stderr)
    cache.b = randn(n_solve)
    @time "2nd solve" solve!(cache)

    age_3D = zeros(Float64, Nwet)
    age_3D[idx] .= age_vec

    vol_mean = sum(age_vec .* v1D) / sum(v1D)
    @info "Volume-weighted mean full steady age: $(vol_mean) years"
end

fig, ax, plt = hist(age_vec)
save(joinpath(matrix_plots_dir, "$(output_tag)_histogram.png"), fig)

output_file = joinpath(matrices_dir, "$(output_tag).jld2")
jldsave(output_file; age = age_3D, wet3D, idx)
@info "Saved steady age to $output_file"

################################################################################
# Age diagnostic plots
################################################################################

@info "Plotting age diagnostic figures"
flush(stdout); flush(stderr)

vol_3D = zeros(Float64, Nwet)
vol_3D[idx] .= v1D
const OCEANS = oceanpolygons()

plot_age_diagnostics(
    age_3D, grid, wet3D, vol_3D, matrix_plots_dir, output_tag
)

@info "solve_matrix_age.jl complete. Outputs in $(matrices_dir)"
flush(stdout); flush(stderr)
