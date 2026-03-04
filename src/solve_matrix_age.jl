"""
Solve for the steady-state age using the pre-built transport matrix M.

Loads M from `outputs/{parentmodel}/matrices/{model_config}/M.jld2` (produced by
`create_matrix.jl`) and solves the linear system M x = -1 for the steady-state age.
Optionally uses lump-and-spray coarsening for a coarsened solve as well.

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
  PRECONDITIONER_MATRIX_TYPE – nonsym | sym_cleaned  (default: nonsym; Pardiso only)
"""

@info "Loading packages"
flush(stdout)

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
flush(stdout)

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

PRECONDITIONER_MATRIX_TYPE = get(ENV, "PRECONDITIONER_MATRIX_TYPE", "nonsym")
(PRECONDITIONER_MATRIX_TYPE ∈ ("nonsym", "sym_cleaned")) || error("PRECONDITIONER_MATRIX_TYPE must be one of: nonsym, sym_cleaned (got: $PRECONDITIONER_MATRIX_TYPE)")

LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
lumpspray_tag = LUMP_AND_SPRAY ? "LSprec" : "prec"

matrices_dir = joinpath(outputdir, "matrices", model_config)
matrix_plots_dir = joinpath(matrices_dir, "plots")
mkpath(matrix_plots_dir)

@info "Run configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- VELOCITY_SOURCE  = $VELOCITY_SOURCE"
@info "- W_FORMULATION    = $W_FORMULATION"
@info "- ADVECTION_SCHEME = $ADVECTION_SCHEME"
@info "- TIMESTEPPER      = $TIMESTEPPER"
@info "- LINEAR_SOLVER    = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY   = $LUMP_AND_SPRAY (tag: $lumpspray_tag)"
@info "- PRECONDITIONER_MATRIX_TYPE = $PRECONDITIONER_MATRIX_TYPE"
@info "- model_config     = $model_config"
@info "- matrices_dir     = $matrices_dir"
flush(stdout)

################################################################################
# Load grid
################################################################################

@info "Loading grid"
flush(stdout)
preprocessed_inputs_dir = normpath(joinpath(@__DIR__, "..", "preprocessed_inputs", parentmodel))
grid_file = joinpath(preprocessed_inputs_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
@info "Grid loaded: $(size(grid))"
flush(stdout)

################################################################################
# Load transport matrix M
################################################################################

M_file = joinpath(matrices_dir, "M.jld2")
@info "Loading transport matrix from $M_file"
flush(stdout)
M = load(M_file, "M")
@info "Loaded M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"
flush(stdout)

################################################################################
# Wet mask and cell volumes
################################################################################

@info "Computing wet cell mask and cell volumes"
flush(stdout)
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
@info "Number of wet cells: $Nidx"
@assert Nidx == size(M, 1) "Mismatch: wet cells ($Nidx) != matrix rows ($(size(M, 1)))"

v1D = interior(compute_volume(grid))[idx]
flush(stdout)

################################################################################
# LUMP/SPRAY and linear solver setup
################################################################################

if LUMP_AND_SPRAY
    @info "Computing LUMP and SPRAY matrices"
    flush(stdout)
    LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
    @info "LUMP ($(size(LUMP, 1))×$(size(LUMP, 2)), nnz=$(nnz(LUMP)))"
    @info "SPRAY ($(size(SPRAY, 1))×$(size(SPRAY, 2)), nnz=$(nnz(SPRAY)))"
    Mc = LUMP * M * SPRAY
    @info "Coarsened Jacobian Mc ($(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc)))"

    jldsave(joinpath(matrices_dir, "LUMP.jld2"); LUMP)
    jldsave(joinpath(matrices_dir, "SPRAY.jld2"); SPRAY)
    jldsave(joinpath(matrices_dir, "Mc.jld2"); Mc)
else
    @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no)"
    LUMP = I
    SPRAY = I
end
flush(stdout)

# Set up linear solver
if LINEAR_SOLVER == "Pardiso"
    # Check structural symmetry
    ISSTRUCTURALLYSYMMETRIC = Pardiso.isstructurallysymmetric(M)
    @info "M is structurally symmetric: $ISSTRUCTURALLYSYMMETRIC"

    if PRECONDITIONER_MATRIX_TYPE == "sym_cleaned" && ISSTRUCTURALLYSYMMETRIC
        matrix_type = Pardiso.REAL_SYM
        @info "Using Pardiso REAL_SYM (mtype=1)"
    else
        matrix_type = Pardiso.REAL_NONSYM
        @info "Using Pardiso REAL_NONSYM (mtype=11)"
    end
    @show solver = MKLPardisoIterate(; nprocs, matrix_type)
elseif LINEAR_SOLVER == "ParU"
    @info "Using ParUFactorization (parallel sparse LU)"
    @show solver = ParUFactorization(; reuse_symbolic = true)
elseif LINEAR_SOLVER == "UMFPACK"
    @info "Using UMFPACKFactorization (serial sparse LU)"
    @show solver = UMFPACKFactorization(; reuse_symbolic = true)
end
flush(stdout)

################################################################################
# Solve for steady-state age
################################################################################

Nwet = size(wet3D)

if LUMP_AND_SPRAY
    # ── Coarsened linear solve ──
    @info "Solving coarsened linear system (Mc \\ -1)"
    flush(stdout)
    init_prob_coarsened = LinearProblem(Mc, -ones(size(Mc, 1)))
    init_prob_coarsened = init(init_prob_coarsened, solver, rtol = 1.0e-12)
    @time "solve coarsened age" age_coarse_vec = SPRAY * solve!(init_prob_coarsened).u / year

    age_coarse_3D = zeros(Float64, Nwet)
    age_coarse_3D[idx] .= age_coarse_vec

    vol_mean_coarse = sum(age_coarse_vec .* v1D) / sum(v1D)
    @info "Volume-weighted mean coarsened steady age: $(vol_mean_coarse) years"

    fig, ax, plt = hist(age_coarse_vec)
    save(joinpath(matrix_plots_dir, "steady_age_coarsened_$(LINEAR_SOLVER)_$(lumpspray_tag)_histogram.png"), fig)

    coarsened_file = joinpath(matrices_dir, "steady_age_coarsened_$(LINEAR_SOLVER)_$(lumpspray_tag).jld2")
    jldsave(coarsened_file; age = age_coarse_3D, wet3D, idx)
    @info "Saved coarsened steady age to $coarsened_file"
end

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
save(joinpath(matrix_plots_dir, "steady_age_full_$(LINEAR_SOLVER)_$(lumpspray_tag)_histogram.png"), fig)

full_file = joinpath(matrices_dir, "steady_age_full_$(LINEAR_SOLVER)_$(lumpspray_tag).jld2")
jldsave(full_file; age = age_full_3D, wet3D, idx)
@info "Saved full steady age to $full_file"

################################################################################
# Age diagnostic plots
################################################################################

@info "Plotting age diagnostic figures"
flush(stdout)

vol_3D = zeros(Float64, Nwet)
vol_3D[idx] .= v1D
const OCEANS = oceanpolygons()

plot_age_diagnostics(
    age_full_3D, grid, wet3D, vol_3D, matrix_plots_dir,
    "steady_age_full_$(LINEAR_SOLVER)_$(lumpspray_tag)"
)

if LUMP_AND_SPRAY
    plot_age_diagnostics(
        age_coarse_3D, grid, wet3D, vol_3D, matrix_plots_dir,
        "steady_age_coarsened_$(LINEAR_SOLVER)_$(lumpspray_tag)"
    )
end

@info "solve_matrix_age.jl complete. Outputs in $(matrices_dir)"
flush(stdout)
