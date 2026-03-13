"""
Solve for the steady-state age using the pre-built transport matrix M on GPU.

Loads M from `outputs/{parentmodel}/TM/{model_config}/M.jld2` (produced by
`create_matrix.jl`) and solves the linear system M x = -1 for the steady-state age
using CUDSS LU factorization on GPU.

This script requires a GPU node (gpuvolta queue).

Usage — interactive:
```
qsub -I -P y99 -l mem=96GB -q gpuvolta -l walltime=01:00:00 -l ngpus=1 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
module load cuda/12.9.0
export JULIA_CUDA_USE_COMPAT=false
julia --project
include("src/solve_matrix_age_gpu.jl")
```

Environment variables:
  PARENT_MODEL      – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE   – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION     – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME  – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER       – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
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
using CUDA
using CUDA.CUSPARSE
using CUDSS
using TOML

@info "Packages loaded"
flush(stdout); flush(stderr)

################################################################################
# Configuration
################################################################################

include("shared_functions.jl")

(; parentmodel, outputdir) = load_project_config()

year = years = 365.25 * 86400  # seconds

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

LINEAR_SOLVER = "CUDSS"

MATRIX_PROCESSING = get(ENV, "MATRIX_PROCESSING", "raw")
(MATRIX_PROCESSING ∈ ("raw", "symfill", "dropzeros", "symdrop")) || error("MATRIX_PROCESSING must be one of: raw, symfill, dropzeros, symdrop (got: $MATRIX_PROCESSING)")

LUMP_AND_SPRAY = lowercase(get(ENV, "LUMP_AND_SPRAY", "no")) == "yes"
coarse_tag = LUMP_AND_SPRAY ? "coarse" : "full"

TM_SOURCE = get(ENV, "TM_SOURCE", "const")
(TM_SOURCE ∈ ("const", "avg")) || error("TM_SOURCE must be one of: const, avg (got: $TM_SOURCE)")

output_tag = "steady_age_$(coarse_tag)_$(LINEAR_SOLVER)_$(MATRIX_PROCESSING)"

matrices_dir = joinpath(outputdir, "TM", model_config)
M_dir = joinpath(matrices_dir, TM_SOURCE)
matrix_plots_dir = joinpath(M_dir, "plots")
mkpath(matrix_plots_dir)

@info "Run configuration"
@info "- PARENT_MODEL      = $parentmodel"
@info "- VELOCITY_SOURCE   = $VELOCITY_SOURCE"
@info "- W_FORMULATION     = $W_FORMULATION"
@info "- ADVECTION_SCHEME  = $ADVECTION_SCHEME"
@info "- TIMESTEPPER       = $TIMESTEPPER"
@info "- LINEAR_SOLVER     = $LINEAR_SOLVER (GPU LU via CUDSS)"
@info "- MATRIX_PROCESSING = $MATRIX_PROCESSING"
@info "- LUMP_AND_SPRAY    = $LUMP_AND_SPRAY (tag: $coarse_tag)"
@info "- TM_SOURCE         = $TM_SOURCE"
@info "- output_tag        = $output_tag"
@info "- model_config      = $model_config"
@info "- M_dir             = $M_dir"
flush(stdout); flush(stderr)

# Print GPU info
@info "CUDA device: $(CUDA.device())"
@info "CUDA memory: $(CUDA.available_memory() / 1.0e9) GB available"
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

M_file = joinpath(M_dir, "M.jld2")
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

M = process_sparse_matrix(M, MATRIX_PROCESSING)

################################################################################
# LUMP/SPRAY coarsening (if requested)
################################################################################

(; Mc, SPRAY) = compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; LUMP_AND_SPRAY)

################################################################################
# GPU LU solve via CUDSS
################################################################################

Nwet = size(wet3D)

solve_matrix = LUMP_AND_SPRAY ? Mc : M
n = size(solve_matrix, 1)
@info "Transferring matrix to GPU (n=$n, nnz=$(nnz(solve_matrix)))"
flush(stdout); flush(stderr)

M_gpu = CuSparseMatrixCSR(solve_matrix)
b_gpu = CuVector(-ones(n))

@info "1st solve (factorize + solve) on GPU via CUDSS"
flush(stdout); flush(stderr)
print("1st solve: "); flush(stdout)
CUDA.@time begin
    F = lu(M_gpu)
    x_gpu = F \ b_gpu
end

# 2nd solve (reuses factorization)
@info "2nd solve with random RHS on GPU"
flush(stdout); flush(stderr)
b2_gpu = CUDA.randn(n)
print("2nd solve: "); flush(stdout)
CUDA.@time x2_gpu = F \ b2_gpu

# Transfer result back to CPU
x_cpu = Array(x_gpu)

if LUMP_AND_SPRAY
    @info "Projecting coarsened solution back to full grid (SPRAY * x)"
    flush(stdout); flush(stderr)
    age_vec = SPRAY * x_cpu / year
else
    age_vec = x_cpu / year
end

age_3D = zeros(Float64, Nwet)
age_3D[idx] .= age_vec

vol_mean = sum(age_vec .* v1D) / sum(v1D)
tag_label = LUMP_AND_SPRAY ? "coarsened" : "full"
@info "Volume-weighted mean $tag_label steady age: $(vol_mean) years"

fig, ax, plt = hist(age_vec)
save(joinpath(matrix_plots_dir, "$(output_tag)_histogram.png"), fig)

output_file = joinpath(M_dir, "$(output_tag).jld2")
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

@info "solve_matrix_age_gpu.jl complete. Outputs in $(M_dir)"
flush(stdout); flush(stderr)
