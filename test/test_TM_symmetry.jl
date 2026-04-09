"""
Test structural symmetry of transport matrix M and coarsened matrix Mc = LUMP * M * SPRAY.

Also checks that sparsity(LUMP) == sparsity(SPRAY').

Usage:
    PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1958-1987 ... julia --project test/test_TM_symmetry.jl
"""

using Oceananigans
using JLD2
using SparseArrays
using LinearAlgebra
import Pardiso

include("../src/shared_functions.jl")

(; parentmodel, experiment_dir, outputdir) = load_project_config()
(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)

const_dir = joinpath(outputdir, "TM", model_config, "const")
M_file = joinpath(const_dir, "M.jld2")
@info "Loading M from $M_file"
M = load(M_file, "M")
@info "M: $(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M))"

# Test 1: M structural symmetry
@info "Test 1: M structural symmetry"
M_sym = Pardiso.isstructurallysymmetric(M)
@info "  Pardiso.isstructurallysymmetric(M) = $M_sym"
if !M_sym
    # Count asymmetric entries (both directions)
    Mt = sparse(M')
    local n_only_M = 0
    local n_only_Mt = 0
    for j in 1:size(M, 2), i in nzrange(M, j)
        r = M.rowval[i]
        if !(r in view(Mt.rowval, nzrange(Mt, j)))
            n_only_M += 1
        end
    end
    for j in 1:size(Mt, 2), i in nzrange(Mt, j)
        r = Mt.rowval[i]
        if !(r in view(M.rowval, nzrange(M, j)))
            n_only_Mt += 1
        end
    end
    @warn "  Asymmetric entries: $n_only_M in M but not M', $n_only_Mt in M' but not M"
    @info "  nnz(M)=$(nnz(M)), nnz(M')=$(nnz(Mt))"
    # Check diagonal coverage
    local n_missing_diag = 0
    for j in 1:size(M, 2)
        if !(j in view(M.rowval, nzrange(M, j)))
            n_missing_diag += 1
        end
    end
    @info "  Missing diagonal entries: $n_missing_diag / $(size(M, 1))"
end

# Test 2: LUMP/SPRAY via OceanTransportMatrixBuilder
@info "Test 2: LUMP and SPRAY"
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
(; wet3D, idx, Nidx) = compute_wet_mask(grid)
v1D = ones(Nidx)

using OceanTransportMatrixBuilder
LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di = 2, dj = 2, dk = 1)
@info "  LUMP: $(size(LUMP)), nnz=$(nnz(LUMP))"
@info "  SPRAY: $(size(SPRAY)), nnz=$(nnz(SPRAY))"

# Check sparsity(LUMP) == sparsity(SPRAY')
SPRAY_T = sparse(SPRAY')
lump_match = (size(LUMP) == size(SPRAY_T)) &&
    (LUMP.colptr == SPRAY_T.colptr) &&
    (LUMP.rowval == SPRAY_T.rowval)
@info "  sparsity(LUMP) == sparsity(SPRAY'): $lump_match"
if !lump_match
    @info "  nnz(LUMP)=$(nnz(LUMP)), nnz(SPRAY')=$(nnz(SPRAY_T))"
end

# Test 3: Mc structural symmetry
@info "Test 3: Mc = LUMP * M * SPRAY structural symmetry"
Mc = LUMP * M * SPRAY
@info "  Mc: $(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc))"
Mc_sym = Pardiso.isstructurallysymmetric(Mc)
@info "  Pardiso.isstructurallysymmetric(Mc) = $Mc_sym"

# Summary
@info "="^60
@info "Summary"
@info "  M structurally symmetric:    $M_sym"
@info "  LUMP == SPRAY' pattern:      $lump_match"
@info "  Mc structurally symmetric:   $Mc_sym"
all_pass = M_sym && lump_match && Mc_sym
@info "  All checks passed:           $all_pass"
all_pass || @warn "Some symmetry checks failed — investigate before using Pardiso"
