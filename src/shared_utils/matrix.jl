################################################################################
# Matrix helpers
#
# Extracted from shared_functions.jl — hydrostatic free surface tendency kernel
# and sparse matrix processing/coarsening utilities.
################################################################################

using Oceananigans.Models.HydrostaticFreeSurfaceModels: hydrostatic_free_surface_tracer_tendency
using KernelAbstractions: @kernel, @index

################################################################################
# Hydrostatic free surface tendency kernel
################################################################################

@kernel function compute_hydrostatic_free_surface_GADc!(GADc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds GADc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end


################################################################################
# Matrix processing helpers
################################################################################

"""
    process_sparse_matrix(M, MATRIX_PROCESSING) -> SparseMatrixCSC

Apply the requested matrix processing to M. Valid values for MATRIX_PROCESSING:
- "raw": no processing
- "symfill": add zero entries at (j,i) for every existing (i,j)
- "dropzeros": remove stored zeros
- "symdrop": keep (i,j) only if (j,i) also exists, then drop zeros
"""
function process_sparse_matrix(M, MATRIX_PROCESSING)
    if MATRIX_PROCESSING == "symfill"
        I_idx, J_idx, V = findnz(M)
        M = sparse([I_idx; J_idx], [J_idx; I_idx], [V; zeros(length(V))], size(M)...)
        @info "After symfill: nnz=$(nnz(M))"
    elseif MATRIX_PROCESSING == "dropzeros"
        nnz_before = nnz(M)
        dropzeros!(M)
        @info "After dropzeros: nnz $nnz_before → $(nnz(M))"
    elseif MATRIX_PROCESSING == "symdrop"
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
    return M
end

"""
    compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; LUMP_AND_SPRAY=false)

If LUMP_AND_SPRAY is true, compute LUMP, SPRAY, Mc and save to matrices_dir.
Returns (; Mc, LUMP, SPRAY) — all `nothing` if LUMP_AND_SPRAY is false.
"""
function compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; LUMP_AND_SPRAY = false)
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
        flush(stdout); flush(stderr)
        return (; Mc, LUMP, SPRAY)
    else
        @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no)"
        flush(stdout); flush(stderr)
        return (; Mc = nothing, LUMP = nothing, SPRAY = nothing)
    end
end
