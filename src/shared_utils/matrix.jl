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
        M = M .* (M_t .≠ 0)
        dropzeros!(M)
        @info "After symdrop: nnz $nnz_before → $(nnz(M))"
    else
        @info "No matrix processing applied (raw)"
    end
    flush(stdout); flush(stderr)
    return M
end

"""
    compute_and_save_coarsening(M, wet3D, v1D, matrices_dir; di, dj, dk, on, tag)

When `on=true`, compute LUMP, SPRAY, Mc for a (`di`, `dj`, `dk`) coarsening
and save them to `matrices_dir/{tag}/` (e.g. `matrices_dir/Q5x5/Mc.jld2`).
When `on=false`, returns named-tuple of `nothing`.

The parsed kwargs come from `parse_lump_and_spray()`.
"""
function compute_and_save_coarsening(
        M, wet3D, v1D, matrices_dir;
        di::Integer, dj::Integer, dk::Integer, on::Bool, tag::AbstractString,
    )
    if on
        @info "Computing LUMP and SPRAY matrices (di=$di, dj=$dj, dk=$dk)"
        flush(stdout); flush(stderr)
        LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(wet3D, v1D, M; di, dj, dk)
        @info "LUMP ($(size(LUMP, 1))×$(size(LUMP, 2)), nnz=$(nnz(LUMP)))"
        @info "SPRAY ($(size(SPRAY, 1))×$(size(SPRAY, 2)), nnz=$(nnz(SPRAY)))"
        Mc = LUMP * M * SPRAY
        @info "Coarsened matrix Mc ($(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc)))"

        save_dir = joinpath(matrices_dir, tag)
        mkpath(save_dir)
        jldsave(joinpath(save_dir, "LUMP.jld2"); LUMP)
        jldsave(joinpath(save_dir, "SPRAY.jld2"); SPRAY)
        jldsave(joinpath(save_dir, "Mc.jld2"); Mc)
        flush(stdout); flush(stderr)
        return (; Mc, LUMP, SPRAY)
    else
        @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no)"
        flush(stdout); flush(stderr)
        return (; Mc = nothing, LUMP = nothing, SPRAY = nothing)
    end
end
