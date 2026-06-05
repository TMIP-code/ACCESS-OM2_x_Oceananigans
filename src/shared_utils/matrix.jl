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

"""
    build_precond_Q(M, wet3D, v1D, ls, stop_time, MATRIX_PROCESSING) -> (; Q, LUMP, SPRAY, Mc)

Build the NK lump-and-spray preconditioner matrix `Q` **bit-for-bit identically** to how
`solve_periodic_NK.jl` builds it, so that a factor of `Q` computed offline is a valid
preconditioner for the in-job NK solve. Single source of truth shared by
`solve_periodic_NK.jl`, `benchmark_precond_solve.jl`, and the offline factorizer.

Steps (exactly as the original NK block):
  coarsen RAW M → `Mc = LUMP*M*SPRAY` → `Q = stop_time .* Mc` → `process_sparse_matrix(Q, …)`.

`ls` is the named tuple from `parse_lump_and_spray()` (uses `ls.di, ls.dj, ls.dk, ls.on`).
When `ls.on == false`, `LUMP = SPRAY = I`, `Mc = M`, and `Q = stop_time .* M` (no coarsening).

The `stop_time` scalar is uniform on the nonzeros — cosmetic for factorization *cost/memory*,
but it changes the *numeric* factor, so it MUST match what NK uses
(`stop_time = n_months * prescribed_Δt`, = 1 yr for `N_MONTHS=12`).
"""
function build_precond_Q(M, wet3D, v1D, ls, stop_time, MATRIX_PROCESSING)
    if ls.on
        @info "Computing LUMP and SPRAY matrices (di=$(ls.di), dj=$(ls.dj), dk=$(ls.dk))"
        flush(stdout); flush(stderr)
        LUMP, SPRAY, v_c = OceanTransportMatrixBuilder.lump_and_spray(
            wet3D, v1D, M; di = ls.di, dj = ls.dj, dk = ls.dk,
        )
        Mc = LUMP * M * SPRAY
        @info "Coarsened Jacobian Mc: $(size(Mc, 1))×$(size(Mc, 2)), nnz=$(nnz(Mc))"
        Q = copy(Mc)
        Q.nzval .*= stop_time
    else
        @info "Skipping LUMP/SPRAY (LUMP_AND_SPRAY=no); using full Q = stop_time * M"
        LUMP = I
        SPRAY = I
        Mc = M
        Q = copy(M)
        Q.nzval .*= stop_time
    end
    flush(stdout); flush(stderr)

    @info "Processing Q (MATRIX_PROCESSING=$MATRIX_PROCESSING)"
    Q = process_sparse_matrix(Q, MATRIX_PROCESSING)
    return (; Q, LUMP, SPRAY, Mc)
end


################################################################################
# Offline factor save / load / apply — stdlib UMFPACK (zero new deps)
#
# The SuiteSparse UMFPACK C handle is not serializable, but the factor *pieces*
# (L, U, p, q, Rs) are plain Julia objects and reproduce the factor via the
# UMFPACK convention   (Rs .* A)[p, q] == L * U.
# These let us factorize Q in one process, save to JLD2, and re-apply Q⁻¹ in a
# fresh process — the offline-preconditioner reuse cycle.
################################################################################

"""
    save_umfpack_factor(path, F)

Save a SuiteSparse UMFPACK factorization `F = lu(Q)` to JLD2 as the plain-Julia
pieces `L, U, p, q, Rs`. Returns `path`.
"""
function save_umfpack_factor(path, F)
    jldsave(path; L = F.L, U = F.U, p = F.p, q = F.q, Rs = F.Rs)
    return path
end

"""
    load_umfpack_factor(path) -> (; L, U, p, q, Rs)

Reload the pieces written by [`save_umfpack_factor`](@ref).
"""
function load_umfpack_factor(path)
    d = load(path)
    return (; L = d["L"], U = d["U"], p = d["p"], q = d["q"], Rs = d["Rs"])
end

"""
    apply_umfpack_factor(fac, b) -> x

Solve `Q x = b` from the saved UMFPACK pieces `fac = (; L, U, p, q, Rs)`, using
`(Rs .* Q)[p, q] == L * U`  ⇒  `x[q] = U \\ (L \\ (Rs .* b)[p])`. `L` is unit lower
triangular and `U` upper triangular (sparse triangular solves).
"""
function apply_umfpack_factor(fac, b)
    (; L, U, p, q, Rs) = fac
    c = (Rs .* b)[p]
    y = LowerTriangular(L) \ c
    z = UpperTriangular(U) \ y
    x = similar(z)
    x[q] = z
    return x
end
