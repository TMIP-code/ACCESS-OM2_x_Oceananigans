"""
Permutation-based 1D wet-cell scatter/gather for partitioned NK.

Used by `periodic_solver_common.jl` to move the NK iterate (a 1D wet-cell
vector of length `Nidx_global` held on rank 0) to and from the rank-local
1D wet-cell vectors that drive each rank's `Φ!` slab.

For `arch::AbstractArchitecture` (serial), `scatter!`/`gather!` collapse to a
single `copyto!`. For `arch::Distributed`, they call `MPI.Scatterv!` /
`MPI.Gatherv!` against rank-0 send/recv buffers, with a precomputed
permutation that maps each rank's local column-major wet-cell order to the
global column-major wet-cell order.

The distributed methods reference module-scope globals set up by
periodic_solver_common.jl: `perm`, `send_buf`, `recv_buf`, `counts`,
`displs`, `Nidx_global`, `rank`, `COMM`.
"""

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.DistributedComputations: Distributed
using MPI

"""
    build_global_permutation(wet3D_global, partition_y_sizes)
        → (perm, counts, displs)

`perm[k]` is the global wet-cell index of the k-th cell in the permuted send
buffer (ranks concatenated in local column-major order). `counts[r]` is the
number of wet cells in rank r's y-slab; `displs[r] = sum(counts[1:r-1])`.
"""
function build_global_permutation(wet3D_global, partition_y_sizes)
    Nx, Ny, Nz = size(wet3D_global)
    nranks = length(partition_y_sizes)
    @assert sum(partition_y_sizes) == Ny "Sum of per-rank Ny ($(sum(partition_y_sizes))) ≠ global Ny ($Ny)"

    # Global wet-cell index (column-major order; 0 = dry)
    global_idx_of = zeros(Int, Nx, Ny, Nz)
    k_g = 0
    @inbounds for kk in 1:Nz, jj in 1:Ny, ii in 1:Nx
        if wet3D_global[ii, jj, kk]
            k_g += 1
            global_idx_of[ii, jj, kk] = k_g
        end
    end
    Nidx_global = k_g

    # Per-rank y-range start (1-based offset to the first row owned by rank r)
    y_offsets = Vector{Int}(undef, nranks)
    y_offsets[1] = 0
    for r in 2:nranks
        y_offsets[r] = y_offsets[r - 1] + partition_y_sizes[r - 1]
    end

    # Per-rank wet-cell counts (= sum over rank r's slab of wet cells)
    counts = zeros(Int, nranks)
    for r in 1:nranks
        jrange = (y_offsets[r] + 1):(y_offsets[r] + partition_y_sizes[r])
        c = 0
        @inbounds for kk in 1:Nz, jj in jrange, ii in 1:Nx
            wet3D_global[ii, jj, kk] && (c += 1)
        end
        counts[r] = c
    end

    displs = [sum(view(counts, 1:(r - 1))) for r in 1:nranks]
    @assert sum(counts) == Nidx_global "Per-rank counts ($(sum(counts))) sum ≠ Nidx_global ($Nidx_global)"

    # Permutation: walk each rank's slab in local column-major order; record
    # the corresponding global wet-cell index.
    perm = Vector{Int}(undef, Nidx_global)
    @inbounds for r in 1:nranks
        jrange = (y_offsets[r] + 1):(y_offsets[r] + partition_y_sizes[r])
        m = 0
        offset = displs[r]
        for kk in 1:Nz, jj in jrange, ii in 1:Nx
            if wet3D_global[ii, jj, kk]
                m += 1
                perm[offset + m] = global_idx_of[ii, jj, kk]
            end
        end
    end

    return perm, counts, displs
end

# ── scatter!/gather! hot-path ───────────────────────────────────────────

# Serial fallthrough: 1D copy. (In serial, Nidx_local == Nidx_global.)
scatter!(age_local, age_global, ::AbstractArchitecture) = copyto!(age_local, age_global)
gather!(age_global, age_local, ::AbstractArchitecture) = copyto!(age_global, age_local)

# Distributed: permuted Scatterv/Gatherv against rank-0 buffers.
# Module-scope globals are resolved at call time.
function scatter!(age_local, age_global, ::Distributed)
    if rank == 0
        @inbounds for k in 1:Nidx_global
            send_buf[k] = age_global[perm[k]]
        end
        MPI.Scatterv!(MPI.VBuffer(send_buf, counts, displs), age_local, 0, COMM)
    else
        MPI.Scatterv!(nothing, age_local, 0, COMM)
    end
    return age_local
end

function gather!(age_global, age_local, ::Distributed)
    if rank == 0
        MPI.Gatherv!(age_local, MPI.VBuffer(recv_buf, counts, displs), 0, COMM)
        @inbounds for k in 1:Nidx_global
            age_global[perm[k]] = recv_buf[k]
        end
    else
        MPI.Gatherv!(age_local, nothing, 0, COMM)
    end
    return age_global
end
