################################################################################
# Land-mask-aware MPI load balancing for tripolar grids.
#
# Adapted from `_assess_y_load!` + `calculate_local_N` in
# https://github.com/CliMA/ClimaOcean.jl/discussions/665#discussioncomment-14737556
# but operating on the saved `bottom` array on the CPU rather than a kernel
# on a TripolarGrid (we use a custom OrthogonalSphericalShellGrid loaded
# from JLD2, not a TripolarGrid).
################################################################################

using JLD2: load

"""
    compute_lb_y_sizes(grid_file, nranks_y; min_size=0) -> NTuple{nranks_y, Int}

Read the saved 2D `bottom` from `grid_file` and return per-rank y-slab sizes
that balance the wet-cell load (≈ same `count(bottom[:,j] .< 0)` per slab).

Greedy split on `wet[j] = count(bottom[:,j] .< 0)`. Each rank gets
contiguous j-rows; the last rank absorbs the remainder so `sum == Ny`.

`min_size` (default 0) enforces a per-rank floor; pass `Hy + 2` to keep the
top-rank fold-halo warning at `build_underlying_grid` line ~479 silent.

Errors loudly if `sum(wet) == 0` — `bottom < 0 ⇒ ocean` is the Oceananigans
convention (z increases upward); a violation means the saved `bottom` was
written under a different sign convention upstream.
"""
function compute_lb_y_sizes(grid_file::AbstractString, nranks_y::Int; min_size::Int = 0)
    bottom = load(grid_file, "bottom")
    Nx, Ny = size(bottom)

    wet = [count(<(0), view(bottom, :, j)) for j in 1:Ny]
    total_wet = sum(wet)
    @assert total_wet > 0 "no wet cells found in `bottom` (sign convention violated): expected `bottom < 0` to mean ocean (Oceananigans convention, z increases upward)"

    target = total_wet / nranks_y
    local_Ny = zeros(Int, nranks_y)
    cum = 0
    j = 1
    for r in 1:(nranks_y - 1)
        slab = 0
        while j ≤ Ny && cum + slab < target * r
            slab += wet[j]
            local_Ny[r] += 1
            j += 1
        end
        cum += slab
    end
    local_Ny[end] = Ny - sum(local_Ny[1:(end - 1)])

    if min_size > 0
        # Insurance: shift rows from the largest neighbour into any rank that
        # falls below `min_size`. Expected to be a no-op for OM2-025/1x2 (Ny=1080,
        # Hy+2=15 ⇒ floor far below either slab), but cheap to keep.
        for r in 1:nranks_y
            while local_Ny[r] < min_size
                donor = argmax(local_Ny)
                donor == r && error("compute_lb_y_sizes: cannot satisfy min_size=$min_size with Ny=$Ny across $nranks_y ranks")
                local_Ny[donor] -= 1
                local_Ny[r] += 1
            end
        end
    end

    @assert sum(local_Ny) == Ny "compute_lb_y_sizes: sum(local_Ny)=$(sum(local_Ny)) != Ny=$Ny"
    return Tuple(local_Ny)
end
