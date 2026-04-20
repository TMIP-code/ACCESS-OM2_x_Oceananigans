################################################################################
# Land-mask-aware MPI load balancing for tripolar grids.
#
# Adapted from `_assess_y_load!` + `calculate_local_N` in
# https://github.com/CliMA/ClimaOcean.jl/discussions/665#discussioncomment-14737556
# but operating on saved JLD2 arrays on the CPU rather than a kernel on a
# TripolarGrid (we use a custom OrthogonalSphericalShellGrid loaded from
# JLD2, not a TripolarGrid).
################################################################################

using JLD2: load

"""
    compute_wet_load_per_y_row(grid_file; method::Symbol) -> Vector{Int}

Return `wet[j]` = load per y-row used by the LB greedy splitter.

`method` selects the load proxy:

- `:surface` — count of wet surface columns at row `j`
  (`count(bottom[:, j] .< 0)`). Treats a shallow column the same as a
  deep one. Corresponds to the `_LBS` (Load-Balance Surface) mode.
- `:cell` — count of wet 3D cells at row `j`, i.e.
  `Σᵢ count(z_centers .> bottom[i, j])`. Weights each column by its
  z-depth. Corresponds to the `_LB` mode.

Errors loudly if the resulting load is zero — `bottom < 0 ⇒ ocean` is
the Oceananigans convention (z increases upward); a violation means the
saved `bottom` was written under a different sign convention upstream.
"""
function compute_wet_load_per_y_row(grid_file::AbstractString; method::Symbol)
    bottom = load(grid_file, "bottom")
    Nx, Ny = size(bottom)

    wet = if method === :surface
        [count(<(0), view(bottom, :, j)) for j in 1:Ny]
    elseif method === :cell
        z_faces = load(grid_file, "z_faces")
        Nz = length(z_faces) - 1
        z_centers = @. 0.5 * (z_faces[1:Nz] + z_faces[2:(Nz + 1)])
        [sum(count(>(bottom[i, j]), z_centers) for i in 1:Nx) for j in 1:Ny]
    else
        error("compute_wet_load_per_y_row: unknown method=$method (expected :surface or :cell)")
    end

    total = sum(wet)
    @assert total > 0 "no wet cells found in `bottom` (sign convention violated): expected `bottom < 0` to mean ocean (Oceananigans convention, z increases upward)"
    return wet
end

"""
    compute_lb_y_sizes(grid_file, nranks_y; method, min_size=0)
        -> NTuple{nranks_y, Int}

Greedy wet-load balanced y-partition. Returns per-rank y-slab sizes that
equalise `sum(wet[jstart:jend])` across ranks, where `wet[j]` is the
per-y-row load from `compute_wet_load_per_y_row(grid_file; method)`.

`method` is `:surface` (column count, `_LBS`) or `:cell` (3D cell count,
`_LB`).

`min_size` (default 0) enforces a per-rank floor; pass `Hy + 2` to keep
the top-rank fold-halo warning at `build_underlying_grid` line ~479
silent.
"""
function compute_lb_y_sizes(
        grid_file::AbstractString, nranks_y::Int;
        method::Symbol, min_size::Int = 0,
    )
    wet = compute_wet_load_per_y_row(grid_file; method)
    Ny = length(wet)
    total_wet = sum(wet)

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
        # falls below `min_size`. Expected to be a no-op for OM2-025 at 1x2/1x4
        # (Ny=1080, Hy+2=15 ⇒ floor far below any plausible slab), but cheap.
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

"""
    parse_load_balance_env(env::AbstractString="LOAD_BALANCE")
        -> (active::Bool, method::Union{Nothing,Symbol}, tag::String)

Parse the `LOAD_BALANCE` env var into a triple:
- `active`: whether LB is on.
- `method`: `:cell` for cell-based, `:surface` for surface-based,
  `nothing` when off.
- `tag`: partition/MODEL_CONFIG suffix: `""`, `"_LB"`, or `"_LBS"`.

Accepted values (case-insensitive): `no` (default), `cell`, `surface`,
`yes` (back-compat alias for `surface` — the original LB implementation
was surface-based).
"""
function parse_load_balance_env(varname::AbstractString = "LOAD_BALANCE")
    v = lowercase(get(ENV, varname, "no"))
    v == "yes" && (v = "surface")
    if v == "no"
        return (false, nothing, "")
    elseif v == "cell"
        return (true, :cell, "_LB")
    elseif v == "surface"
        return (true, :surface, "_LBS")
    else
        error("$varname must be one of: no | surface | cell (got: $v)")
    end
end
