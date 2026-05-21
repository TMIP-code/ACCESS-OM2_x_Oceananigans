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
using Statistics: mean
using Oceananigans.Architectures: CPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, immersed_cell

"""
    compute_wet_load_per_y_row(grid; method::Symbol) -> Vector{Int}
    compute_wet_load_per_y_row(grid_file::AbstractString; method::Symbol)
        -> Vector{Int}

Return `wet[j]` = load per y-row used by the LB greedy splitter.

The string overload loads the grid via `load_tripolar_grid(grid_file,
CPU())` (defined in `shared_utils/grid.jl` — must be in scope at call
time) and forwards to the grid-arg method.

`method` selects the load proxy:

- `:surface` — count of wet surface columns at row `j`
  (`count(bottom[:, j] .< 0)`). Treats a shallow column the same as a
  deep one. Corresponds to the `_LBS` (Load-Balance Surface) mode.
- `:cell` — count of wet (non-immersed) 3D cells at row `j`, computed
  via Oceananigans' `immersed_cell(i, j, k, grid)`. This is the
  ground-truth definition of "is this cell in the prognostic state":
  with `PartialCellBottom` it correctly counts partial bottom cells
  as wet (they ARE simulated). Corresponds to the `_LB` mode.
- `:mix` — equal-weighted mix of `:cell` and `:surface`, each
  normalised to its own total so both contribute the same total load
  (`wet_cells[j]/Σ wet_cells + wet_cols[j]/Σ wet_cols`). Aimed at
  workloads where some kernels are cell-bound and some are
  column-bound: balancing in between is better than fully balancing
  either. Corresponds to the `_LBmix` mode.
- `:cell_obsolete` — previous `:cell` definition: counts cells with
  `z_center > bottom[i,j]`. Under-counts the true wet-cell load
  because it excludes partial bottom cells, biasing the greedy
  splitter southward (rank 0 stops too early → rank 1 ends up with
  more real work). Kept only for back-comparison; do NOT use to build
  new `_LB` partitions.

Errors loudly if the resulting load is zero — `bottom < 0 ⇒ ocean` is
the Oceananigans convention (z increases upward); a violation means the
saved `bottom` was written under a different sign convention upstream.
"""
function compute_wet_load_per_y_row(grid_file::AbstractString; method::Symbol)
    # `:cell` and `:mix` need the full ImmersedBoundaryGrid to call
    # `immersed_cell`; `:surface` and `:cell_obsolete` only read raw arrays
    # from grid.jld2, so avoid constructing the grid (slow on OM2-01) when
    # we don't need to.
    if method === :cell || method === :mix
        grid = load_tripolar_grid(grid_file, CPU())
        return compute_wet_load_per_y_row(grid; method)
    end

    bottom = load(grid_file, "bottom")
    Nx, Ny = size(bottom)

    wet = if method === :surface
        [count(<(0), view(bottom, :, j)) for j in 1:Ny]
    elseif method === :cell_obsolete
        z_faces = load(grid_file, "z_faces")
        Nz = length(z_faces) - 1
        z_centers = @. 0.5 * (z_faces[1:Nz] + z_faces[2:(Nz + 1)])
        [sum(count(>(bottom[i, j]), z_centers) for i in 1:Nx) for j in 1:Ny]
    else
        error("compute_wet_load_per_y_row: unknown method=$method (expected :surface, :cell, :mix, or :cell_obsolete)")
    end

    total = sum(wet)
    @assert total > 0 "no wet cells found in `bottom` (sign convention violated): expected `bottom < 0` to mean ocean (Oceananigans convention, z increases upward)"
    return wet
end

function compute_wet_load_per_y_row(grid::ImmersedBoundaryGrid; method::Symbol)
    Nx, Ny, Nz = size(grid.underlying_grid)
    if method === :cell
        wet = [
            sum(!immersed_cell(i, j, k, grid) for i in 1:Nx, k in 1:Nz)
                for j in 1:Ny
        ]
        @assert sum(wet) > 0 "no non-immersed cells found in grid — check immersed boundary construction"
        return wet
    elseif method === :mix
        # Equal-weighted normalised mix of 3D cells and surface columns.
        cells_row = [
            sum(!immersed_cell(i, j, k, grid) for i in 1:Nx, k in 1:Nz)
                for j in 1:Ny
        ]
        cols_row = [
            count(any(!immersed_cell(i, j, k, grid) for k in 1:Nz) for i in 1:Nx)
                for j in 1:Ny
        ]
        Tc = sum(cells_row); Tk = sum(cols_row)
        @assert Tc > 0 && Tk > 0 "no non-immersed cells found in grid — check immersed boundary construction"
        return Float64[cells_row[j] / Tc + cols_row[j] / Tk for j in 1:Ny]
    else
        error("compute_wet_load_per_y_row(grid; method): only :cell and :mix are supported on a grid object (got method=$method). Use the grid_file string overload for :surface and :cell_obsolete.")
    end
end

"""
    compute_lb_y_sizes(grid_file, nranks_y; method, min_size=0)
        -> NTuple{nranks_y, Int}

Greedy wet-load balanced y-partition. Returns per-rank y-slab sizes that
equalise `sum(wet[jstart:jend])` across ranks, where `wet[j]` is the
per-y-row load from `compute_wet_load_per_y_row(grid_file; method)`.

`method` is one of:
- `:surface` — column count (`_LBS`).
- `:cell` — 3D cell count via Oceananigans `immersed_cell` (`_LB`).
- `:mix` — equal-weighted mix of `:cell` and `:surface` (`_LBmix`).
- `:minmax` — α-weighted mix where α is chosen by bisection to
  minimise `max(imb%(cells), imb%(surface))` (`_LBminmax`). Lands on
  the Pareto crossover where the cell- and surface-imbalances are
  approximately equal — both kernel classes see the same overload.
- `:cell_obsolete` — old z_center > bottom formula, back-comparison only.

`min_size` (default 0) enforces a per-rank floor; pass `Hy + 2` to keep
the top-rank fold-halo warning at `build_underlying_grid` line ~479
silent.
"""
function compute_lb_y_sizes(
        grid_file::AbstractString, nranks_y::Int;
        method::Symbol, min_size::Int = 0,
    )
    method === :minmax && return _compute_minmax_y_sizes(grid_file, nranks_y; min_size)
    wet = compute_wet_load_per_y_row(grid_file; method)
    return _greedy_split(wet, nranks_y; min_size)
end

"""
    _greedy_split(wet, nranks_y; min_size=0) -> NTuple{nranks_y, Int}

Core greedy y-slab splitter shared by all LB methods. Accepts any
`AbstractVector{<:Real}` so that `:mix` and `:minmax` (Float-valued
loads) work alongside the integer-count methods.
"""
function _greedy_split(wet::AbstractVector{<:Real}, nranks_y::Int; min_size::Int = 0)
    Ny = length(wet)
    total_wet = sum(wet)

    target = total_wet / nranks_y
    local_Ny = zeros(Int, nranks_y)
    cum = zero(eltype(wet))
    j = 1
    for r in 1:(nranks_y - 1)
        slab = zero(eltype(wet))
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
                donor == r && error("_greedy_split: cannot satisfy min_size=$min_size with Ny=$Ny across $nranks_y ranks")
                local_Ny[donor] -= 1
                local_Ny[r] += 1
            end
        end
    end

    @assert sum(local_Ny) == Ny "_greedy_split: sum(local_Ny)=$(sum(local_Ny)) ≠ Ny=$Ny"
    return Tuple(local_Ny)
end

"""
    _compute_minmax_y_sizes(grid_file, nranks_y; min_size=0)

Minmax LB: pick α ∈ [0,1] that minimises `max(imb%(cells),
imb%(surface))` for the α-weighted load
`wet_α[j] = α·cells[j]/Σcells + (1-α)·cols[j]/Σcols`. Bisection on
the sign of `imb%(cells) - imb%(surface)` (monotone in α modulo
integer-greedy stair-stepping); tracks best `max%` seen.
"""
function _compute_minmax_y_sizes(grid_file::AbstractString, nranks_y::Int; min_size::Int = 0)
    grid = load_tripolar_grid(grid_file, CPU())
    Nx, Ny, Nz = size(grid.underlying_grid)

    cells_row = [
        sum(!immersed_cell(i, j, k, grid) for i in 1:Nx, k in 1:Nz)
            for j in 1:Ny
    ]
    cols_row = [
        count(any(!immersed_cell(i, j, k, grid) for k in 1:Nz) for i in 1:Nx)
            for j in 1:Ny
    ]
    Tc = sum(cells_row); Tk = sum(cols_row)
    @assert Tc > 0 && Tk > 0 "no non-immersed cells found in grid — check immersed boundary construction"

    eval_α = function (α)
        wet = Float64[α * cells_row[j] / Tc + (1 - α) * cols_row[j] / Tk for j in 1:Ny]
        sizes = _greedy_split(wet, nranks_y; min_size)
        bounds = vcat(0, cumsum(collect(sizes)))
        per_cells = [sum(view(cells_row, (bounds[r] + 1):bounds[r + 1])) for r in 1:nranks_y]
        per_cols = [sum(view(cols_row, (bounds[r] + 1):bounds[r + 1])) for r in 1:nranks_y]
        ic = 100 * (maximum(per_cells) - mean(per_cells)) / mean(per_cells)
        is = 100 * (maximum(per_cols) - mean(per_cols)) / mean(per_cols)
        return sizes, ic, is
    end

    lo, hi = 0.0, 1.0
    best_sizes = first(eval_α(0.5))
    best_max = Inf
    for _ in 1:30
        α = (lo + hi) / 2
        sizes, ic, is = eval_α(α)
        m = max(ic, is)
        if m < best_max
            best_max = m
            best_sizes = sizes
        end
        # `ic - is` is monotone decreasing in α: higher α → more cell
        # weight → cells better balanced → ic smaller. Bisect toward
        # the crossover where ic ≈ is.
        if ic > is
            lo = α
        else
            hi = α
        end
    end
    return best_sizes
end

"""
    parse_load_balance_env(env::AbstractString="LOAD_BALANCE")
        -> (active::Bool, method::Union{Nothing,Symbol}, tag::String)

Parse the `LOAD_BALANCE` env var into a triple:
- `active`: whether LB is on.
- `method`: `:cell`, `:surface`, `:mix`, `:minmax`, or `nothing` when off.
- `tag`: partition/MODEL_CONFIG suffix: `""`, `"_LB"`, `"_LBS"`,
  `"_LBmix"`, or `"_LBminmax"`.

Accepted values (case-insensitive): `no` (default), `cell`, `surface`,
`mix`, `minmax`, `yes` (back-compat alias for `surface` — the original
LB implementation was surface-based).
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
    elseif v == "mix"
        return (true, :mix, "_LBmix")
    elseif v == "minmax"
        return (true, :minmax, "_LBminmax")
    else
        error("$varname must be one of: no | surface | cell | mix | minmax (got: $v)")
    end
end
