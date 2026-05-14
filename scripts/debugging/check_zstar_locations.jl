"""
Find WHERE the zstar serial-vs-distributed diff is located.
Hypothesis: it lives in halo cells (or land cells whose BC fill differs
between serial and distributed paths), not in the interior wet cells.

For each rank, compute diff = rank - global_slice (using the best offsets
found previously), then report:
  - max|diff| over the full (halo-inclusive) array
  - max|diff| over interior-only (strip 2*H from each side)
  - top-5 cells by |diff| with their (i, j, k) coords + halo position
"""

using JLD2
using Printf

const MC_DIR = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"
const H = 7  # halo size (GRID_HX = GRID_HY = GRID_HZ = 7 per memory)
const Nx = 360
const Ny_total_serial = 300

function load_iter(path, field, iter_label)
    return jldopen(path, "r") do f
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
        it = iter_label == :first ? iters[1] : iters[end]
        return (it, f["timeseries/$field/$it"])
    end
end

function report_diff(rank_arr, sl, name, rank)
    diff = rank_arr .- sl
    flat = filter(isfinite, vec(diff))
    max_abs = isempty(flat) ? NaN : maximum(abs, flat)

    sz = size(diff)
    n1, n2 = sz[1], sz[2]
    # Conservative interior: strip 2H from each side in both i and j
    i1, i2 = (2H + 1), (n1 - 2H)
    j1, j2 = (2H + 1), (n2 - 2H)
    interior = diff[i1:i2, j1:j2, :]
    flat_int = filter(isfinite, vec(interior))
    max_abs_int = isempty(flat_int) ? NaN : maximum(abs, flat_int)

    @printf "  %s rank %d: full=%d×%d max|diff|=%.3e  interior(%d:%d, %d:%d) max|diff|=%.3e\n" name rank n1 n2 max_abs i1 i2 j1 j2 max_abs_int

    # Find top-5 cells by absolute diff
    abs_diff = abs.(diff)
    # set non-finite to -Inf so they don't dominate
    abs_diff[.!isfinite.(diff)] .= -Inf
    flat_idx = vec(abs_diff)
    perm = sortperm(flat_idx; rev = true)[1:min(5, length(flat_idx))]
    for p in perm
        cidx = CartesianIndices(size(diff))[p]
        v = diff[p]
        isfinite(v) || continue
        in_halo_i = cidx[1] <= 2H || cidx[1] > n1 - 2H
        in_halo_j = cidx[2] <= 2H || cidx[2] > n2 - 2H
        tag = (in_halo_i || in_halo_j) ? "HALO" : "INTERIOR"
        @printf "    %s top: (%3d,%3d,%2d) diff=%.3e  rank=%.3e global=%.3e\n" tag cidx[1] cidx[2] cidx[3] v rank_arr[cidx] (rank_arr[cidx] - v)
    end
    return
end

for zname in ("sigma_cc", "eta_n")
    println("=== $zname ===")
    serial_path = joinpath(MC_DIR, "$(zname)_diag_cpu.jld2")
    isfile(serial_path) || continue
    for it_label in (:first,)
        it_s, gdata = load_iter(serial_path, zname, it_label)
        println("  iter $it_s, serial shape=$(size(gdata))")
        # Rank 0 best offset was (0, 0); rank 1 best (0, 150)
        for (r, (xo, yo)) in zip(0:1, ((0, 0), (0, 150)))
            rank_path = joinpath(MC_DIR, "1x2", "$(zname)_diag_cpu_rank$(r).jld2")
            isfile(rank_path) || continue
            it_r, rdata = load_iter(rank_path, zname, it_label)
            nx_r, ny_r, nz_r = size(rdata, 1), size(rdata, 2), size(rdata, 3)
            sl = ndims(gdata) == 3 ? gdata[(xo + 1):(xo + nx_r), (yo + 1):(yo + ny_r), :] : gdata[(xo + 1):(xo + nx_r), (yo + 1):(yo + ny_r)]
            report_diff(rdata, sl, zname, r)
        end
    end
    println()
end
