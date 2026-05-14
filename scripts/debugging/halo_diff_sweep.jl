"""
Comprehensive serial-vs-distributed diff sweep over all saved JLD2 fields, at
multiple halo-depth trim levels.

Why: outermost halos are allowed to be stale (they aren't read by interior
stencils), but if any of the immediate-neighbour halos (interior+1) differ
between serial and 1×2 distributed, that could feed an advection stencil and
contaminate the next interior cell. For centered2, interior+1 matters;
WENO3/5 cares about +2 as well.

For each saved field:
  1. Load global (1×1) parent array and each rank's (1×2) parent array.
  2. Detect Hx, Hy from a reference Center-y field (`age`) once.
  3. Print serial shape, rank shape, and the global-index range that each
     rank's parent covers (so you can verify alignment by eye).
  4. For trim levels {interior, +1 halo, +2 halos, full halos} compute
     max|diff|, count NaNs, count finite cells.

Usage:
  DURATION_TAG=diag      julia --project scripts/debugging/halo_diff_sweep.jl
  DURATION_TAG=diag_cpu  julia --project scripts/debugging/halo_diff_sweep.jl
  DURATION_TAG=1year     julia --project scripts/debugging/halo_diff_sweep.jl
"""

using JLD2
using OffsetArrays
using Printf
using Statistics

const MC_DIR = "outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2"
const DURATION_TAG = get(ENV, "DURATION_TAG", "diag")
const PY = 2

const FIELDS = [
    "age", "u", "v", "w", "eta",
    "sigma_cc", "dt_sigma", "eta_n",
    "u_manual", "v_manual", "w_manual", "eta_manual",
    "u_fts", "v_fts", "eta_fts",
]

function iter_choices(path, field)
    return jldopen(path, "r") do f
        haskey(f, "timeseries/$field") || return nothing
        iters = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(f["timeseries/$field"])))))
        isempty(iters) && return nothing
        return (first_nonzero = length(iters) >= 2 ? iters[2] : iters[1], last = iters[end])
    end
end

function load_field(path, field, iter)
    return jldopen(path, "r") do f
        return f["timeseries/$field/$iter"]
    end
end

# Trim by `tH_x` in x and `tH_y` in y, leaving z untouched.
# Operates on the bare data (1-indexed) — caller passes `parent(...)` if OffsetArray.
function trim_xy(arr, tH_x, tH_y)
    n1, n2 = size(arr, 1), size(arr, 2)
    (tH_x * 2 >= n1 || tH_y * 2 >= n2) && return similar(arr, 0)
    if ndims(arr) >= 3
        return arr[(tH_x + 1):(n1 - tH_x), (tH_y + 1):(n2 - tH_y), :]
    else
        return arr[(tH_x + 1):(n1 - tH_x), (tH_y + 1):(n2 - tH_y)]
    end
end

function summarize_diff(diff_arr)
    flat = vec(diff_arr)
    n_total = length(flat)
    n_nan = count(!isfinite, flat)
    finite = filter(isfinite, flat)
    n_finite = length(finite)
    max_abs = isempty(finite) ? NaN : maximum(abs, finite)
    return (; max_abs, n_total, n_finite, n_nan)
end

axes_str(a) = string(axes(a))

# --- Step 1: detect Hx, Hy from `age` (Center-Center-Center) ---
age_serial = joinpath(MC_DIR, "age_$(DURATION_TAG).jld2")
isfile(age_serial) || error("Need $(age_serial) to detect halo size")
age_iters = iter_choices(age_serial, "age")
age_iters === nothing && error("Cannot read iters from $(age_serial)")
gage = load_field(age_serial, "age", age_iters.last)
gage_data = gage isa OffsetArray ? parent(gage) : gage

# Center-Center interior on OM2-1 is 360 × 300
const Nx_INT = 360
const Ny_INT = 300
const Hx = (size(gage_data, 1) - Nx_INT) ÷ 2
const Hy = (size(gage_data, 2) - Ny_INT) ÷ 2
@printf "Detected Hx=%d, Hy=%d from %s (size=%s)\n" Hx Hy basename(age_serial) size(gage_data)

# --- Step 2: figure out each rank's interior Ny (Center-y) to compute y_offsets ---
age_rank0 = joinpath(MC_DIR, "1x2", "age_$(DURATION_TAG)_rank0.jld2")
isfile(age_rank0) || error("Need $(age_rank0)")
ar0 = load_field(age_rank0, "age", age_iters.last)
ar0_data = ar0 isa OffsetArray ? parent(ar0) : ar0
const Ny_INT_RANK_CENTER = size(ar0_data, 2) - 2 * Hy   # rank 0 Center-y interior size
@printf "Rank 0 age parent size=%s  → Center-y interior per rank = %d\n" size(ar0_data) Ny_INT_RANK_CENTER

# Cumulative offset of each rank's "south halo start" into the global parent (in y).
# For Center-y: rank r's parent occupies global rows [r*Ny_INT_RANK_CENTER + 1 : r*Ny_INT_RANK_CENTER + 2H + Ny_INT_RANK_CENTER]
const Y_OFF_CENTER = [r * Ny_INT_RANK_CENTER for r in 0:(PY - 1)]
@printf "Center-y y-offsets per rank: %s\n" Y_OFF_CENTER

println()
println("# Halo-depth diff sweep — DURATION_TAG=$DURATION_TAG, partition=1×$PY")
println()

for field in FIELDS
    serial_path = joinpath(MC_DIR, "$(field)_$(DURATION_TAG).jld2")
    isfile(serial_path) || continue
    its = iter_choices(serial_path, field)
    its === nothing && continue

    iter = field in ("age", "sigma_cc", "dt_sigma", "eta_n") ? its.last : its.first_nonzero
    gd_raw = load_field(serial_path, field, iter)
    gd = gd_raw isa OffsetArray ? parent(gd_raw) : gd_raw
    gd_axes = axes_str(gd_raw)

    nxs, nys = size(gd, 1), size(gd, 2)
    println("## $field  iter=$iter  serial size=$(size(gd_raw))  axes=$(gd_axes)")
    println()
    println("| rank | rank size | rank axes | global y-range covered | trim | max\\|diff\\| | n_finite | n_NaN |")
    println("|---|---|---|---|---|---|---|---|")

    for r in 0:(PY - 1)
        rank_path = joinpath(MC_DIR, "1x2", "$(field)_$(DURATION_TAG)_rank$(r).jld2")
        isfile(rank_path) || continue
        rd_raw = load_field(rank_path, field, iter)
        rd = rd_raw isa OffsetArray ? parent(rd_raw) : rd_raw

        nxr, nyr = size(rd, 1), size(rd, 2)
        # rank r's parent covers global parent rows [Y_OFF_CENTER[r+1]+1 : Y_OFF_CENTER[r+1]+nyr] in y.
        # In x (periodic, single column): rank's parent covers [1 : nxr] (== [1 : nxs] since nxr==nxs).
        y_lo = Y_OFF_CENTER[r + 1] + 1
        y_hi = Y_OFF_CENTER[r + 1] + nyr

        if y_hi > nys
            println("| $r | $(size(rd_raw)) | $(axes_str(rd_raw)) | SLICE OOB ($y_lo:$y_hi vs nys=$nys) | — | — | — | — |")
            continue
        end
        if size(rd, 1) != nxs
            println("| $r | $(size(rd_raw)) | $(axes_str(rd_raw)) | X SIZE MISMATCH ($nxr vs $nxs) | — | — | — | — |")
            continue
        end

        sl = if ndims(gd) >= 3
            gd[1:nxr, y_lo:y_hi, :]
        else
            gd[1:nxr, y_lo:y_hi]
        end
        if size(sl) != size(rd)
            println("| $r | $(size(rd_raw)) | $(axes_str(rd_raw)) | SHAPE MISMATCH $(size(sl)) vs $(size(rd)) | — | — | — | — |")
            continue
        end

        diff = rd .- sl

        levels = [
            ("interior", Hx, Hy),
            ("+1 halo", max(0, Hx - 1), max(0, Hy - 1)),
            ("+2 halos", max(0, Hx - 2), max(0, Hy - 2)),
            ("full halos", 0, 0),
        ]
        first_row = true
        for (label, tHx, tHy) in levels
            t = trim_xy(diff, tHx, tHy)
            s = summarize_diff(t)
            max_str = isnan(s.max_abs) ? "—" : @sprintf("%.3e", s.max_abs)
            lead = first_row ? "| $r | $(size(rd_raw)) | $(axes_str(rd_raw)) | y=$y_lo:$y_hi" : "| | | | "
            println(lead * " | $label | $max_str | $(s.n_finite) | $(s.n_nan) |")
            first_row = false
        end
    end
    println()
end
