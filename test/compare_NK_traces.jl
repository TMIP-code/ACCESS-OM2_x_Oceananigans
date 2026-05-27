"""
Compare per-Φ!-call trace files between two NK runs (typically serial vs
partitioned) to identify where their trajectories first diverge.

Each NK run with `TRACE_SOLVER_HISTORY=yes` saves one global age 3D field
per Φ! call into `…/periodic/{MC}_DTx{M}/[{GPU_TAG}/]NK/age_trace_iter_{####}_{JOBID}.jld2`.
We walk Φ!-call numbers in lock-step, compute per-call diff diagnostics
(max|·|, volume-weighted RMS, mean|·|, location of max), and produce:

1. A scan plot of max|diff| and vol-RMS|diff| vs Φ!-call number (semilog y).
2. For the first divergent call (`max|diff| > DIVERGE_TOL_YR`, default 1e-3 yr):
   `plot_age_diagnostics` on serial_age, partitioned_age, and (partitioned − serial)
   — same zonal-average × 4 basins + horizontal slices layout used by
   `compare_runs_across_architectures.jl`.

Usage — interactive (CPU node):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
REF_JOB_ID=168279435.gadi-pbs CMP_JOB_ID=168279436.gadi-pbs \\
GPU_TAG=1x2 TIMESTEP_MULT=4 \\
julia --project test/compare_NK_traces.jl
```

Env vars:
    REF_JOB_ID    – reference (serial) NK job ID  (e.g., "168279435.gadi-pbs")
    CMP_JOB_ID    – comparison (partitioned) NK job ID
    GPU_TAG       – partition subdir of the cmp run (e.g., "1x2"); empty if cmp is serial too
    REF_GPU_TAG   – partition subdir of the ref run (default: "" = serial)
    TIMESTEP_MULT – integer matching the {MC}_DTx{M} of both runs
    DIVERGE_TOL_YR – max|diff| threshold flagging "first divergence" (default 1e-3 yr)
    PARENT_MODEL, EXPERIMENT, TIME_WINDOW, and the 4 MODEL_CONFIG axes – standard
"""

@info "Loading packages for NK trace comparison"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Grids: znodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_field!
using Oceananigans.Architectures: CPU
using Oceananigans.Units: day, days, second, seconds
year = years = 365.25days

using CairoMakie
using Statistics
using JLD2
using Printf

include("../src/shared_functions.jl")

################################################################################
# Configuration
################################################################################

REF_JOB_ID = get(ENV, "REF_JOB_ID", "")
CMP_JOB_ID = get(ENV, "CMP_JOB_ID", "")
isempty(REF_JOB_ID) && error("REF_JOB_ID must be set (e.g., REF_JOB_ID=168279435.gadi-pbs)")
isempty(CMP_JOB_ID) && error("CMP_JOB_ID must be set (e.g., CMP_JOB_ID=168279436.gadi-pbs)")

GPU_TAG = get(ENV, "GPU_TAG", "1x2")
REF_GPU_TAG = get(ENV, "REF_GPU_TAG", "")
DIVERGE_TOL_YR = parse(Float64, get(ENV, "DIVERGE_TOL_YR", "1.0e-3"))

(; parentmodel, experiment_dir, outputdir) = load_project_config()
model_config = require_env("MODEL_CONFIG")

# Directory layout:
#   serial:      …/periodic/{MC}/NK/
#   partitioned: …/periodic/{MC}/{GPU_TAG}/NK/
ref_dir = isempty(REF_GPU_TAG) ?
    joinpath(outputdir, "periodic", model_config, "NK") :
    joinpath(outputdir, "periodic", model_config, REF_GPU_TAG, "NK")
cmp_dir = isempty(GPU_TAG) ?
    joinpath(outputdir, "periodic", model_config, "NK") :
    joinpath(outputdir, "periodic", model_config, GPU_TAG, "NK")

plot_output_dir = joinpath(cmp_dir, "compare_vs_$(isempty(REF_GPU_TAG) ? "serial" : REF_GPU_TAG)_$(REF_JOB_ID)_vs_$(CMP_JOB_ID)")
mkpath(plot_output_dir)

@info "NK trace comparison configuration"
@info "- PARENT_MODEL   = $parentmodel"
@info "- MODEL_CONFIG   = $model_config"
@info "- REF_JOB_ID     = $REF_JOB_ID (dir: $ref_dir)"
@info "- CMP_JOB_ID     = $CMP_JOB_ID (dir: $cmp_dir)"
@info "- DIVERGE_TOL_YR = $DIVERGE_TOL_YR"
@info "- Plot output    = $plot_output_dir"
flush(stdout); flush(stderr)

################################################################################
# Grid, wet mask, cell volumes (global)
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading grid"; flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx, Ny, Nz = size(wet3D)
vol = compute_volume(grid)
vol_3D = Array(interior(vol))
v1D = vol_3D[idx]
vol_norm = make_vol_norm(v1D, year)

@info "Grid loaded" Nx Ny Nz Nidx
flush(stdout); flush(stderr)

################################################################################
# Discover trace files (lock-step Φ!-call numbers)
################################################################################

# JLD2Writer with array_type=Array{Float64} on a DistributedField does NOT
# auto-gather — each rank writes its own local slab to `..._rank{r}.jld2`.
# Serial runs write a single global file (no _rank suffix).
function parse_gpu_tag(tag)
    isempty(tag) && return (1, 1)
    parts = split(tag, "x")
    return parse(Int, parts[1]), parse(Int, parts[2])
end
ref_px, ref_py = parse_gpu_tag(REF_GPU_TAG)
cmp_px, cmp_py = parse_gpu_tag(GPU_TAG)
@assert ref_px == 1 && cmp_px == 1 "PARTITION_X==1 assumed; got ref=$REF_GPU_TAG, cmp=$GPU_TAG"

function trace_regex(job_id, partitioned)
    return if partitioned
        Regex("^age_trace_iter_(\\d{4})_" * Base.escape_string(job_id) * "_rank0\\.jld2\$")
    else
        Regex("^age_trace_iter_(\\d{4})_" * Base.escape_string(job_id) * "\\.jld2\$")
    end
end

ref_partitioned = ref_py > 1
cmp_partitioned = cmp_py > 1
ref_glob_re = trace_regex(REF_JOB_ID, ref_partitioned)
cmp_glob_re = trace_regex(CMP_JOB_ID, cmp_partitioned)

function find_trace_calls(dir, regex)
    isdir(dir) || error("Trace directory not found: $dir")
    calls = Int[]
    for fn in readdir(dir)
        m = match(regex, fn)
        !isnothing(m) && push!(calls, parse(Int, m.captures[1]))
    end
    return sort!(calls)
end

ref_calls = find_trace_calls(ref_dir, ref_glob_re)
cmp_calls = find_trace_calls(cmp_dir, cmp_glob_re)
common_calls = sort!(collect(intersect(ref_calls, cmp_calls)))
@info "Discovered" length_ref = length(ref_calls) length_cmp = length(cmp_calls) length_common = length(common_calls)
isempty(common_calls) && error("No Φ!-call indices common to both runs — check job IDs / dirs")
flush(stdout); flush(stderr)

################################################################################
# Helper: load the final-time age slice from a trace file
################################################################################

"""
Read the largest iteration key under `timeseries/age` (the final state at
t = stop_time saved by `schedule = TimeInterval(stop_time)`) from one JLD2
file. Returns (age_array, t_seconds). The array includes halos.
"""
function _load_final_age_one_file(fpath)
    return jldopen(fpath, "r") do f
        iters = collect(keys(f["timeseries/age"]))
        iters_int = parse.(Int, filter(k -> !isnothing(tryparse(Int, k)), iters))
        sort!(iters_int)
        last_iter = string(last(iters_int))
        age = f["timeseries/age/$(last_iter)"]
        t = f["timeseries/t/$(last_iter)"]
        return age, t
    end
end

"""
Strip halos from a halo-inclusive parent array. Halo sizes (Hx, Hy, Hz)
are inferred from the difference vs the known interior `(Nx_int, Ny_int, Nz_int)`,
assuming symmetric halos in each dimension.

For a 2D-stored field (Nz_int < halo z-dim), the z-extent is not haloed.
"""
function strip_halos(arr, Nx_int, Ny_int, Nz_int)
    sx, sy, sz = size(arr, 1), size(arr, 2), size(arr, 3)
    Hx = (sx - Nx_int) ÷ 2
    Hy = (sy - Ny_int) ÷ 2
    Hz = (sz - Nz_int) ÷ 2
    @assert sx == Nx_int + 2Hx "x-halo not symmetric: parent=$sx int=$Nx_int Hx=$Hx"
    @assert sy == Ny_int + 2Hy "y-halo not symmetric: parent=$sy int=$Ny_int Hy=$Hy"
    z_range = sz > 2Hz ? ((Hz + 1):(Hz + Nz_int)) : (1:Nz_int)
    return arr[(Hx + 1):(Hx + Nx_int), (Hy + 1):(Hy + Ny_int), z_range]
end

"""
Load the final-state global age 3D array (INTERIOR only, halos stripped)
for Φ!-call `call_idx`. If `py > 1`, stitches per-rank interior slabs along
y. For px=1, no x-stitching needed.

`Ny_local_each` is the per-rank interior Ny for each rank (`length py`).
"""
function load_final_age(dir, call_idx, job_id, py, Nx_int, Ny_int, Nz_int, Ny_local_each)
    iter_str = @sprintf("%04d", call_idx)
    if py == 1
        fpath = joinpath(dir, "age_trace_iter_$(iter_str)_$(job_id).jld2")
        age_raw, t = _load_final_age_one_file(fpath)
        return strip_halos(age_raw, Nx_int, Ny_int, Nz_int), t
    end
    rank_slabs = Vector{Array{Float64, 3}}(undef, py)
    t_out = NaN
    for r in 0:(py - 1)
        fpath = joinpath(dir, "age_trace_iter_$(iter_str)_$(job_id)_rank$(r).jld2")
        age_raw, t = _load_final_age_one_file(fpath)
        # Strip halos with this rank's LOCAL Ny interior
        rank_slabs[r + 1] = strip_halos(age_raw, Nx_int, Ny_local_each[r + 1], Nz_int)
        t_out = t
    end
    # Stitch along y (dim 2) — interiors only, no halo overlap
    return cat(rank_slabs...; dims = 2), t_out
end

# Determine per-rank interior Ny for the cmp run (peek at rank-0 file).
function peek_rank_ny(dir, call_idx, job_id, py, Nx_int, Nz_int)
    iter_str = @sprintf("%04d", call_idx)
    nys = Int[]
    for r in 0:(py - 1)
        fpath = joinpath(dir, "age_trace_iter_$(iter_str)_$(job_id)_rank$(r).jld2")
        sample = _load_final_age_one_file(fpath)[1]
        sx, sy, _ = size(sample)
        Hx = (sx - Nx_int) ÷ 2
        # Halos are symmetric across x/y in this project (GRID_HX == GRID_HY,
        # and free-surface/advection add the same amount to both).
        Hy = Hx
        push!(nys, sy - 2Hy)
    end
    return nys
end

ref_ny_local = ref_py > 1 ? peek_rank_ny(ref_dir, first(common_calls), REF_JOB_ID, ref_py, Nx, Nz) : [Ny]
cmp_ny_local = cmp_py > 1 ? peek_rank_ny(cmp_dir, first(common_calls), CMP_JOB_ID, cmp_py, Nx, Nz) : [Ny]
@assert sum(ref_ny_local) == Ny "ref per-rank Ny sums to $(sum(ref_ny_local)), expected $Ny"
@assert sum(cmp_ny_local) == Ny "cmp per-rank Ny sums to $(sum(cmp_ny_local)), expected $Ny"
@info "Per-rank Ny" ref = ref_ny_local cmp = cmp_ny_local

sample_age, sample_t = load_final_age(ref_dir, first(common_calls), REF_JOB_ID, ref_py, Nx, Ny, Nz, ref_ny_local)
exp_shape = (Nx, Ny, Nz)
@info "Sample trace dims (ref, interior)" size_age = size(sample_age) sample_t_yr = sample_t / year
@assert size(sample_age) == exp_shape "Ref interior shape $(size(sample_age)) doesn't match wet3D shape $(exp_shape)"
sample_age_cmp, _ = load_final_age(cmp_dir, first(common_calls), CMP_JOB_ID, cmp_py, Nx, Ny, Nz, cmp_ny_local)
@info "Sample trace dims (cmp, stitched interior)" size_age = size(sample_age_cmp)
@assert size(sample_age_cmp) == exp_shape "Cmp stitched shape $(size(sample_age_cmp)) doesn't match wet3D shape $(exp_shape)"

################################################################################
# Per-call comparison
################################################################################

@info "Per-Φ!-call comparison"
@info @sprintf("  %5s  %14s  %14s  %14s  %20s", "call", "max|d|(yr)", "vol_rms(yr)", "mean|d|(yr)", "argmax(i,j,k)")
flush(stdout); flush(stderr)

scan = NamedTuple{
    (:call, :max_diff_yr, :vol_rms_yr, :mean_diff_yr, :imax, :jmax, :kmax),
    Tuple{Int, Float64, Float64, Float64, Int, Int, Int},
}[]
first_diverge_call = -1

for c in common_calls
    age_ref, _ = load_final_age(ref_dir, c, REF_JOB_ID, ref_py, Nx, Ny, Nz, ref_ny_local)
    age_cmp, _ = load_final_age(cmp_dir, c, CMP_JOB_ID, cmp_py, Nx, Ny, Nz, cmp_ny_local)
    diff_3D = age_cmp .- age_ref
    diff_1D = diff_3D[idx]      # wet-cell only
    max_d = maximum(abs, diff_1D)
    vn = vol_norm(diff_1D)
    md = mean(abs, diff_1D)
    # Location of max diff (in interior coordinates)
    diff_3D_masked = copy(diff_3D); diff_3D_masked[.!wet3D] .= 0.0
    cmax = argmax(abs.(diff_3D_masked))
    max_d_yr = max_d / year

    @info @sprintf(
        "  %5d  %14.3e  %14.3e  %14.3e  (%4d,%4d,%4d)",
        c, max_d_yr, vn, md / year, cmax[1], cmax[2], cmax[3]
    )
    push!(
        scan, (
            call = c, max_diff_yr = max_d_yr, vol_rms_yr = vn,
            mean_diff_yr = md / year, imax = cmax[1], jmax = cmax[2], kmax = cmax[3],
        )
    )

    if first_diverge_call == -1 && max_d_yr > DIVERGE_TOL_YR
        global first_diverge_call = c
    end
end
flush(stdout); flush(stderr)

################################################################################
# Scan plot: max|diff| and vol-RMS|diff| vs Φ!-call number
################################################################################

@info "Plotting divergence scan"
flush(stdout); flush(stderr)

fig = Figure(; size = (900, 500))
ax = Axis(
    fig[1, 1];
    title = "NK trace divergence (ref=$REF_JOB_ID, cmp=$CMP_JOB_ID)",
    xlabel = "Φ! call #",
    ylabel = "|cmp − ref| (yr)",
    yscale = log10,
)
calls_x = [s.call for s in scan]
max_y = [max(s.max_diff_yr, 1.0e-20) for s in scan]
rms_y = [max(s.vol_rms_yr, 1.0e-20) for s in scan]
mean_y = [max(s.mean_diff_yr, 1.0e-20) for s in scan]
lines!(ax, calls_x, max_y; label = "max|diff|", linewidth = 2)
lines!(ax, calls_x, rms_y; label = "vol-RMS|diff|", linewidth = 2)
lines!(ax, calls_x, mean_y; label = "mean|diff|", linewidth = 2)
hlines!(ax, [DIVERGE_TOL_YR]; color = :red, linestyle = :dash, label = "DIVERGE_TOL=$DIVERGE_TOL_YR yr")
axislegend(ax; position = :rb)
save(joinpath(plot_output_dir, "divergence_scan.png"), fig)
@info "Saved $(joinpath(plot_output_dir, "divergence_scan.png"))"
flush(stdout); flush(stderr)

################################################################################
# Spatial diagnostics at the first divergent call (and call 1 for reference)
################################################################################

plot_calls = unique(filter(>(0), [first(common_calls), first_diverge_call, last(common_calls)]))
@info "Spatial plots for calls $(plot_calls)" first_diverge_call
flush(stdout); flush(stderr)

for c in plot_calls
    age_ref, _ = load_final_age(ref_dir, c, REF_JOB_ID, ref_py, Nx, Ny, Nz, ref_ny_local)
    age_cmp, _ = load_final_age(cmp_dir, c, CMP_JOB_ID, cmp_py, Nx, Ny, Nz, cmp_ny_local)
    age_ref_yr = age_ref ./ year
    age_cmp_yr = age_cmp ./ year
    age_diff_yr = age_cmp_yr .- age_ref_yr

    wet_diff_yr = age_diff_yr[wet3D]
    max_abs = maximum(abs, wet_diff_yr)
    mean_abs = mean(abs, wet_diff_yr)
    @info @sprintf("Call %d spatial stats: max|diff|=%.3e yr, mean|diff|=%.3e yr", c, max_abs, mean_abs)

    iter_label = @sprintf("call%04d", c)

    # 1. Reference age field
    plot_age_diagnostics(
        age_ref_yr, grid, wet3D, vol_3D, plot_output_dir,
        "ref_$(REF_JOB_ID)_$(iter_label)"
        # Match compare_runs_across_architectures defaults; auto-range still
        # works for serial age in years
    )

    # 2. Comparison age field
    plot_age_diagnostics(
        age_cmp_yr, grid, wet3D, vol_3D, plot_output_dir,
        "cmp_$(CMP_JOB_ID)_$(iter_label)"
    )

    # 3. Difference field (balance colormap, scale = 3 × mean|diff|)
    n_levels = 11
    diff_scale = mean_abs > 0 ? 3 * mean_abs : 1.0e-12
    diff_range = (-diff_scale, diff_scale)
    diff_levels = range(diff_range[1], diff_range[2]; length = n_levels)
    plot_age_diagnostics(
        age_diff_yr, grid, wet3D, vol_3D, plot_output_dir,
        "diff_cmp_minus_ref_$(iter_label)";
        colorrange = diff_range, levels = diff_levels,
        colormap = cgrad(:balance, n_levels - 1, categorical = true),
        lowclip = :blue, highclip = :red,
        colorbar_label = "Δage (yr) = cmp − ref",
    )

    # 4. Horizontal slice at the depth of largest disagreement
    s = scan[findfirst(x -> x.call == c, scan)]
    @info "  argmax(diff) at (i,j,k) = ($(s.imax), $(s.jmax), $(s.kmax))"
    flush(stdout); flush(stderr)
end

@info "compare_NK_traces.jl complete — plots in $plot_output_dir"
flush(stdout); flush(stderr)
