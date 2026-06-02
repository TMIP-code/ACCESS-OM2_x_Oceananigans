"""
Stitch the per-rank 1-year periodic age FieldTimeSeries (written by `run1yrNK` /
`src/run_periodic_1year.jl`) into a single combined file that the downstream
consumers expect.

Partitioned (`PARTITION ≠ 1x1`) runs write per-rank, interior-only files:
```
.../periodic/{MC}/1year/{solver_tag}/age_periodic_1year{omega}_rank0.jld2
.../periodic/{MC}/1year/{solver_tag}/age_periodic_1year{omega}_rank1.jld2
```
but `compute_ventilation_diagnostic.jl`, `plot_periodic_1year_age.jl`, and
`animate_ventilation.jl` all load a single combined `age_periodic_1year{omega}.jld2`
via `FieldTimeSeries(file, "age")`. This script reads the per-rank files, stitches
each snapshot into a global interior array, and streams a combined FieldTimeSeries
to disk (one snapshot at a time, via the `OnDisk()` backend).

Fully CPU / serial: no GPU, no MPI, no model build. The per-rank files are already
on disk after the GPU run, so this is decoupled and can also back-fill existing
per-rank-only trees. The per-rank files are kept (the combined file is authoritative).

Usage — interactive (CPU node, no GPU needed):
```
qsub -I -P y99 -l mem=47GB -q express -l walltime=00:30:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/combine_periodic_1year.jl")
```

Environment variables (same as the consumers):
  PARENT_MODEL     – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE / W_FORMULATION / ADVECTION_SCHEME / TIMESTEPPER – model config
  LINEAR_SOLVER    – Pardiso | ParU | UMFPACK  (default: Pardiso)
  LUMP_AND_SPRAY   – no | AxB  (default: no; e.g. 2x2)
  OMEGA            – all | z<depth>  (default: all)
  PARTITION_X / PARTITION_Y – partition layout (default: 1 / 1)
"""

@info "Loading packages for 1-year periodic age combine"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: halo_size
using Oceananigans.Units: days
year = years = 365.25days

using Statistics
using JLD2

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, experiment_dir, outputdir) = load_project_config()

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = require_env("MODEL_CONFIG")

LINEAR_SOLVER = get(ENV, "LINEAR_SOLVER", "Pardiso")
ls = parse_lump_and_spray()
LUMP_AND_SPRAY = ls.on
lumpspray_tag = ls.tag
solver_tag = "$(LINEAR_SOLVER)_$(lumpspray_tag)"

omega = parse_omega()
omega_suffix = omega.suffix

px = parse(Int, get(ENV, "PARTITION_X", "1"))
py = parse(Int, get(ENV, "PARTITION_Y", "1"))
nranks = px * py

file_base = "age_periodic_1year$(omega_suffix)"

# Resolve the 1-year dir, trying the current tag then legacy "LSprec"/"prec"
# (older runs used the `yes`→LSprec naming; current parse gives e.g. Q2x2).
solver_tag_candidates = unique(["$(LINEAR_SOLVER)_$(lumpspray_tag)", "$(LINEAR_SOLVER)_LSprec", "$(LINEAR_SOLVER)_prec"])
periodic_1year_dir = joinpath(outputdir, "periodic", model_config, "1year", solver_tag)
for st in solver_tag_candidates
    d = joinpath(outputdir, "periodic", model_config, "1year", st)
    # Match on the rank-0 file for partitioned runs, or the combined file for serial.
    if isfile(joinpath(d, "$(file_base)_rank0.jld2")) || isfile(joinpath(d, "$(file_base).jld2"))
        global solver_tag = st
        global periodic_1year_dir = d
        break
    end
end

combined_file = joinpath(periodic_1year_dir, "$(file_base).jld2")

@info "1-year periodic age combine configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- MODEL_CONFIG     = $model_config"
@info "- LINEAR_SOLVER    = $LINEAR_SOLVER"
@info "- LUMP_AND_SPRAY   = $LUMP_AND_SPRAY (tag: $solver_tag)"
@info "- OMEGA            = $(omega.tag) (suffix='$(omega_suffix)')"
@info "- PARTITION        = $(px)x$(py)  (nranks=$nranks)"
@info "- periodic_1year_dir = $periodic_1year_dir"
@info "- combined_file    = $combined_file"
flush(stdout); flush(stderr)

################################################################################
# Serial guard: nothing to stitch — the run wrote the combined file directly.
################################################################################

if nranks == 1
    isfile(combined_file) || error(
        "Serial (1x1) run but combined file is missing: $combined_file\n" *
            "Expected `run_periodic_1year.jl` to have written it directly.",
    )
    @info "Serial run: combined file already present, nothing to combine."
    @info "Done: $combined_file"
    exit(0)
end

# Distributed: the per-rank files must exist.
for r in 0:(nranks - 1)
    rf = joinpath(periodic_1year_dir, "$(file_base)_rank$(r).jld2")
    isfile(rf) || error("Per-rank file not found: $rf")
end

################################################################################
# Load serial CPU grid + interior dims / wet mask
################################################################################

grid_file = joinpath(experiment_dir, "grid.jld2")
@info "Loading serial CPU grid from $grid_file"
flush(stdout); flush(stderr)
grid = load_tripolar_grid(grid_file, CPU())

(; wet3D, idx, Nidx) = compute_wet_mask(grid)
Nx′, Ny′, Nz′ = size(wet3D)
# Per-rank files are written by JLD2Writer including the grid's halos, so strip them
# on read. Derive the halo from the grid (auto-adapts to e.g. WENO5's larger z-halo).
Hx, Hy, Hz = halo_size(grid)
@info "Interior grid: (Nx′=$Nx′, Ny′=$Ny′, Nz′=$Nz′); wet cells = $Nidx; halo = ($Hx, $Hy, $Hz)"

################################################################################
# Discover iterations + times from rank 0
################################################################################

iters = list_iterations_rank0(periodic_1year_dir, file_base, "age")
n_snapshots = length(iters)
@info "Found $n_snapshots snapshots (iters $(iters[1]) … $(iters[end]))"

times = jldopen(joinpath(periodic_1year_dir, "$(file_base)_rank0.jld2"), "r") do f
    [Float64(f["timeseries/t/$it"]) for it in iters]
end

################################################################################
# Stitch each snapshot and stream to the combined OnDisk FieldTimeSeries
################################################################################

@info "Writing combined FieldTimeSeries to $combined_file"
flush(stdout); flush(stderr)

fts = FieldTimeSeries{Center, Center, Center}(grid, times; backend = OnDisk(), path = combined_file, name = "age")
age_field = CenterField(grid)

max_age_yr = 0.0
for (n, it) in enumerate(iters)
    global_age, t = load_distributed_snapshot_by_base(periodic_1year_dir, file_base, "age", it, px, py, Nx′, Ny′; halo = (Hx, Hy, Hz))
    @assert size(global_age) == (Nx′, Ny′, Nz′) "Snapshot $it stitched to $(size(global_age)), expected ($Nx′, $Ny′, $Nz′)"

    # Physical-age check on wet cells (project rule: error, do not sanitize).
    age_wet = global_age[wet3D]
    all(isfinite, age_wet) || error("Non-finite age at snapshot $it (iter, t=$t s)")
    snap_max_yr = maximum(age_wet) / year
    snap_max_yr ≤ 10_000 || error("Unphysical age $(round(snap_max_yr; digits = 1)) yr (> 10,000 yr) at snapshot $it (t=$t s)")
    global max_age_yr = max(max_age_yr, snap_max_yr)

    interior(age_field) .= global_age
    set!(fts, age_field, n)
    @info "  wrote snapshot $n/$n_snapshots (iter $it, t=$(round(t / year; digits = 4)) yr, max age $(round(snap_max_yr; digits = 1)) yr)"
    flush(stdout); flush(stderr)
end

################################################################################
# Silent-F guard: fail loudly if the combined file did not materialise.
################################################################################

isfile(combined_file) || error("Combine finished but combined file is missing: $combined_file")

@info "Combine complete: $n_snapshots snapshots, max age $(round(max_age_yr; digits = 1)) yr"
@info "Done: $combined_file"
