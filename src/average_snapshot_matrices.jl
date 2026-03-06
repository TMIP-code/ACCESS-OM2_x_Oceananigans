"""
Average snapshot transport matrices and compare to the constant-field matrix.

Loads the 24 snapshot matrices produced by `create_snapshot_matrices.jl`,
computes three averages (avg12a, avg12b, avg24), compares them pairwise and
against the constant-field matrix M, and saves each average as `M.jld2` in
subdirectories of the TM output directory.

Usage — interactive:
```
qsub -I -P y99 -l mem=47GB -q normal -l walltime=01:00:00 -l ncpus=12 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/average_snapshot_matrices.jl")
```

Environment variables: PARENT_MODEL, VELOCITY_SOURCE, W_FORMULATION,
  ADVECTION_SCHEME, TIMESTEPPER (same as create_matrix.jl)
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using SparseArrays
using LinearAlgebra
using Statistics
using JLD2
using Printf
using TOML

using Oceananigans.Units: second, seconds, day, days
year = years = 365.25days
month = months = year / 12

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

cfg_file = "LocalPreferences.toml"
cfg = isfile(cfg_file) ? TOML.parsefile(cfg_file) : Dict("models" => Dict(), "defaults" => Dict())

parentmodel = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "PARENT_MODEL")
    ENV["PARENT_MODEL"]
else
    get(get(cfg, "defaults", Dict()), "parentmodel", "ACCESS-OM2-1")
end

profile = get(get(cfg, "models", Dict()), parentmodel, nothing)
if profile === nothing
    @warn "Profile for $parentmodel not found in $cfg_file; using sensible defaults"
    outputdir = normpath(joinpath(@__DIR__, "..", "outputs", parentmodel))
else
    outputdir = profile["outputdir"]
end

(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
model_config = "$(VELOCITY_SOURCE)_$(W_FORMULATION)_$(ADVECTION_SCHEME)_$(TIMESTEPPER)"

matrices_dir = joinpath(outputdir, "TM", model_config)
snapshot_matrices_dir = joinpath(matrices_dir, "snapshots")

@info "Run configuration"
@info "- PARENT_MODEL     = $parentmodel"
@info "- model_config     = $model_config"
@info "- matrices_dir     = $matrices_dir"
@info "- snapshot_dir     = $snapshot_matrices_dir"
flush(stdout); flush(stderr)

################################################################################
# Load snapshot matrices
################################################################################

@info "Loading snapshot matrices from $snapshot_matrices_dir"
flush(stdout); flush(stderr)

# Find all snapshot files
snapshot_files = sort(
    [
        f for f in readdir(snapshot_matrices_dir; join = true)
            if startswith(basename(f), "M_snapshot_") && endswith(f, ".jld2")
    ]
)

n_snapshots = length(snapshot_files)
@info "Found $n_snapshots snapshot matrices"
n_snapshots > 0 || error("No snapshot matrices found in $snapshot_matrices_dir")
flush(stdout); flush(stderr)

# Load all matrices and times
matrices = Vector{SparseMatrixCSC{Float64, Int}}(undef, n_snapshots)
snap_times = Vector{Float64}(undef, n_snapshots)

for (i, f) in enumerate(snapshot_files)
    data = load(f)
    matrices[i] = data["M"]
    snap_times[i] = data["t"]
    @info "  Loaded snapshot $(@sprintf("%02d", i)): t = $(@sprintf("%.4f", snap_times[i] / year)) yr, nnz = $(nnz(matrices[i]))"
end
flush(stdout); flush(stderr)

################################################################################
# Classify snapshots: start-of-month vs mid-month
################################################################################

@info "Classifying snapshots into start-of-month and mid-month"
flush(stdout); flush(stderr)

# Snapshots are at half-month intervals. Start-of-month snapshots are at
# t ≈ k * month for k = 0, 1, ..., 11. Mid-month snapshots are at
# t ≈ (k + 0.5) * month.
# We classify by checking whether t/month is closer to an integer or half-integer.

start_of_month_idx = Int[]
mid_month_idx = Int[]

for (i, t) in enumerate(snap_times)
    t_months = t / month
    # Distance to nearest integer vs nearest half-integer
    dist_to_int = abs(t_months - round(t_months))
    dist_to_half = abs(t_months - (floor(t_months) + 0.5))
    if dist_to_int < dist_to_half
        push!(start_of_month_idx, i)
    else
        push!(mid_month_idx, i)
    end
end

@info "Start-of-month snapshots ($(length(start_of_month_idx))):"
for i in start_of_month_idx
    @info "  snapshot $(@sprintf("%02d", i)): t = $(@sprintf("%.4f", snap_times[i] / year)) yr"
end
@info "Mid-month snapshots ($(length(mid_month_idx))):"
for i in mid_month_idx
    @info "  snapshot $(@sprintf("%02d", i)): t = $(@sprintf("%.4f", snap_times[i] / year)) yr"
end
flush(stdout); flush(stderr)

################################################################################
# Compute averaged matrices
################################################################################

@info "Computing averaged matrices"
flush(stdout); flush(stderr)

function average_matrices(mats)
    M_avg = copy(mats[1])
    for i in 2:length(mats)
        M_avg .+= mats[i]
    end
    M_avg ./= length(mats)
    return M_avg
end

M_avg12a = average_matrices(matrices[start_of_month_idx])
@info "M_avg12a: averaged $(length(start_of_month_idx)) start-of-month matrices, nnz=$(nnz(M_avg12a))"

M_avg12b = average_matrices(matrices[mid_month_idx])
@info "M_avg12b: averaged $(length(mid_month_idx)) mid-month matrices, nnz=$(nnz(M_avg12b))"

M_avg24 = average_matrices(matrices)
@info "M_avg24: averaged $n_snapshots matrices, nnz=$(nnz(M_avg24))"
flush(stdout); flush(stderr)

################################################################################
# Load constant-field matrix for comparison
################################################################################

M_constant_file = joinpath(matrices_dir, "M.jld2")
if isfile(M_constant_file)
    @info "Loading constant-field matrix from $M_constant_file"
    M_constant = load(M_constant_file, "M")
    @info "M_constant: nnz=$(nnz(M_constant))"
    has_constant = true
else
    @warn "Constant-field matrix not found at $M_constant_file — skipping comparison"
    has_constant = false
end
flush(stdout); flush(stderr)

################################################################################
# Compare matrices
################################################################################

@info "="^72
@info "Matrix comparisons"
@info "="^72
flush(stdout); flush(stderr)

function compare_matrices(M1, M2, label1, label2)
    diff = M1 - M2
    nz_diff = nonzeros(diff)
    max_abs_diff = length(nz_diff) > 0 ? maximum(abs, nz_diff) : 0.0
    frob_diff = norm(nz_diff)
    frob_M1 = norm(nonzeros(M1))
    rel_diff = frob_M1 > 0 ? frob_diff / frob_M1 : NaN
    @info "$label1 vs $label2:" max_abs_diff frob_diff rel_diff nnz_diff = nnz(diff)
    flush(stdout)
    return flush(stderr)
end

# Compare averaged matrices to each other
compare_matrices(M_avg12a, M_avg12b, "M_avg12a", "M_avg12b")
compare_matrices(M_avg12a, M_avg24, "M_avg12a", "M_avg24")
compare_matrices(M_avg12b, M_avg24, "M_avg12b", "M_avg24")

# Compare to constant-field matrix
if has_constant
    compare_matrices(M_avg12a, M_constant, "M_avg12a", "M_constant")
    compare_matrices(M_avg12b, M_constant, "M_avg12b", "M_constant")
    compare_matrices(M_avg24, M_constant, "M_avg24", "M_constant")
end

################################################################################
# Save averaged matrices
################################################################################

@info "="^72
@info "Saving averaged matrices"
@info "="^72
flush(stdout); flush(stderr)

for (label, M) in [("avg12a", M_avg12a), ("avg12b", M_avg12b), ("avg24", M_avg24)]
    out_dir = joinpath(matrices_dir, label)
    mkpath(out_dir)
    outfile = joinpath(out_dir, "M.jld2")
    jldsave(outfile; M)
    @info "Saved $outfile (nnz=$(nnz(M)))"
end
flush(stdout); flush(stderr)

@info "average_snapshot_matrices.jl complete"
@info "Run solve_matrix_age.jl with MATRIX_SUBDIR=avg24 (etc.) to solve age from each"
flush(stdout); flush(stderr)
