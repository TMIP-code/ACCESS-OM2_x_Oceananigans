"""
Average snapshot transport matrices and compare to the constant-field matrix.

Loads the 24 snapshot matrices produced by `create_snapshot_matrices.jl`,
computes three averages (avg12a, avg12b, avg24), compares them pairwise and
against the constant-field matrix M, and saves each average as `M.jld2` in
subdirectories of the TM output directory.

Matrices are loaded one at a time for memory efficiency. Sparsity pattern
identity is asserted (colptr, rowval) and accumulation operates only on
.nzval to prevent dropzeros from breaking the pattern.

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
using JLD2
using Printf
using TOML

import Pardiso

using Oceananigans.Units: second, seconds, day, days
year = years = 365.25days
month = months = year / 12

include("shared_functions.jl")

################################################################################
# Configuration
################################################################################

(; parentmodel, outputdir) = load_project_config()

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
# Find snapshot files and load times only
################################################################################

@info "Finding snapshot matrices in $snapshot_matrices_dir"
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

# Load only times (not full matrices) to save memory
snap_times = Vector{Float64}(undef, n_snapshots)
for (i, f) in enumerate(snapshot_files)
    snap_times[i] = load(f, "t")
    @info "  snapshot $(@sprintf("%02d", i)): t = $(@sprintf("%.4f", snap_times[i] / year)) yr"
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
# Sparsity-safe averaging function
################################################################################

"""
    average_snapshot_matrices(snapshot_files, indices, label) -> SparseMatrixCSC

Average the snapshot matrices at `indices` from `snapshot_files`, loading one at
a time. Asserts all matrices share the same sparsity pattern (colptr, rowval).
Operates only on .nzval to avoid dropzeros.
"""
function average_snapshot_matrices(snapshot_files, indices, label)
    # Load the first matrix to establish the sparsity pattern
    M_avg = load(snapshot_files[indices[1]], "M")
    @info "  [$label] Loaded reference snapshot $(indices[1]): nnz=$(nnz(M_avg))"
    flush(stdout); flush(stderr)

    for k in 2:length(indices)
        i = indices[k]
        Mi = load(snapshot_files[i], "M")
        # Assert identical sparsity pattern
        Mi.colptr == M_avg.colptr || error(
            "[$label] Sparsity pattern mismatch at snapshot $i: colptr differs from reference"
        )
        Mi.rowval == M_avg.rowval || error(
            "[$label] Sparsity pattern mismatch at snapshot $i: rowval differs from reference"
        )
        M_avg.nzval .+= Mi.nzval
        @info "  [$label] Accumulated snapshot $i"
        flush(stdout); flush(stderr)
    end

    M_avg.nzval ./= length(indices)
    @info "  [$label] Averaged $(length(indices)) matrices, nnz=$(nnz(M_avg))"
    flush(stdout); flush(stderr)
    return M_avg
end

################################################################################
# Load constant-field matrix for comparison (optional)
################################################################################

M_constant_file = joinpath(matrices_dir, "const", "M.jld2")
has_constant = isfile(M_constant_file)
if has_constant
    @info "Loading constant-field matrix from $M_constant_file"
    flush(stdout); flush(stderr)
    M_constant = load(M_constant_file, "M")
    @info "M_constant: nnz=$(nnz(M_constant))"
else
    @warn "Constant-field matrix not found at $M_constant_file — skipping comparison"
end
flush(stdout); flush(stderr)

################################################################################
# Compute, check, compare, and save each average
################################################################################

@info "="^72
@info "Computing and saving averaged matrices"
@info "="^72
flush(stdout); flush(stderr)

for (label, indices) in [
        ("avg24", collect(1:n_snapshots)),
        ("avg12a", start_of_month_idx),
        ("avg12b", mid_month_idx),
    ]
    out_dir = joinpath(matrices_dir, label)
    mkpath(out_dir)
    outfile = joinpath(out_dir, "M.jld2")

    # Remove existing file so a failed run doesn't leave stale data
    isfile(outfile) && rm(outfile)

    @info "Computing $label average..."
    flush(stdout); flush(stderr)
    M = average_snapshot_matrices(snapshot_files, indices, label)

    # Assert structural symmetry
    if !Pardiso.isstructurallysymmetric(M)
        error("[$label] Averaged matrix is NOT structurally symmetric (nnz=$(nnz(M)))")
    end
    @info "  [$label] Structural symmetry check passed"
    flush(stdout); flush(stderr)

    # Compare to constant-field matrix if available and same sparsity
    if has_constant && M_constant.colptr == M.colptr && M_constant.rowval == M.rowval
        diff_nzval = M.nzval .- M_constant.nzval
        max_abs_diff = maximum(abs, diff_nzval)
        frob_rel = norm(diff_nzval) / norm(M_constant.nzval)
        @info "  [$label] vs M_constant:" max_abs_diff frob_rel
        flush(stdout); flush(stderr)
    elseif has_constant
        @warn "  [$label] Sparsity pattern differs from M_constant — skipping comparison"
        flush(stdout); flush(stderr)
    end

    jldsave(outfile; M)
    @info "Saved $outfile (nnz=$(nnz(M)))"
    flush(stdout); flush(stderr)
end

@info "average_snapshot_matrices.jl complete"
@info "Run solve_matrix_age.jl with TM_SOURCE=avg24 (etc.) to solve age from each"
flush(stdout); flush(stderr)
