"""
Test whether the forward map Φ(x) is linear in x by comparing the Jacobian
matrix M(x) = dG/dx evaluated at different linearization points.

If Φ is linear, M(x) should be identical regardless of x. Any differences
indicate nonlinearity introduced by Oceananigans (e.g., immersed boundary
masking, halo handling, or advection scheme internals).

Usage — interactive:
```
qsub -I -P y99 -l mem=190GB -q normal -l walltime=04:00:00 -l ncpus=48 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 \\
     -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/test_linearity.jl")
```

Environment variables: same as create_matrix.jl (PARENT_MODEL, VELOCITY_SOURCE, etc.)
"""

include("matrix_setup.jl")

using OrderedCollections: OrderedDict

################################################################################
# Load TMage (transport-matrix steady-state age) for use as linearization point
################################################################################

@info "Loading TMage for linearization point"
flush(stdout); flush(stderr)

TMage_vec = zeros(Nidx)
candidates = [
    "steady_age_full_$(solver)_$(mp).jld2"
        for mp in ("raw", "dropzeros", "symfill", "symdrop")
        for solver in ("ParU", "UMFPACK", "Pardiso")
]
loaded_TMage = false
for candidate in candidates
    fpath = joinpath(matrices_dir, candidate)
    if isfile(fpath)
        @info "Loading TMage from $fpath"
        flush(stdout); flush(stderr)
        age_data = load(fpath, "age")
        # Matrix age files store age in years → convert to seconds
        TMage_vec .= view(age_data, idx) .* year
        @info "TMage loaded" max_years = maximum(abs, TMage_vec) / year mean_years = mean(TMage_vec) / year
        global loaded_TMage = true
        break
    end
end
if !loaded_TMage
    @warn "No TMage file found in $matrices_dir — using 1000yr fill instead"
    TMage_vec .= 1000year
end

################################################################################
# Cell volumes for weighted statistics
################################################################################

grid_cpu = on_architecture(CPU(), grid)
v1D = interior(compute_volume(grid_cpu))[idx]
inv_sumv = 1 / sum(v1D)

################################################################################
# Define linearization points
################################################################################

x_values = OrderedDict{String, Vector{Float64}}(
    "zeros" => zeros(Nidx),
    "ones" => ones(Nidx),
    "1yr" => fill(Float64(year), Nidx),
    "1000yr" => fill(1000.0 * year, Nidx),
    "TMage" => copy(TMage_vec),
)

@info "Linearization points: $(collect(keys(x_values)))"
flush(stdout); flush(stderr)

################################################################################
# Compute Jacobian at each linearization point
################################################################################

M_dict = OrderedDict{String, SparseMatrixCSC{Float64, Int}}()

for (label, x) in x_values
    @info "Computing Jacobian at x = $label"
    flush(stdout); flush(stderr)

    ADcvec .= x
    @time "Jacobian at $label" jacobian!(
        mytendency!, GADcvec, jac_buffer, jac_prep,
        sparse_forward_backend, ADcvec,
        Cache(ADc_buf), Cache(GADc_buf),
    )
    M_dict[label] = copy(jac_buffer)
    @info "  M($label): nnz=$(nnz(M_dict[label])), max|M|=$(maximum(abs, nonzeros(M_dict[label])))"
    flush(stdout); flush(stderr)
end

################################################################################
# Compare all pairs of Jacobians
################################################################################

@info "="^72
@info "Jacobian comparison (pairwise differences)"
@info "="^72
flush(stdout); flush(stderr)

labels = collect(keys(M_dict))
for i in 1:length(labels)
    for j in (i + 1):length(labels)
        l1, l2 = labels[i], labels[j]
        M1, M2 = M_dict[l1], M_dict[l2]
        diff = M1 - M2
        max_abs_diff = length(nonzeros(diff)) > 0 ? maximum(abs, nonzeros(diff)) : 0.0
        frob_diff = norm(nonzeros(diff))
        frob_M1 = norm(nonzeros(M1))
        rel_diff = frob_M1 > 0 ? frob_diff / frob_M1 : NaN
        @info "M($l1) vs M($l2):" max_abs_diff frob_diff rel_diff nnz_diff = nnz(diff)
        flush(stdout); flush(stderr)
    end
end

################################################################################
# Apply each M(x) to test vectors and compare results
################################################################################

@info "="^72
@info "Applying M(x) to test vectors"
@info "="^72
flush(stdout); flush(stderr)

# Compute tendency at TMage for use as a test vector
tendency_TMage = similar(TMage_vec)
mytendency!(tendency_TMage, TMage_vec, ADc_buf, GADc_buf)

test_vectors = OrderedDict{String, Vector{Float64}}(
    "ones" => ones(Nidx),
    "1000yr" => fill(1000.0 * year, Nidx),
    "TMage" => copy(TMage_vec),
    "tendency_TMage" => copy(tendency_TMage),
)

# Use first Jacobian (M_ref) as reference
M_ref_label = first(labels)
M_ref = M_dict[M_ref_label]

for (v_label, v) in test_vectors
    @info "--- Test vector: $v_label ---"
    flush(stdout); flush(stderr)

    ref_result = M_ref * v
    ref_norm = norm(ref_result)
    vol_rms_ref = sqrt(dot(v1D, ref_result .^ 2) * inv_sumv)

    for (m_label, M) in M_dict
        result = M * v
        diff = result .- ref_result
        max_abs_diff = maximum(abs, diff)
        rms_diff = sqrt(dot(v1D, diff .^ 2) * inv_sumv)
        rel_diff = ref_norm > 0 ? norm(diff) / ref_norm : NaN
        @info "  M($m_label) * $v_label:" max_abs_diff rms_diff rel_diff vol_rms_result = sqrt(dot(v1D, result .^ 2) * inv_sumv)
        flush(stdout); flush(stderr)
    end
end

################################################################################
# Save all matrices
################################################################################

@info "Saving all Jacobians"
flush(stdout); flush(stderr)

linearity_dir = joinpath(matrices_dir, "linearity_test")
mkpath(linearity_dir)

for (label, M) in M_dict
    outfile = joinpath(linearity_dir, "M_at_$(label).jld2")
    jldsave(outfile; M)
    @info "  Saved $outfile"
end

@info "test_linearity.jl complete"
flush(stdout); flush(stderr)
