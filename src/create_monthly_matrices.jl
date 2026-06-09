"""
Build an averaged transport matrix from the 12 monthly velocity fields, reusing
the time-averaged (const) build pipeline.

This is a lighter-weight alternative to `create_snapshot_matrices.jl`. Instead of
reading velocity snapshots saved by a 1-year forward run, it reads the
preprocessed monthly velocity FieldTimeSeries directly from
`preprocessed_inputs/.../{TW}/monthly/` and computes one Jacobian per month using
*exactly* the same model / closure / autodiff machinery as the const build
(`matrix_setup.jl`), then averages the 12 matrices into `TM/{MC}/avg/M.jld2`.

Trade-off vs `create_snapshot_matrices.jl`: this inherits the const build's
assumption of zero sea-surface tracer tendency (∂η/∂t is not folded into the
prescribed w), so it is a coarser approximation of the seasonally-varying
transport. In exchange it builds much faster and — crucially — needs no prior
1-year simulation, so the averaged-matrix NK preconditioner can be produced from
preprocessing alone. The sparsity pattern and graph colouring are computed once
(in `matrix_setup.jl`, from the yearly fields) and reused for every month, since
the pattern depends on the grid/advection stencil, not the velocity values.

Outputs:
  TM/{MODEL_CONFIG}/monthly/M_month_NN.jld2   (per-month matrices)
  TM/{MODEL_CONFIG}/avg/M.jld2                (the averaged matrix; same path as
                                               the snapshot build, so downstream
                                               TM_SOURCE=avg consumers are unchanged)

Environment variables: same as `create_matrix.jl` (PARENT_MODEL, VELOCITY_SOURCE,
W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER, ...).
"""

# Oceananigans must be loaded before shared_functions.jl (config.jl uses
# Oceananigans.Grids/Architectures); matrix_setup.jl pulls in the rest.
using Oceananigans
using Oceananigans.Architectures: CPU
include("matrix_setup.jl")

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _update_zstar_scaling!, surface_kernel_parameters
using Oceananigans.Utils: launch!
using Oceananigans.OutputReaders: InMemory

################################################################################
# Load the 12 monthly velocity / η FieldTimeSeries
#
# These are the same preprocessed monthly fields the forward simulation uses
# (see setup_model.jl); we read the time-mean-free monthly snapshots and prescribe
# them one month at a time. The yearly fields loaded by matrix_setup.jl above were
# only needed for sparsity detection / Jacobian preparation.
################################################################################

vs_prefix = VELOCITY_SOURCE == "totaltransport" ? "total_transport" : "mass_transport"
u_monthly_file = joinpath(monthly_dir, "u_from_$(vs_prefix)_monthly.jld2")
v_monthly_file = joinpath(monthly_dir, "v_from_$(vs_prefix)_monthly.jld2")
w_monthly_file = joinpath(monthly_dir, "w_from_$(vs_prefix)_monthly.jld2")
η_monthly_file = joinpath(monthly_dir, "eta_monthly.jld2")

@info "Loading monthly velocity FieldTimeSeries"
@info "- $u_monthly_file"
@info "- $v_monthly_file"
W_FORMULATION == "wprescribed" && @info "- $w_monthly_file"
@info "- $η_monthly_file"
flush(stdout); flush(stderr)

u_ts = FieldTimeSeries(u_monthly_file, "u"; architecture = arch, grid, backend = InMemory())
v_ts = FieldTimeSeries(v_monthly_file, "v"; architecture = arch, grid, backend = InMemory())
η_ts = FieldTimeSeries(η_monthly_file, "η"; architecture = arch, grid, backend = InMemory())
if W_FORMULATION == "wprescribed"
    w_ts = FieldTimeSeries(w_monthly_file, "w"; architecture = arch, grid, backend = InMemory())
end

n_months = length(u_ts.times)
(n_months == length(v_ts.times) == length(η_ts.times)) ||
    error("Monthly FTS length mismatch: u=$(length(u_ts.times)), v=$(length(v_ts.times)), η=$(length(η_ts.times))")
@info "Loaded $n_months monthly velocity snapshots"
flush(stdout); flush(stderr)

################################################################################
# Per-month Jacobians (reusing const machinery) + running average
################################################################################

# Saving the 12 per-month matrices is handy for inspection at low resolution, but
# at OM2-01 each is ~39 GB (468 GB total), so allow skipping via env. The running
# average is always kept; only the per-month files are gated.
save_intermediate = lowercase(get(ENV, "SAVE_INTERMEDIATE_MATRICES", "yes")) == "yes"
monthly_matrices_dir = joinpath(matrices_dir, "monthly")
save_intermediate && mkpath(monthly_matrices_dir)
save_intermediate || @info "SAVE_INTERMEDIATE_MATRICES=no — keeping only the averaged matrix, not the 12 per-month files"

nzval_avg = zeros(Float64, nnz(jac_buffer))

@info "="^72
@info "Computing Jacobians at $n_months monthly velocity fields"
@info "="^72
flush(stdout); flush(stderr)

for m in 1:n_months
    mlabel = @sprintf("%02d", m)
    @info "Month $mlabel/$n_months: prescribing this month's velocities"
    flush(stdout); flush(stderr)

    # Mutate the prescribed velocity/η fields in place; the jacobian_model holds
    # references to these, so the next jacobian! call sees the updated month.
    set!(u_constant, interior(u_ts[m])); fill_halo_regions!(u_constant)
    set!(v_constant, interior(v_ts[m])); fill_halo_regions!(v_constant)
    if W_FORMULATION == "wprescribed"
        set!(w_constant, interior(w_ts[m])); fill_halo_regions!(w_constant)
    end
    set!(η_constant, interior(η_ts[m])); fill_halo_regions!(η_constant)

    # zstar scaling from this month's η (mirrors the const build's single call)
    launch!(CPU(), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η_constant, grid)

    @time "Jacobian at month $mlabel" jacobian!(
        mytendency!, GADcvec, jac_buffer, jac_prep,
        sparse_forward_backend, ADcvec,
        Cache(ADc_buf), Cache(GADc_buf),
    )
    @info "  M(month $mlabel): nnz=$(nnz(jac_buffer)), max|M|=$(maximum(abs, nonzeros(jac_buffer)))"

    # Accumulate into the average (all months share the one sparsity pattern)
    nzval_avg .+= jac_buffer.nzval

    if save_intermediate
        outfile = joinpath(monthly_matrices_dir, "M_month_$(mlabel).jld2")
        jldsave(outfile; M = jac_buffer, month = m)
        @info "  Saved $outfile"
    end
    flush(stdout); flush(stderr)
end

################################################################################
# Finalise and save averaged matrix
################################################################################

@info "="^72
@info "Finalising averaged matrix"
@info "="^72
flush(stdout); flush(stderr)

nzval_avg ./= n_months
M_avg = SparseMatrixCSC(
    size(jac_buffer, 1), size(jac_buffer, 2),
    jac_buffer.colptr, jac_buffer.rowval, copy(nzval_avg),
)
@info "M_avg: $(size(M_avg, 1))×$(size(M_avg, 2)), nnz=$(nnz(M_avg))"

avg_dir = joinpath(matrices_dir, "avg")
mkpath(avg_dir)
outfile = joinpath(avg_dir, "M.jld2")
isfile(outfile) && rm(outfile)
jldsave(outfile; M = M_avg)
@info "Saved $outfile (nnz=$(nnz(M_avg)))"
flush(stdout); flush(stderr)

@info "create_monthly_matrices.jl complete — averaged $n_months monthly matrices into $avg_dir"
flush(stdout); flush(stderr)
