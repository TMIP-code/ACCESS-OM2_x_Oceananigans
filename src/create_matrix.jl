"""
Build the transport matrix (Jacobian of the tracer tendency) from time-averaged
(constant) velocity and free-surface fields produced by `create_velocities.jl`.

Unlike `run_ACCESS-OM2.jl`, no simulation is run: the model is initialised
with constant prescribed fields and the Jacobian is computed in a single pass.
The matrix build always uses the CPU (sparsity detection and coloring require it).

Usage — interactive:
```
qsub -I -P y99 -l mem=192GB -q normal -l walltime=04:00:00 -l ncpus=48 \\
     -l storage=gdata/xp65+gdata/ik11+scratch/y99+gdata/y99 \\
     -o logs/PBS/ -j oe
cd /home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
julia --project
include("src/create_matrix.jl")
```

Environment variables:
  PARENT_MODEL      – model resolution tag  (default: ACCESS-OM2-1)
  VELOCITY_SOURCE   – cgridtransports | bgridvelocities  (default: cgridtransports)
  W_FORMULATION     – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME  – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER       – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  (Age solving has been factored out into solve_matrix_age.jl)
"""

include("matrix_setup.jl")

################################################################################
# Compute Jacobian
#
# Single computation — constant (time-averaged) fields mean no monthly loop.
################################################################################

@info "Computing Jacobian (single pass — time-averaged constant fields)"
flush(stdout); flush(stderr)
@time "Compute Jacobian" jacobian!(
    mytendency!, GADcvec, jac_buffer, jac_prep,
    sparse_forward_backend, ADcvec,
    Cache(ADc_buf), Cache(GADc_buf),
)

M = copy(jac_buffer)  # units: 1/s
@info "Jacobian M ($(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M)), density=$(@sprintf("%.2e", nnz(M) / length(M))))"
flush(stdout); flush(stderr)
@info "Sparsity pattern of M:"
display(M)

@info "Saving Jacobian to $(const_dir)"
flush(stdout); flush(stderr)
jldsave(joinpath(const_dir, "M.jld2"); M)

fig = Figure()
ax = Axis(fig[1, 1])
plt = spy!(
    0.5 .. size(M, 1) + 0.5,
    0.5 .. size(M, 2) + 0.5,
    M;
    colormap = :coolwarm,
    colorrange = maximum(abs.(M)) .* (-1, 1),
    markersize = size(M, 2) / 1000,
)
ylims!(ax, size(M, 2) + 0.5, 0.5)
Colorbar(fig[1, 2], plt)
save(joinpath(const_plots_dir, "M_spy.png"), fig)


@info "create_matrix.jl complete. Outputs in $(const_dir)"
@info "(Run solve_matrix_age.jl to solve for steady-state age using the saved matrix)"
flush(stdout); flush(stderr)
