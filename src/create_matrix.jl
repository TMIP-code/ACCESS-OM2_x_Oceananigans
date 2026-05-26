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
  VELOCITY_SOURCE   – cgridtransports | totaltransport (default: cgridtransports)
  W_FORMULATION     – wdiagnosed | wprescribed  (default: wdiagnosed)
  ADVECTION_SCHEME  – centered2 | weno3 | weno5  (default: centered2)
  TIMESTEPPER       – AB2 | SRK2 | SRK3 | SRK4 | SRK5  (default: AB2)
  (Age solving has been factored out into solve_matrix_age.jl)
"""

# Need Oceananigans loaded before shared_functions.jl (config.jl uses
# Oceananigans.Grids/Architectures) so require_env is in scope below.
using Oceananigans
using Oceananigans.Architectures: CPU
include("shared_functions.jl")

TRAF = lowercase(require_env("TRAF")) == "yes"
TRAF_TM_SOURCE = require_env("TRAF_TM_SOURCE")

if TRAF && TRAF_TM_SOURCE == "invVMtV"
    # ── Option B: algebraic synthesis  M_traf = V⁻¹ Mᵀ V  from the forward M ──
    # No autodiff, no Oceananigans heavy setup — just sparse linear algebra.
    using JLD2, SparseArrays, LinearAlgebra

    (; parentmodel, experiment_dir, outputdir) = load_project_config()
    (; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
    traf_mc = build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER)
    fwd_mc = replace(traf_mc, r"_traf$" => "")
    fwd_mc == traf_mc && error(
        "TRAF=yes but model_config has no _traf suffix: $traf_mc. " *
            "Check that build_model_config and env_defaults.sh agree on the _traf rule.",
    )
    fwd_M_path = joinpath(outputdir, "TM", fwd_mc, "const", "M.jld2")
    @info "TRAF/invVMtV: loading forward M from $fwd_M_path"
    isfile(fwd_M_path) || error(
        "Forward M not found at $fwd_M_path. Build it first with the same MODEL_CONFIG " *
            "but TRAF=no (e.g. TRAF=no JOB_CHAIN=TMbuild bash scripts/driver.sh).",
    )
    M = load(fwd_M_path, "M")

    grid_file = joinpath(experiment_dir, "grid.jld2")
    grid = load_tripolar_grid(grid_file, CPU())
    (; idx, Nidx) = compute_wet_mask(grid)
    v = interior(compute_volume(grid))[idx]
    size(M, 1) == Nidx == length(v) || error(
        "Size mismatch: M is $(size(M, 1))×$(size(M, 2)), Nidx=$Nidx, length(v)=$(length(v)). " *
            "Forward M was likely built on a different grid configuration.",
    )

    invVMtV = sparse(Diagonal(v .^ -1)) * M' * sparse(Diagonal(v))
    out_dir = joinpath(outputdir, "TM", traf_mc, "const")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "invVMtV.jld2")
    jldsave(out_path; M = invVMtV)
    @info "TRAF/invVMtV: wrote $out_path  (nnz = $(nnz(invVMtV)))"
    exit()
end

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

M = jac_buffer  # units: 1/s
@info "Jacobian M ($(size(M, 1))×$(size(M, 2)), nnz=$(nnz(M)), density=$(@sprintf("%.2e", nnz(M) / length(M))))"
flush(stdout); flush(stderr)
@info "Sparsity pattern of M:"
display(M)

output_M_name = (TRAF && TRAF_TM_SOURCE == "M_traf") ? "M_traf.jld2" : "M.jld2"
@info "Saving Jacobian to $(joinpath(const_dir, output_M_name))"
flush(stdout); flush(stderr)
jldsave(joinpath(const_dir, output_M_name); M)


@info "create_matrix.jl complete. Outputs in $(const_dir)"
@info "(Run solve_matrix_age.jl to solve for steady-state age using the saved matrix)"
flush(stdout); flush(stderr)
