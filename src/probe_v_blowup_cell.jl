"""
Definitive cross-check of the v_from_mass_transport blow-up: on ONE loaded grid,
read the saved v FTS (month 10), find its argmax|v| cell, and at that SAME index
print v, the reference AyCFC (σ=1), and the raw source ty_trans that the C-grid
copy pulled from — so we can see, self-consistently, which factor is anomalous.

The prep_velocities copy is  field[mod1(i,Nx), j+1, Nz+1-k] = ty_data[i,j,k]
so the C-grid v-cell (I,J,K) sources ty_data[I, J-1, Nz+1-K].

Env: PARENT_MODEL, EXPERIMENT, TIME_WINDOW, optional PROBE_MONTH (default 10).
CPU only.
"""

@info "Loading packages"
flush(stdout); flush(stderr)

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.AbstractOperations: grid_metric_operation, Ay
using Oceananigans.Grids: znodes
using NCDatasets
using Printf

include("shared_functions.jl")

(; parentmodel, experiment, time_window, experiment_dir, monthly_dir) = load_project_config()
const ρ₀ = 1035.0
month = parse(Int, get(ENV, "PROBE_MONTH", "10"))

grid = load_tripolar_grid(joinpath(experiment_dir, "grid.jld2"), CPU())
Nx, Ny, Nz = size(grid)
@info "grid size" Nx Ny Nz

# ── 1. saved v FTS, month `month`: argmax|v| ──────────────────────────────
vpath = joinpath(monthly_dir, "v_from_mass_transport_monthly.jld2")
vfts = FieldTimeSeries(vpath, "v"; backend = OnDisk())
vi = Array(interior(vfts[month]))
absv = map(x -> isfinite(x) ? abs(x) : -Inf, vi)
mx, ci = findmax(absv)
I, J, K = Tuple(ci)
@info @sprintf(
    "saved v FTS month %d: max|v|=%.6g @ interior (I=%d,J=%d,K=%d) of size %s",
    month, mx, I, J, K, string(size(vi))
)
@info @sprintf(
    "  v at fixed (848,1246,16) = %.6g",
    checkbounds(Bool, vi, 848, 1246, 16) ? vi[848, 1246, 16] : NaN
)

# ── 2. reference AyCFC (σ=1) at the SAME index ────────────────────────────
AyCFC = Field(grid_metric_operation((Center(), Face(), Center()), Ay, grid))
compute!(AyCFC)
Ai = Array(interior(AyCFC))
Aval = checkbounds(Bool, Ai, I, J, K) ? Ai[I, J, K] : NaN
@info @sprintf("  AyCFC(σ=1) at (%d,%d,%d) = %.6g", I, J, K, Aval)

# ── 3. raw source ty_trans[I, J-1, Nz+1-K], small hyperslab around it ─────
src_i, src_j, src_k = I, J - 1, Nz + 1 - K
@info @sprintf("  → source raw ty_trans index (i=%d, j=%d, k=%d)", src_i, src_j, src_k)
NCDataset(joinpath(monthly_dir, "ty_trans_monthly.nc")) do ds
    v = ds["ty_trans"]           # dims in Julia order: (xt_ocean, yu_ocean, st_ocean, month)
    @info "  ty_trans var size (Julia order) = $(size(v))"
    ir = clamp(src_i - 2, 1, Nx):clamp(src_i + 2, 1, Nx)
    jr = clamp(src_j - 2, 1, Ny):clamp(src_j + 2, 1, Ny)
    kr = clamp(src_k - 1, 1, Nz):clamp(src_k + 1, 1, Nz)
    block = v[ir, jr, kr, month]
    @info @sprintf(
        "  raw ty_trans at source (%d,%d,%d) = %.6g", src_i, src_j, src_k,
        v[src_i, src_j, src_k, month]
    )
    @info "  |ty| max in 5x5x3 block = $(maximum(x -> isfinite(x) ? abs(x) : -Inf, block))"
end

# ── 4. recompute v from those factors ─────────────────────────────────────
@info "  (if v≈ty/(ρ₀·AyCFC) with the above, factors are self-consistent; if not, the FTS holds something else)"
@info "probe_v_blowup_cell.jl complete"
flush(stdout); flush(stderr)
