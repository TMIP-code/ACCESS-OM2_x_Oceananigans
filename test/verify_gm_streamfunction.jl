"""
Verify the streamfunction_to_perlayer! sign convention.

This script:
1. Loads ty_trans_gm (GM streamfunction on z-faces) for one month
2. Applies streamfunction_to_perlayer! to get per-layer transport
3. Checks reconstruction: cumsum from bottom should recover the original streamfunction
4. Compares MOC computation both ways:
   - Direct: zonal sum of raw streamfunction (as in plot_MOC.jl compute_moc_gm)
   - Via per-layer: zonal sum + vertical cumsum of per-layer transport (as in compute_moc)
   Both should give the same MOC.

Usage:
    julia --project test/verify_gm_streamfunction.jl
"""

using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FPivotZipperBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using JLD2
using YAXArrays
using DimensionalData
using NCDatasets
using NetCDF
using Statistics: mean

include("../src/shared_functions.jl")

const ρ₀ = 1035.0

(; parentmodel, experiment_dir, monthly_dir, yearly_dir) = load_project_config()

# ── Load grid ────────────────────────────────────────────────────────────
@info "Loading grid"
grid_file = joinpath(experiment_dir, "grid.jld2")
grid = load_tripolar_grid(grid_file, CPU())
Nx, Ny, Nz = size(grid)
@info "Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"

# ── Load GM transport for month 1 ────────────────────────────────────────
@info "Loading ty_trans_gm for month 1"
ty_gm_ds = open_dataset(joinpath(monthly_dir, "ty_trans_gm_monthly.nc"))
ty_gm_data = readcubedata(ty_gm_ds.ty_trans_gm[month = At(1)]).data
map!(x -> isnan(x) ? zero(x) : x, ty_gm_data, ty_gm_data)

# Place on grid
north_t = FPivotZipperBoundaryCondition(-1)
tx_bcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); north = north_t)
ty_bcs = FieldBoundaryConditions(grid, (Center(), Face(), Center()); north = north_t)
tx_gm = XFaceField(grid; boundary_conditions = tx_bcs)
ty_gm = YFaceField(grid; boundary_conditions = ty_bcs)

# Dummy tx_data (zeros)
tx_gm_data = zeros(Nx, Ny - 1, Nz)
fill_Cgrid_transport_from_MOM_output!(tx_gm, ty_gm, grid, tx_gm_data, ty_gm_data)

# ── Save copy of streamfunction before conversion ────────────────────────
ψ_orig = copy(Array(interior(ty_gm)))  # (Nx', Ny_f, Nz')
Nx′, Ny_f, Nz′ = size(ψ_orig)
@info "Interior shape: ($Nx′, $Ny_f, $Nz′)"

# Check that bottom face values are near zero (ψ_gm ≈ 0 at bottom)
bottom_vals = ψ_orig[:, :, 1]
bottom_nonzero = filter(!iszero, vec(bottom_vals))
if !isempty(bottom_nonzero)
    @info "Bottom face (k=1) stats: min=$(minimum(bottom_nonzero)), max=$(maximum(bottom_nonzero)), mean=$(mean(bottom_nonzero))"
else
    @info "Bottom face (k=1): all zeros"
end

# Check surface values
surface_vals = ψ_orig[:, :, Nz′]
surface_nonzero = filter(!iszero, vec(surface_vals))
if !isempty(surface_nonzero)
    @info "Surface (k=Nz=$Nz′) stats: min=$(minimum(surface_nonzero)), max=$(maximum(surface_nonzero)), mean=$(mean(surface_nonzero))"
else
    @info "Surface (k=Nz=$Nz′): all zeros"
end

# ── Apply conversion ────────────────────────────────────────────────────
@info "Applying streamfunction_to_perlayer!"
streamfunction_to_perlayer!(ty_gm, grid)
perlayer = copy(Array(interior(ty_gm)))

# ── Test 1: Reconstruction ──────────────────────────────────────────────
# If T[k] = ψ[k] - ψ[k+1], then cumsum(T[1:k]) = ψ[1] - ψ[k+1]
# Since ψ[1] ≈ 0 (bottom), cumsum(T[1:k]) ≈ -ψ[k+1]
# So ψ[k] ≈ -cumsum(T[1:k-1]) for k ≥ 2, and ψ[1] ≈ 0

# Alternatively: reverse cumsum from surface should recover ψ
# T[k] = ψ[k] - ψ[k+1], so ψ[k] = T[k] + ψ[k+1] = T[k] + T[k+1] + ... + T[Nz]
# i.e. ψ[k] = sum(T[k:Nz]) = reverse cumsum from surface
reconstructed = similar(ψ_orig)
for k in Nz′:-1:1
    if k == Nz′
        reconstructed[:, :, k] = perlayer[:, :, k]
    else
        reconstructed[:, :, k] = perlayer[:, :, k] .+ reconstructed[:, :, k + 1]
    end
end

# Compare
diff = abs.(reconstructed .- ψ_orig)
max_diff = maximum(diff)
mean_diff = mean(diff)
max_ψ = maximum(abs.(ψ_orig))
@info "Reconstruction test: max|ψ_reconstructed - ψ_orig| = $max_diff (relative to max|ψ| = $max_ψ)"
if max_diff / max(max_ψ, 1.0e-10) < 1.0e-10
    @info "✓ Reconstruction PASSED — streamfunction_to_perlayer! is self-consistent"
else
    @warn "✗ Reconstruction FAILED — sign convention may be wrong"
end

# ── Test 2: MOC comparison ──────────────────────────────────────────────
# Method A: Direct streamfunction (as compute_moc_gm does)
# Zonal sum of raw ψ_orig → ψ_gm
ψ_direct = dropdims(sum(ψ_orig; dims = 1); dims = 1) ./ (ρ₀ * 1.0e6)  # (Ny_f, Nz') in Sv

# Method B: Per-layer → zonal sum → cumsum (as compute_moc does for resolved)
perlayer_zonal = dropdims(sum(perlayer; dims = 1); dims = 1)  # (Ny_f, Nz')
cs = cumsum(perlayer_zonal; dims = 2)
ψ_from_perlayer = similar(cs)
ψ_from_perlayer[:, 1] .= 0.0
ψ_from_perlayer[:, 2:end] .= .-cs[:, 1:(end - 1)]
ψ_from_perlayer ./= (ρ₀ * 1.0e6)

# Compare
moc_diff = abs.(ψ_from_perlayer .- ψ_direct)
max_moc_diff = maximum(moc_diff)
max_moc = maximum(abs.(ψ_direct))
@info "MOC comparison: max|ψ_perlayer_moc - ψ_direct_moc| = $(round(max_moc_diff; sigdigits = 3)) Sv (max|ψ| = $(round(max_moc; sigdigits = 3)) Sv)"
if max_moc_diff / max(max_moc, 1.0e-10) < 0.01
    @info "✓ MOC comparison PASSED — per-layer transport produces correct GM MOC"
else
    @warn "✗ MOC comparison FAILED — sign convention likely wrong"
    @info "Trying negated convention: T[k] = ψ[k+1] - ψ[k]..."
    perlayer_neg = -perlayer
    perlayer_neg_zonal = dropdims(sum(perlayer_neg; dims = 1); dims = 1)
    cs_neg = cumsum(perlayer_neg_zonal; dims = 2)
    ψ_neg = similar(cs_neg)
    ψ_neg[:, 1] .= 0.0
    ψ_neg[:, 2:end] .= .-cs_neg[:, 1:(end - 1)]
    ψ_neg ./= (ρ₀ * 1.0e6)
    neg_diff = maximum(abs.(ψ_neg .- ψ_direct))
    @info "Negated MOC diff: $(round(neg_diff; sigdigits = 3)) Sv"
    if neg_diff < max_moc_diff
        @warn "→ Negated convention is BETTER — kernel sign should be flipped!"
    end
end

# ── Print sample MOC values for sanity check ────────────────────────────
j_60S = argmin(abs.(dropdims(maximum(Array(grid.underlying_grid.φᶜᶠᵃ[1:Nx′, 1:Ny_f]); dims = 1); dims = 1) .+ 60))
@info "Sample GM MOC at ~60°S (j=$j_60S):"
@info "  Direct:     ψ_gm = $(round.(ψ_direct[j_60S, :]; sigdigits = 3))"
@info "  Per-layer:  ψ_pl = $(round.(ψ_from_perlayer[j_60S, :]; sigdigits = 3))"

println("\nDone!")
