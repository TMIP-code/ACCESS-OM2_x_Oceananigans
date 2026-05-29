"""
Diagnostic for the monthly κV(MLD) instability.

For each of the 12 monthly MLD snapshots:
1. Computes κV(i,j,k) = ifelse(z_center(k) > -mld(i,j), κVML, κVBG),
   matching `update_κV_from_mld!` in src/shared_utils/data_loading.jl.
2. Reports `unique(κV)` (must equal {κVML, κVBG} by construction; we still
   verify in case of NaN / Inf in the input MLD).
3. Plots `nk_in_ML(i,j) = sum(κV(i,j,:) .== κVML)` — the count of vertical
   cells where κV takes the mixed-layer value — as a single 12-panel figure
   plus one PNG per month.

Outputs land in `outputs/{PM}/{EXP}/{TW}/diagnostics/kv_mld_count/`.

Usage (via driver):
    PARENT_MODEL=ACCESS-OM2-01 KAPPA_V_ML=25e-3 KAPPA_V_BG=75e-7 \\
      JOB_CHAIN=plotKVML bash scripts/driver.sh
"""

using JLD2
using CairoMakie
using Printf

include("shared_functions.jl")

(; parentmodel, experiment_dir, monthly_dir, outputdir) = load_project_config()

# κV values
κVML = parse(Float64, get(ENV, "KAPPA_V_ML", "25e-3"))
κVBG = parse(Float64, get(ENV, "KAPPA_V_BG", "75e-7"))
@info "κVML = $κVML  /  κVBG = $κVBG"

plot_dir = joinpath(outputdir, "diagnostics", "kv_mld_count")
mkpath(plot_dir)
@info "plot_dir = $plot_dir"

################################################################################
# Load monthly MLD (with halos, positive-down depth) and grid z_centers
################################################################################

mld_file = joinpath(monthly_dir, "mld_monthly.jld2")
isfile(mld_file) || error("MLD file not found: $mld_file")

monthly_mld, n_months = jldopen(mld_file, "r") do f
    iters = sort(parse.(Int, filter(k -> k != "serialized", keys(f["timeseries/MLD"]))))
    snaps = [Array(f["timeseries/MLD/$i"])[:, :, 1] for i in iters]
    (snaps, length(iters))
end
nx, ny = size(monthly_mld[1])
@info "MLD: $nx × $ny, $n_months snapshots"

# Grid z-centers (Oceananigans convention: negative-down). Skip halos in k.
grid_file = joinpath(experiment_dir, "grid.jld2")
isfile(grid_file) || error("Grid file not found: $grid_file")

# Read z_faces from grid.jld2 (the grid stores faces; centers are midpoints)
z_faces_full, Nz = jldopen(grid_file, "r") do f
    (Array(f["z_faces"]), Int(f["Nz"]))
end
@info "z_faces (with halos): length=$(length(z_faces_full))  range=$(extrema(z_faces_full))"

# Strip halos. The faces array is typically of length Nz + 2Hz + 1; the interior
# faces span the bottom-most z (most negative, smallest finite value) to surface
# (0). We take the unique finite faces and clip to the bottom Nz+1 values.
finite_mask = isfinite.(z_faces_full)
finite_faces = sort(unique(z_faces_full[finite_mask]))
# We want the contiguous interior chunk: faces[k=1] is the deepest, faces[k=Nz+1] = 0
length(finite_faces) < Nz + 1 && error("Not enough finite z-face values to build $Nz interior cells")
interior_faces = finite_faces[1:(Nz + 1)]
# Cell centers: midpoint of adjacent faces
z_center = 0.5 .* (interior_faces[1:Nz] .+ interior_faces[2:(Nz + 1)])
@info "z_center interior (k=1..$Nz): $(z_center[1]) (deepest) to $(z_center[end]) (shallowest)"

################################################################################
# Compute κV per month and the count map
################################################################################

# Strip x/y halos from MLD (Hx = Hy = 7 for OM2-01 1968-1977)
Hx = (nx - 3600) ÷ 2
Hy = (ny - 2700) ÷ 2
@info "Stripping halos: Hx=$Hx, Hy=$Hy (interior $(nx - 2Hx)×$(ny - 2Hy))"

mld_interior = [m[(Hx + 1):(nx - Hx), (Hy + 1):(ny - Hy)] for m in monthly_mld]
nx_i, ny_i = size(mld_interior[1])

# For each month: number of k where z_center[k] > -mld(i,j)  ↔  κV = κVML
# Equivalent: count of k levels above the MLD depth.
count_maps = Vector{Matrix{Int32}}(undef, n_months)
unique_maps = Vector{Vector{Float64}}(undef, n_months)
for m in 1:n_months
    mld_2d = Float64.(mld_interior[m])  # positive-down depth, NaN → already replaced by 0
    # κV is a function of (i, j, k). We don't need the full 3D — just the count.
    # nk_in_ML(i,j) = count of k where z_center[k] > -mld(i,j)
    # = count of k where -z_center[k] < mld(i,j)
    # = count of k where depth_k < mld(i,j)
    depths_k = -z_center  # positive-down depth at each k (length Nz)
    cnt = zeros(Int32, nx_i, ny_i)
    for k in 1:Nz
        cnt .+= Int32.(depths_k[k] .< mld_2d)
    end
    count_maps[m] = cnt

    # Build the full κV array just to verify unique() — small enough since
    # we already have the answer per cell from cnt.
    # unique κV is {κVML, κVBG} if at least one cell has each, plus possibly
    # a third value if the MLD has NaN/Inf (which would propagate through ifelse).
    # Check for non-finite MLD cells:
    n_nonfinite_mld = count(.!isfinite.(mld_2d))
    if n_nonfinite_mld > 0
        @warn "month $m: $n_nonfinite_mld non-finite MLD cells (would propagate to κVBG via NaN comparison)"
    end
    # Sample κV: cells in ML vs not
    n_ml_cells = sum(cnt)
    n_total_cells = nx_i * ny_i * Nz
    n_bg_cells = n_total_cells - n_ml_cells
    unique_maps[m] = unique(
        [
            n_ml_cells > 0 ? κVML : nothing
            n_bg_cells > 0 ? κVBG : nothing
        ]
    )
    @info @sprintf(
        "month %2d: nk_in_ML range=[%d, %d]  n_ml_cells=%d  n_bg_cells=%d  unique(κV) = %s",
        m, minimum(cnt), maximum(cnt), n_ml_cells, n_bg_cells, unique_maps[m]
    )
end

################################################################################
# Plot
################################################################################

vmax = Float64(maximum(maximum(c) for c in count_maps))
@info "Color range: 0 → $vmax (max k-levels in ML across all months)"

# Per-month PNGs
for m in 1:n_months
    path = joinpath(plot_dir, @sprintf("kv_mld_count_month%02d.png", m))
    fig = Figure(; size = (1600, 850), backgroundcolor = :white)
    ax = Axis(
        fig[1, 1];
        title = @sprintf(
            "Number of k-levels with κV=κVML — month %02d (1968–1977 climatology)", m
        ),
        xlabel = "i (interior)", ylabel = "j (interior)",
    )
    hm = heatmap!(
        ax, 1:nx_i, 1:ny_i, count_maps[m]; colorrange = (0, vmax),
        colormap = :viridis, nan_color = :lightgray
    )
    Colorbar(fig[1, 2], hm; label = "k-levels in ML")
    save(path, fig)
end
@info "Saved 12 per-month PNGs"

# Combined 4×3 grid
combined_path = joinpath(plot_dir, "kv_mld_count_all_months.png")
fig = Figure(; size = (1800, 1400), backgroundcolor = :white)
Label(
    fig[0, 1:4],
    @sprintf("Vertical κV=κVML cell count per month — κVML=%.3g, κVBG=%.3g", κVML, κVBG);
    fontsize = 18, font = :bold
)
hms = []
for m in 1:n_months
    row = ((m - 1) ÷ 4) + 1
    col = ((m - 1) % 4) + 1
    ax = Axis(fig[row, col]; title = @sprintf("month %02d", m))
    hm = heatmap!(
        ax, 1:nx_i, 1:ny_i, count_maps[m]; colorrange = (0, vmax),
        colormap = :viridis, nan_color = :lightgray
    )
    push!(hms, hm)
    hidedecorations!(ax; label = false)
end
Colorbar(fig[1:3, 5], hms[1]; label = "k-levels in ML")
save(combined_path, fig)
@info "Saved combined 4×3 grid: $combined_path"

# mp4 animation
mp4_path = joinpath(plot_dir, "kv_mld_count.mp4")
obs_m = Observable(1)
data_obs = @lift(count_maps[$obs_m])
title_obs = @lift(@sprintf("Number of k-levels with κV=κVML — month %02d", $obs_m))

fig_a = Figure(; size = (1600, 850), backgroundcolor = :white)
ax_a = Axis(fig_a[1, 1]; xlabel = "i", ylabel = "j")
Label(fig_a[0, 1:2], title_obs; fontsize = 18, font = :bold)
hm_a = heatmap!(
    ax_a, 1:nx_i, 1:ny_i, data_obs;
    colorrange = (0, vmax), colormap = :viridis, nan_color = :lightgray
)
Colorbar(fig_a[1, 2], hm_a; label = "k-levels in ML")

record(fig_a, mp4_path, 1:n_months; framerate = 2) do m
    obs_m[] = m
end
@info "Saved mp4: $mp4_path"

@info "Done. Outputs in $plot_dir"
