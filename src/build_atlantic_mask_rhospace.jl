# Build a 2D Atlantic basin mask on the (grid_xt_ocean, grid_yu_ocean) axes
# used by the density-space MOC NetCDFs.
#
# The mask is `isatlantic(lat, lon) AND lat ≥ -30°S` (same convention as
# src/plot_MOC_rho.jl line 79 and src/plot_MOC.jl line 86), saved as an Int8
# 2D variable. Coordinates are read from psi_tot_global.nc to guarantee the
# mask aligns 1:1 with the global ψ used downstream.
#
# Usage:
#   PARENT_MODEL=ACCESS-OM2-1   julia --project src/build_atlantic_mask_rhospace.jl
#   PARENT_MODEL=ACCESS-OM2-025 julia --project src/build_atlantic_mask_rhospace.jl
#
# Writes: /scratch/{PROJECT}/TMIP/data/{PARENT_MODEL}/{EXPERIMENT}/rhospace/atlantic_mask.nc

using NCDatasets
using OceanBasins

PROJECT = get(ENV, "PROJECT", "y99")
PARENT_MODEL = get(ENV, "PARENT_MODEL", "ACCESS-OM2-1")
EXPERIMENT = if haskey(ENV, "EXPERIMENT")
    ENV["EXPERIMENT"]
elseif PARENT_MODEL == "ACCESS-OM2-1"
    "1deg_jra55_iaf_omip2_cycle6"
elseif PARENT_MODEL == "ACCESS-OM2-025"
    "025deg_jra55_iaf_omip2_cycle6"
elseif PARENT_MODEL == "ACCESS-OM2-01"
    "01deg_jra55v140_iaf_cycle4"
else
    error("No default EXPERIMENT for PARENT_MODEL=$PARENT_MODEL")
end

rhospace_dir = "/scratch/$PROJECT/TMIP/data/$PARENT_MODEL/$EXPERIMENT/rhospace"
psi_path = joinpath(rhospace_dir, "psi_tot_global.nc")
out_path = joinpath(rhospace_dir, "atlantic_mask.nc")

isfile(psi_path) || error(
    "$psi_path not found — run scripts/prepreprocessing/compute_MOC_rho_timeseries.sh (BASIN=global) first."
)

# `psi_tot_global.nc` is already zonally summed, so it only has `grid_yu_ocean`.
# To get the matching `grid_xt_ocean` axis we look for any preprocessed
# `ty_trans_monthly.nc` in the experiment (lon is static across time windows).
exp_dir = joinpath(@__DIR__, "..", "preprocessed_inputs", PARENT_MODEL, EXPERIMENT)
ty_trans_path = nothing
if isdir(exp_dir)
    for tw_subdir in readdir(exp_dir; join = true)
        candidate = joinpath(tw_subdir, "monthly", "ty_trans_monthly.nc")
        if isfile(candidate)
            ty_trans_path = candidate
            break
        end
    end
end
ty_trans_path === nothing && error(
    "No preprocessed ty_trans_monthly.nc found under $exp_dir/*/monthly/ — " *
        "run the preprocessing pipeline (JOB_CHAIN=prep) for any time window first."
)

@info "Reading lat from $psi_path"
lat = NCDataset(psi_path, "r") do ds
    ds["grid_yu_ocean"][:]
end
@info "Reading lon from $ty_trans_path (preprocessed `xt_ocean`)"
lon = NCDataset(ty_trans_path, "r") do ds
    ds["xt_ocean"][:]
end
Ny, Nx = length(lat), length(lon)
@info "Grid: Ny=$Ny, Nx=$Nx"

@info "Building Atlantic mask (isatlantic AND lat ≥ -30°S)"
OCEANS = oceanpolygons()
lon2D = repeat(reshape(lon, 1, Nx); outer = (Ny, 1))
lat2D = repeat(lat; outer = (1, Nx))
atl_2D = reshape(isatlantic(vec(lat2D), vec(lon2D), OCEANS), Ny, Nx) .& (lat2D .>= -30.0)

frac_atl = count(atl_2D) / length(atl_2D)
@info "Atlantic fraction of total cells: $(round(100 * frac_atl; digits = 1))%"

@info "Writing $out_path"
isfile(out_path) && rm(out_path)
NCDataset(out_path, "c") do dsout
    defDim(dsout, "grid_xt_ocean", Nx)
    defDim(dsout, "grid_yu_ocean", Ny)
    defVar(dsout, "grid_xt_ocean", Float64, ("grid_xt_ocean",); attrib = ["units" => "degrees_E"])[:] = lon
    defVar(dsout, "grid_yu_ocean", Float64, ("grid_yu_ocean",); attrib = ["units" => "degrees_N"])[:] = lat
    v = defVar(
        dsout, "atlantic_mask", Int8, ("grid_yu_ocean", "grid_xt_ocean");
        attrib = [
            "long_name" => "Atlantic basin mask (1 = Atlantic, 0 = other), lat >= -30 clamp applied",
            "source" => "OceanBasins.isatlantic on (grid_yu_ocean, grid_xt_ocean) from psi_tot_global.nc",
        ]
    )
    v[:, :] = Int8.(atl_2D)
end
@info "Done."
