################################################################################
# Water mass transport metrics from density-space MOC streamfunctions
#
# Helpers shared by `plot_AABW_timeseries.jl` and `plot_AABW_NADW_timeseries.jl`.
# Both work on ψ(lat, ρ, t) loaded from the NetCDFs written by
# `compute_MOC_rho_timeseries.py` (psi_tot_global.nc, psi_tot_atlantic.nc, ...).
################################################################################

using NCDatasets
using Dates
using Statistics: mean

"""
    load_rho_psi(filepath) -> (psi, lat, potrho, time_vals)

Load ψ + axes from a NetCDF written by `compute_MOC_rho_timeseries.py`.
ψ is returned in Sv, with shape `(grid_yu_ocean, potrho, time)`.
"""
function load_rho_psi(filepath)
    ds = NCDataset(filepath, "r"; maskingvalue = NaN)
    psi = ds["psi_tot"][:, :, :]
    lat = ds["grid_yu_ocean"][:]
    potrho = ds["potrho"][:]
    time_vals = ds["time"][:]
    close(ds)
    return psi, lat, potrho, time_vals
end

"""
    rho_aabw_metric(psi, lat, potrho; lat_max, rho_threshold) -> Vector

AABW metric: `min ψ` over `(lat < lat_max, ρ ≥ rho_threshold)` at each time step.
Returns a length-`nt` vector (negative, since AABW is the lower negative cell).
"""
function rho_aabw_metric(psi, lat, potrho; lat_max, rho_threshold)
    lat_mask = lat .< lat_max
    rho_mask = potrho .>= rho_threshold
    sub = psi[lat_mask, rho_mask, :]
    return [minimum(skipmissing(sub[:, :, i])) for i in axes(sub, 3)]
end

"""
    rho_nadw_metric(psi, lat, potrho; lat_min, lat_max, rho_min=-Inf, rho_max=Inf) -> Vector

NADW metric: `max ψ` over `(lat_min ≤ lat ≤ lat_max, rho_min ≤ ρ ≤ rho_max)` at
each time step. Returns a length-`nt` vector (positive, the NADW overturning cell).
`NaN` is returned for any time step where the box contains only missing values.
"""
function rho_nadw_metric(psi, lat, potrho; lat_min, lat_max, rho_min = -Inf, rho_max = Inf)
    lat_mask = (lat .>= lat_min) .& (lat .<= lat_max)
    rho_mask = (potrho .>= rho_min) .& (potrho .<= rho_max)
    sub = psi[lat_mask, rho_mask, :]
    return [_safe_max(sub[:, :, i]) for i in axes(sub, 3)]
end

function _safe_max(slice)
    nonmissing = collect(skipmissing(slice))
    isempty(nonmissing) && return NaN
    return maximum(nonmissing)
end

"""
    yearly_mean_of_monthly(time_vals, monthly_vals) -> (years, yearly_means)

Apply the spatial metric to each month first (already done by caller), then
average over the 12 months of each calendar year.
"""
function yearly_mean_of_monthly(time_vals, monthly_vals)
    yrs = [Dates.year(t) for t in time_vals]
    unique_yrs = sort(unique(yrs))
    yearly_means = Float64[]
    for y in unique_yrs
        idx = findall(yrs .== y)
        push!(yearly_means, mean(monthly_vals[idx]))
    end
    return unique_yrs, yearly_means
end
