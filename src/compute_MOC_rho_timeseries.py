# Compute density-space MOC streamfunction as a full monthly timeseries.
#
# Output: one NetCDF file per model with the global streamfunction
#   psi_tot(time, potrho, grid_yu_ocean)  in Sv
# computed from ty_trans_rho + ty_trans_rho_gm (resolved + GM), summed
# zonally and cumsummed over potrho.
#
# Usage:
#   python3 src/compute_MOC_rho_timeseries.py ACCESS-OM2-1   1deg_jra55_iaf_omip2_cycle6
#   python3 src/compute_MOC_rho_timeseries.py ACCESS-OM2-025 025deg_jra55_iaf_omip2_cycle6
#   python3 src/compute_MOC_rho_timeseries.py ACCESS-OM2-01  01deg_jra55v140_iaf_cycle4

import sys
import traceback
from os import environ, makedirs

environ["PYTHONWARNINGS"] = "ignore"

from dask.distributed import Client
import intake

# Parse arguments
if len(sys.argv) < 3:
    print("Usage: python3 compute_MOC_rho_timeseries.py MODEL SUBCATALOG")
    sys.exit(1)

model = sys.argv[1]
subcatalog = sys.argv[2]

PROJECT = environ["PROJECT"]

# Reference density for kg/s → Sv conversion
rho0 = 1035.0

# Chunk sizes. For OM2-1/025, match the on-disk _ChunkSizes (see ncdump -hs).
# For OM2-01, load the full horizontal slab per time step (per the proven
# approach in notebooks/scripts/MOC_ACCESS-OM2-01.py): on-disk chunks
# (135×180) lead to too many dask chunks and poor throughput.
CHUNKS_BY_MODEL = {
    "ACCESS-OM2-1":   {"time": -1, "potrho": 27,  "grid_xt_ocean": 120,  "grid_yu_ocean": 100},
    "ACCESS-OM2-025": {"time": -1, "potrho": 40,  "grid_xt_ocean": 120,  "grid_yu_ocean": 108},
    "ACCESS-OM2-01":  {"time": 1,  "potrho": 160, "grid_xt_ocean": 3600, "grid_yu_ocean": 2700},
}


def select_data(cat, xarray_open_kwargs, **kwargs):
    selectedcat = cat.search(**kwargs)
    print("\nselectedcat:", selectedcat)
    xarray_combine_by_coords_kwargs = dict(
        compat="override",
        data_vars="minimal",
        coords="minimal",
    )
    return selectedcat.to_dask(
        xarray_open_kwargs=xarray_open_kwargs,
        xarray_combine_by_coords_kwargs=xarray_combine_by_coords_kwargs,
        parallel=True,
    )


if __name__ == "__main__":
    chunks = CHUNKS_BY_MODEL.get(model)
    if chunks is None:
        print(f"ERROR: no chunk sizes for model '{model}'")
        sys.exit(1)

    client = Client(n_workers=48, threads_per_worker=1)
    print(f"Dask client: {client}")

    # Load catalog
    print(f"Loading catalog for {model} / {subcatalog}")
    catalogs = intake.cat.access_nri
    cat = catalogs[subcatalog]
    searched_cat = cat.search(variable=["ty_trans_rho", "ty_trans_rho_gm"])
    print(searched_cat)

    # Output directory
    datadir = f"/scratch/{PROJECT}/TMIP/data"
    outputdir = f"{datadir}/{model}/{subcatalog}/rhospace"
    makedirs(outputdir, exist_ok=True)
    print(f"Output directory: {outputdir}")

    # Resolved: ty_trans_rho → zonal sum, then cumsum over potrho, minus total
    # (same convention as compute_AABW_depthspace.py: ψ = 0 at densest)
    # MOM writes _FillValue for density classes that don't intersect the
    # water column; fillna(0) treats them as zero transport (physically
    # correct) so NaNs don't leak through the zonal sum + cumsum.
    try:
        print("Loading ty_trans_rho")
        ty = select_data(
            searched_cat,
            dict(chunks=chunks),
            variable="ty_trans_rho",
            frequency="1mon",
        )
        ty = ty.fillna(0.0)
        print("Zonal sum + cumsum over potrho")
        psi_res = ty.sum("grid_xt_ocean")
        psi_res = psi_res.cumulative("potrho").sum() - psi_res.sum("potrho")
    except Exception:
        print(f"Error processing {model} ty_trans_rho")
        print(traceback.format_exc())
        sys.exit(1)

    # GM: ty_trans_rho_gm → zonal sum only. OM2-01 is eddy-resolving and
    # doesn't output GM, so treat as zero when the variable is absent.
    has_gm = len(searched_cat.search(variable="ty_trans_rho_gm").df) > 0
    if has_gm:
        try:
            print("Loading ty_trans_rho_gm")
            ty_gm = select_data(
                searched_cat,
                dict(chunks=chunks),
                variable="ty_trans_rho_gm",
                frequency="1mon",
            )
            ty_gm = ty_gm.fillna(0.0)
            print("Zonal sum (no cumsum — already a transport)")
            psi_gm = ty_gm.sum("grid_xt_ocean")
        except Exception:
            print(f"Error processing {model} ty_trans_rho_gm")
            print(traceback.format_exc())
            sys.exit(1)
    else:
        print(
            f"{model} has no ty_trans_rho_gm (eddy-resolving, no GM param) — "
            "setting ψ_gm = 0"
        )
        psi_gm = None

    # Total = resolved + GM, converted to Sv
    try:
        print("Computing total ψ = resolved + GM, converting to Sv")
        if psi_gm is not None:
            psi_tot_raw = psi_res.ty_trans_rho + psi_gm.ty_trans_rho_gm
        else:
            psi_tot_raw = psi_res.ty_trans_rho
        psi_tot = psi_tot_raw / (rho0 * 1e6)
        psi_tot = psi_tot.to_dataset(name="psi_tot")
        psi_tot["psi_tot"].attrs["units"] = "Sv"
        psi_tot["psi_tot"].attrs["long_name"] = (
            "Global density-space meridional overturning streamfunction"
        )
        outfile = f"{outputdir}/psi_tot_global.nc"
        print(f"Saving to {outfile}")
        psi_tot.to_netcdf(outfile, compute=True)
    except Exception:
        print(f"Error writing {model} psi_tot")
        print(traceback.format_exc())
        sys.exit(1)

    # 12-month rolling mean (day-of-month weighted), same pattern as
    # compute_AABW_depthspace.py
    try:
        print("Loading psi_tot into memory for rolling mean")
        psi_tot = psi_tot.load()

        print("Computing 12-month day-weighted rolling mean")
        window_size = 12
        psi_rolling = psi_tot.rolling(time=window_size).construct("window")
        month_weights = psi_tot.time.dt.days_in_month
        month_weights_rolling = (
            month_weights.rolling(time=window_size).construct("window")
        )
        psi_rolling_weighted = psi_rolling.weighted(month_weights_rolling.fillna(0))
        psi_rollingyear = psi_rolling_weighted.mean("window", skipna=False)
        psi_rollingyear["psi_tot"].attrs["units"] = "Sv"
        psi_rollingyear["psi_tot"].attrs["long_name"] = (
            "Global density-space MOC, 12-month day-weighted rolling mean"
        )
        outfile = f"{outputdir}/psi_tot_rollingyear_global.nc"
        print(f"Saving to {outfile}")
        psi_rollingyear.to_netcdf(outfile, compute=True)
    except Exception:
        print(f"Error writing {model} psi_tot_rollingyear")
        print(traceback.format_exc())
        sys.exit(1)

    client.close()
    print("Done!")
