# Compute depth-space overturning streamfunction and AABW transport timeseries
#
# Adapted from /home/561/bp3051/Projects/TMIP/notebooks/scripts/MOC_ACCESS-OM2_1deg_jra55_iaf_omip2_cycle6.py
# Key difference: uses ty_trans (depth-space) instead of ty_trans_rho (density-space)
#
# Usage:
#   python3 src/compute_AABW_depthspace.py ACCESS-OM2-1 1deg_jra55_iaf_omip2_cycle6
#   python3 src/compute_AABW_depthspace.py ACCESS-OM2-025 025deg_jra55_iaf_omip2_cycle6

import sys
from os import environ, makedirs
environ["PYTHONWARNINGS"] = "ignore"

from dask.distributed import Client
import intake
import xarray as xr
import numpy as np
import traceback

# Parse arguments
if len(sys.argv) < 3:
    print("Usage: python3 compute_AABW_depthspace.py MODEL SUBCATALOG")
    print("  e.g. python3 compute_AABW_depthspace.py ACCESS-OM2-1 1deg_jra55_iaf_omip2_cycle6")
    sys.exit(1)

model = sys.argv[1]
subcatalog = sys.argv[2]

PROJECT = environ["PROJECT"]


def select_data(cat, xarray_open_kwargs, **kwargs):
    selectedcat = cat.search(**kwargs)
    print("\nselectedcat: ", selectedcat)
    xarray_combine_by_coords_kwargs = dict(
        compat="override",
        data_vars="minimal",
        coords="minimal"
    )
    datadask = selectedcat.to_dask(
        xarray_open_kwargs=xarray_open_kwargs,
        xarray_combine_by_coords_kwargs=xarray_combine_by_coords_kwargs,
        parallel=True,
    )
    return datadask


def yearlymeans(ds):
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    np.testing.assert_allclose(weights.groupby("time.year").sum().values, 1.0)
    return (ds * weights).groupby("time.year").sum(dim="time")


# Load catalog
print(f"Loading catalog for {model} / {subcatalog}")
catalogs = intake.cat.access_nri
cat = catalogs[subcatalog]
print(cat)

# Search for depth-space transport variables
searched_cat = cat.search(variable=["ty_trans", "ty_trans_gm"])
print(searched_cat)

# Output directory
datadir = f'/scratch/{PROJECT}/TMIP/data'
outputdir = f'{datadir}/{model}/{subcatalog}/depthspace'

if __name__ == '__main__':
    client = Client(n_workers=48, threads_per_worker=1)

    makedirs(outputdir, exist_ok=True)
    print(f"Output directory: {outputdir}")

    # Chunk sizes matching periodicaverage.py (CHUNKS_TY / CHUNKS_TY_GM)
    if "025" in model:
        chunks_ty = {"time": -1, "xt_ocean": 120, "yu_ocean": 108, "st_ocean": 25}
        chunks_gm = {"time": -1, "xt_ocean": 120, "yu_ocean": 108, "st_ocean": 25}
    else:
        chunks_ty = {"time": -1, "xt_ocean": 180, "yu_ocean": 150, "st_ocean": 25}
        chunks_gm = {"time": -1, "xt_ocean": 180, "yu_ocean": 150, "st_ocean": 25}

    # 1. Resolved streamfunction: ty_trans → sum longitude + cumsum depth
    try:
        print("Loading ty_trans data (depth-space meridional transport)")
        ty_trans_datadask = select_data(searched_cat,
            dict(chunks=chunks_ty),
            variable="ty_trans",
            frequency="1mon",
        )
        print("\nty_trans_datadask: ", ty_trans_datadask)
        print("Sum longitudinally and cumsum vertically (depth)")
        psi = ty_trans_datadask.sum("xt_ocean")
        psi = psi.cumulative('st_ocean').sum() - psi.sum('st_ocean')
        print("\npsi: ", psi)
        print(f"Saving psi to: {outputdir}/psi.nc")
        psi.to_netcdf(f'{outputdir}/psi.nc', compute=True)
    except Exception:
        print(f'Error processing {model} ty_trans')
        print(traceback.format_exc())

    # 2. GM streamfunction: ty_trans_gm → sum longitude only (already a transport)
    try:
        print("Loading ty_trans_gm data (GM meridional transport)")
        ty_trans_gm_datadask = select_data(searched_cat,
            dict(chunks=chunks_gm),
            variable="ty_trans_gm",
            frequency="1mon",
        )
        print("\nty_trans_gm_datadask: ", ty_trans_gm_datadask)
        print("Sum longitudinally (no cumsum — already a transport)")
        psi_gm = ty_trans_gm_datadask.sum("xt_ocean")
        print("\npsi_gm: ", psi_gm)
        print(f"Saving psi_gm to: {outputdir}/psi_gm.nc")
        psi_gm.to_netcdf(f'{outputdir}/psi_gm.nc', compute=True)
    except Exception:
        print(f'Error processing {model} ty_trans_gm')
        print(traceback.format_exc())

    # 3. Total streamfunction = resolved + GM
    try:
        print("Calculating total depth-space overturning streamfunction")
        psi_tot = (psi.ty_trans + psi_gm.ty_trans_gm).to_dataset(name='psi_tot')
        print("\npsi_tot: ", psi_tot)
        psi_tot.to_netcdf(f'{outputdir}/psi_tot.nc', compute=True)
    except Exception:
        print(f'Error processing {model} psi_tot')
        print(traceback.format_exc())

    # 4. Yearly means
    try:
        print("Yearly means")
        psi_tot_year = yearlymeans(psi_tot)
        print("\npsi_tot_year: ", psi_tot_year)
        psi_tot_year.to_netcdf(f'{outputdir}/psi_tot_year.nc', compute=True)
    except Exception:
        print(f'Error processing {model} psi_tot_year')
        print(traceback.format_exc())

    # 5. Time-mean
    try:
        print("Averaging total overturning streamfunction")
        psi_tot_avg = psi_tot.weighted(psi_tot.time.dt.days_in_month).mean(dim="time")
        print("\npsi_tot_avg: ", psi_tot_avg)
        psi_tot_avg.to_netcdf(f'{outputdir}/psi_tot_avg.nc', compute=True)
    except Exception:
        print(f'Error processing {model} psi_tot_avg')
        print(traceback.format_exc())

    # 6. Rolling yearly average
    try:
        print("Loading psi_tot into memory for rolling averages")
        psi_tot = psi_tot.load()
    except Exception:
        print(f'Error loading {model} psi_tot into memory')
        print(traceback.format_exc())

    try:
        print("Calculating rolling yearly average")
        window_size = 12
        psi_tot_rolling = psi_tot.rolling(time=window_size).construct("window")
        month_weights = psi_tot.time.dt.days_in_month
        month_weights_rolling = month_weights.rolling(time=window_size).construct("window")
        psi_tot_rolling_weighted = psi_tot_rolling.weighted(month_weights_rolling.fillna(0))
        psi_tot_rollingyear = psi_tot_rolling_weighted.mean("window", skipna=False)
        psi_tot_rollingyear.to_netcdf(f'{outputdir}/psi_tot_rollingyear.nc', compute=True)
    except Exception:
        print(f'Error processing {model} psi_tot_rollingyear')
        print(traceback.format_exc())

    # 7. Rolling decadal average
    try:
        print("Calculating rolling decadal average")
        window_size = 120
        psi_tot_rolling = psi_tot.rolling(time=window_size).construct("window")
        month_weights = psi_tot.time.dt.days_in_month
        month_weights_rolling = month_weights.rolling(time=window_size).construct("window")
        psi_tot_rolling_weighted = psi_tot_rolling.weighted(month_weights_rolling.fillna(0))
        psi_tot_rollingdecade = psi_tot_rolling_weighted.mean("window", skipna=False)
        psi_tot_rollingdecade.to_netcdf(f'{outputdir}/psi_tot_rollingdecade.nc', compute=True)
    except Exception:
        print(f'Error processing {model} psi_tot_rollingdecade')
        print(traceback.format_exc())

    client.close()
    print("Done!")
