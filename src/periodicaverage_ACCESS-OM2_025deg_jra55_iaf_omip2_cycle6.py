
# import sys to access script arguments (experiment, ensemble, first_year, last_year)
import sys

# interactive use only
model = "ACCESS-OM2-025"
subcatalog = "025deg_jra55_iaf_omip2_cycle6"
year_start = 1960
num_years = 20

# 1. Load packages

# Ignore warnings
from os import environ
environ["PYTHONWARNINGS"] = "ignore"
PROJECT = environ["PROJECT"]

# Import makedirs to create directories where I write new files
from os import makedirs

# Load dask
from dask.distributed import Client

# Load intake and cosima cookbook
import intake

# Load xarray for N-dimensional arrays
import xarray as xr

# Load datetime to deal with time formats
import datetime

# Load traceback to print exceptions
import traceback

# Load pandas for data manipulation
import pandas as pd

# Import numpy
import numpy as np

# 2. Define some functions
# (to avoid too much boilerplate code)
print("Defining functions")

def time_window_strings(year_start, num_years):
    """
    return strings for start_time and end_time
    """
    # start_time is first second of year_start
    start_time = f'{year_start:04d}'
    # end_time is last second of last_year
    end_time = f'{year_start + num_years - 1:04d}'
    # Return the weighted average
    return start_time, end_time

def select_data(cat, xarray_open_kwargs, **kwargs):
    selectedcat = cat.search(**kwargs)
    print("\nselectedcat: ", selectedcat)
    xarray_combine_by_coords_kwargs=dict(
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

def month_climatology(ds):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby("time.month") / month_length.groupby("time.month").sum()
    # Test that the sum of the weights for each month is 1.0
    np.testing.assert_allclose(weights.groupby("time.month").sum().values, np.ones(12))
    # Calculate the weighted average
    ds_out = (ds * weights).groupby("time.month").sum(dim="time")
    # Keep track of mean number of days per month
    mean_days_in_month = month_length.groupby("time.month").mean()
    # And assign it to new coordinate
    ds_out = ds_out.assign_coords(mean_days_in_month=('month', mean_days_in_month.data))
    return ds_out

# 3. Load catalog

catalogs = intake.cat.access_nri
print(catalogs)
print(catalogs.keys())
cat = catalogs[subcatalog]
print(cat)

# Only keep the required data
searched_cat = cat.search(variable = ["u", "v", "wt", "tx_trans", "ty_trans", "tx_trans_gm", "ty_trans_gm", "mld", "area_t", "dht", "eta_t"])
print(searched_cat)
# Find config.yaml by walking up from first catalog path
from pathlib import Path
_p = Path(searched_cat.df.path.iloc[0])
while _p != _p.parent:
    _config = _p / "config.yaml"
    if _config.exists():
        print(f"\nFound config: {_config}")
        break
    _p = _p.parent
else:
    print("\nconfig.yaml not found in any parent directory")


# Create directory on scratch to save the data
datadir = f'/scratch/{PROJECT}/TMIP/data'
start_time, end_time = time_window_strings(year_start, num_years)
start_time_str = f'Jan{start_time}'
end_time_str = f'Dec{end_time}'



# 4. Load data, preprocess it, and save it to NetCDF

# print("Starting client")

# # This `if` statement is required in scripts (not required in Jupyter)
# if __name__ == '__main__':
#     client = Client(n_workers=48, threads_per_worker=1) #, memory_limit='16GB') # Note: with 1thread/worker cannot plot thetao. Maybe I need to understand why?


#     # directory to save the data to (as NetCDF)
#     outputdir = f'{datadir}/{model}/{subcatalog}/{start_time_str}-{end_time_str}'
#     print("Creating directory: ", outputdir)
#     makedirs(outputdir, exist_ok=True)

    # # area_t
    # try:
    #     print("Loading area_t data")
    #     area_t_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'xt_ocean':240, 'yt_ocean':216}
    #         ),
    #         variable = "area_t",
    #         frequency = "fx",
    #     )
    #     print("\narea_t_datadask: ", area_t_datadask)
    #     area_t = area_t_datadask["area_t"]
    #     print("\narea_t: ", area_t)
    #     print("Saving area_t to: ", f'{outputdir}/area_t.nc')
    #     area_t.to_netcdf(f'{outputdir}/area_t.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} area_t')
    #     print(traceback.format_exc())


    # # tx_trans
    # try:
    #     print("Loading tx_trans data")
    #     tx_trans_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xu_ocean':120, 'yt_ocean':108, 'lev':25}
    #         ),
    #         variable = "tx_trans",
    #         frequency = "1mon",
    #     )
    #     print("\ntx_trans_datadask: ", tx_trans_datadask)
    #     print("Slicing tx_trans for the time period")
    #     tx_trans_datadask_sel = tx_trans_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging tx_trans into monthly climatology")
    #     tx_trans = month_climatology(tx_trans_datadask_sel["tx_trans"])
    #     print("\ntx_trans: ", tx_trans)
    #     print("Saving tx_trans to: ", f'{outputdir}/tx_trans_periodic.nc')
    #     tx_trans.to_dataset(name="tx_trans").to_netcdf(f'{outputdir}/tx_trans_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} tx_trans')
    #     print(traceback.format_exc())

    # # ty_trans
    # try:
    #     print("Loading ty_trans data")
    #     ty_trans_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xt_ocean':120, 'yu_ocean':108, 'lev':25}
    #         ),
    #         variable = "ty_trans",
    #         frequency = "1mon",
    #     )
    #     print("\nty_trans_datadask: ", ty_trans_datadask)
    #     print("Slicing ty_trans for the time period")
    #     ty_trans_datadask_sel = ty_trans_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging ty_trans into monthly climatology")
    #     ty_trans = month_climatology(ty_trans_datadask_sel["ty_trans"])
    #     print("\nty_trans: ", ty_trans)
    #     print("Saving ty_trans to: ", f'{outputdir}/ty_trans_periodic.nc')
    #     ty_trans.to_dataset(name="ty_trans").to_netcdf(f'{outputdir}/ty_trans_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} ty_trans')
    #     print(traceback.format_exc())

    # # tx_trans_gm
    # try:
    #     print("Loading tx_trans_gm data")
    #     tx_trans_gm_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xu_ocean':120, 'yt_ocean':108, 'lev':25}
    #         ),
    #         variable = "tx_trans_gm",
    #         frequency = "1mon",
    #     )
    #     print("\ntx_trans_gm_datadask: ", tx_trans_gm_datadask)
    #     print("Slicing tx_trans_gm for the time period")
    #     tx_trans_gm_datadask_sel = tx_trans_gm_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging tx_trans_gm into monthly climatology")
    #     tx_trans_gm = month_climatology(tx_trans_gm_datadask_sel["tx_trans_gm"])
    #     print("\ntx_trans_gm: ", tx_trans_gm)
    #     print("Saving tx_trans_gm to: ", f'{outputdir}/tx_trans_gm_periodic.nc')
    #     tx_trans_gm.to_dataset(name="tx_trans_gm").to_netcdf(f'{outputdir}/tx_trans_gm_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} tx_trans_gm')
    #     print(traceback.format_exc())

    # # ty_trans_gm
    # try:
    #     print("Loading ty_trans_gm data")
    #     ty_trans_gm_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xt_ocean':120, 'yu_ocean':108, 'lev':25}
    #         ),
    #         variable = "ty_trans_gm",
    #         frequency = "1mon",
    #     )
    #     print("\nty_trans_gm_datadask: ", ty_trans_gm_datadask)
    #     print("Slicing ty_trans_gm for the time period")
    #     ty_trans_gm_datadask_sel = ty_trans_gm_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging ty_trans_gm into monthly climatology")
    #     ty_trans_gm = month_climatology(ty_trans_gm_datadask_sel["ty_trans_gm"])
    #     print("\nty_trans_gm: ", ty_trans_gm)
    #     print("Saving ty_trans_gm to: ", f'{outputdir}/ty_trans_gm_periodic.nc')
    #     ty_trans_gm.to_dataset(name="ty_trans_gm").to_netcdf(f'{outputdir}/ty_trans_gm_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} ty_trans_gm')
    #     print(traceback.format_exc())

    # # mld dataset
    # try:
    #     print("Loading mld data")
    #     mld_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xt_ocean':240, 'yt_ocean':216}
    #         ),
    #         variable = "mld",
    #         frequency = "1mon",
    #     )
    #     print("\nmld_datadask: ", mld_datadask)
    #     print("Slicing mld for the time period")
    #     mld_datadask_sel = mld_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging mld into monthly climatology")
    #     mld = month_climatology(mld_datadask_sel["mld"])
    #     print("\nmld: ", mld)
    #     print("Saving mld to: ", f'{outputdir}/mld_periodic.nc')
    #     mld.to_dataset(name="mld").to_netcdf(f'{outputdir}/mld_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} mld')
    #     print(traceback.format_exc())

    # # dht
    # try:
    #     print("Loading dht data")
    #     dht_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xt_ocean':120, 'yt_ocean':108, 'lev':25}
    #         ),
    #         variable = "dht",
    #         frequency = "1mon",
    #     )
    #     print("\ndht_datadask: ", dht_datadask)
    #     print("Slicing dht for the time period")
    #     dht_datadask_sel = dht_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging dht into monthly climatology")
    #     dht = month_climatology(dht_datadask_sel["dht"])
    #     print("\ndht: ", dht)
    #     print("Saving dht to: ", f'{outputdir}/dht_periodic.nc')
    #     dht.to_dataset(name="dht").to_netcdf(f'{outputdir}/dht_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} dht')
    #     print(traceback.format_exc())

    # # u
    # try:
    #     print("Loading u data")
    #     u_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xu_ocean':120, 'yu_ocean':108, 'lev':25}
    #         ),
    #         variable = "u",
    #         frequency = "1mon",
    #     )
    #     print("\nu_datadask: ", u_datadask)
    #     print("Slicing u for the time period")
    #     u_datadask_sel = u_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging u into monthly climatology")
    #     u = month_climatology(u_datadask_sel["u"])
    #     print("\nu: ", u)
    #     print("Saving u to: ", f'{outputdir}/u_periodic.nc')
    #     u.to_dataset(name="u").to_netcdf(f'{outputdir}/u_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} u')
    #     print(traceback.format_exc())

    # # v
    # try:
    #     print("Loading v data")
    #     v_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xu_ocean':120, 'yu_ocean':108, 'lev':25}
    #         ),
    #         variable = "v",
    #         frequency = "1mon",
    #     )
    #     print("\nv_datadask: ", v_datadask)
    #     print("Slicing v for the time period")
    #     v_datadask_sel = v_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging v into monthly climatology")
    #     v = month_climatology(v_datadask_sel["v"])
    #     print("\nv: ", v)
    #     print("Saving v to: ", f'{outputdir}/v_periodic.nc')
    #     v.to_dataset(name="v").to_netcdf(f'{outputdir}/v_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} v')
    #     print(traceback.format_exc())

    # # wt
    # try:
    #     print("Loading wt data")
    #     wt_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xt_ocean':120, 'yt_ocean':108, 'sw_ocean':25}
    #         ),
    #         variable = "wt",
    #         frequency = "1mon",
    #     )
    #     print("\nwt_datadask: ", wt_datadask)
    #     print("Slicing wt for the time period")
    #     wt_datadask_sel = wt_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging wt into monthly climatology")
    #     wt = month_climatology(wt_datadask_sel["wt"])
    #     print("\nwt: ", wt)
    #     print("Saving wt to: ", f'{outputdir}/wt_periodic.nc')
    #     wt.to_dataset(name="wt").to_netcdf(f'{outputdir}/wt_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} wt')
    #     print(traceback.format_exc())

    # # eta_t
    # try:
    #     print("Loading eta_t data")
    #     eta_t_datadask = select_data(searched_cat,
    #         dict(
    #             chunks={'time': -1, 'xt_ocean':240, 'yt_ocean':216}
    #         ),
    #         variable = "eta_t",
    #         frequency = "1mon",
    #     )
    #     print("\neta_t_datadask: ", eta_t_datadask)
    #     print("Slicing eta_t for the time period")
    #     eta_t_datadask_sel = eta_t_datadask.sel(time=slice(start_time, end_time))
    #     print("Averaging eta_t into monthly climatology")
    #     eta_t = month_climatology(eta_t_datadask_sel["eta_t"])
    #     print("\neta_t: ", eta_t)
    #     print("Saving eta_t to: ", f'{outputdir}/eta_t_periodic.nc')
    #     eta_t.to_dataset(name="eta_t").to_netcdf(f'{outputdir}/eta_t_periodic.nc', compute=True)
    # except Exception:
    #     print(f'Error processing {model} eta_t')
    #     print(traceback.format_exc())


    #     client.close()

