"""
Density-space preprocessing: compute monthly climatologies and yearly averages
of ty_trans_rho and ty_trans_rho_gm from ACCESS-OM2 output for use by
plot_MOC_rho.jl.

Runs independently of the main velocity pipeline (periodicaverage.py). These
variables are only needed for density-space MOC plots.

Environment variables:
    PARENT_MODEL   — ACCESS-OM2-1 or ACCESS-OM2-025 (default: ACCESS-OM2-1)
    EXPERIMENT     — intake catalog key (e.g. 1deg_jra55_iaf_omip2_cycle6)
    TIME_WINDOW    — year range "YYYY-YYYY" or single year "YYYY" (default: 1968-1977)

Output:
    preprocessed_inputs/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/monthly/ty_trans_rho{,_gm}_monthly.nc
    preprocessed_inputs/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/yearly/ty_trans_rho{,_gm}_yearly.nc
"""

import sys
import os
from os import makedirs
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"

from dask.distributed import Client
import intake
import numpy as np
import xarray as xr


# ── Configuration from environment ──────────────────────────────────────────

PARENT_MODEL = os.environ.get("PARENT_MODEL", "ACCESS-OM2-1")
DEFAULT_EXPERIMENTS = {
    "ACCESS-OM2-1": "1deg_jra55_iaf_omip2_cycle6",
    "ACCESS-OM2-025": "025deg_jra55_iaf_omip2_cycle6",
}
EXPERIMENT = os.environ.get("EXPERIMENT", DEFAULT_EXPERIMENTS.get(PARENT_MODEL, ""))
if not EXPERIMENT:
    print(f"ERROR: No default EXPERIMENT for {PARENT_MODEL}; set EXPERIMENT env var", file=sys.stderr)
    sys.exit(1)

TIME_WINDOW = os.environ.get("TIME_WINDOW", "1968-1977")

if "-" in TIME_WINDOW:
    year_start_str, year_end_str = TIME_WINDOW.split("-", 1)
else:
    year_start_str = year_end_str = TIME_WINDOW

print(f"PARENT_MODEL = {PARENT_MODEL}")
print(f"EXPERIMENT   = {EXPERIMENT}")
print(f"TIME_WINDOW  = {TIME_WINDOW} (slice {year_start_str}:{year_end_str})")

repo_root = Path(__file__).resolve().parent.parent
base_dir = repo_root / "preprocessed_inputs" / PARENT_MODEL / EXPERIMENT / TIME_WINDOW
monthly_dir = base_dir / "monthly"
yearly_dir = base_dir / "yearly"
makedirs(monthly_dir, exist_ok=True)
makedirs(yearly_dir, exist_ok=True)

print(f"Monthly output: {monthly_dir}")
print(f"Yearly output:  {yearly_dir}")

# ── Resolution-dependent chunk sizes ───────────────────────────────────────

if PARENT_MODEL == "ACCESS-OM2-1":
    CHUNKS_TY_RHO = {"time": -1, "potrho": 27, "grid_xt_ocean": 120, "grid_yu_ocean": 100}
elif PARENT_MODEL == "ACCESS-OM2-025":
    CHUNKS_TY_RHO = {"time": -1, "potrho": 40, "grid_xt_ocean": 120, "grid_yu_ocean": 108}
else:
    print(f"ERROR: Unknown PARENT_MODEL '{PARENT_MODEL}'; cannot determine chunk sizes", file=sys.stderr)
    sys.exit(1)

CHUNKS_TY_RHO_GM = CHUNKS_TY_RHO


# ── Helper functions (mirrored from periodicaverage.py) ─────────────────────

def select_data(cat, xarray_open_kwargs, **kwargs):
    selectedcat = cat.search(**kwargs)
    print(f"\nselectedcat: {selectedcat}")
    datadask = selectedcat.to_dask(
        xarray_open_kwargs=xarray_open_kwargs,
        xarray_combine_by_coords_kwargs=dict(
            compat="override",
            data_vars="minimal",
            coords="minimal",
        ),
        parallel=True,
    )
    return datadask


def month_climatology(ds):
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby("time.month") / month_length.groupby("time.month").sum()
    np.testing.assert_allclose(weights.groupby("time.month").sum().values, np.ones(12))
    ds_out = (ds * weights).groupby("time.month").sum(dim="time")
    mean_days_in_month = month_length.groupby("time.month").mean()
    ds_out = ds_out.assign_coords(mean_days_in_month=("month", mean_days_in_month.data))
    return ds_out


def weighted_yearly_mean(ds):
    month_length = ds.time.dt.days_in_month
    weights = month_length / month_length.sum()
    return (ds * weights).sum(dim="time")


def process_variable(searched_cat, varname, chunks):
    print(f"\n{'='*60}")
    print(f"Processing: {varname}")
    print(f"{'='*60}")

    datadask = select_data(
        searched_cat,
        dict(chunks=chunks),
        variable=varname,
        frequency="1mon",
    )
    print(f"\ndatadask: {datadask}")

    print(f"Slicing for time window {year_start_str}:{year_end_str}")
    datadask_sel = datadask.sel(time=slice(year_start_str, year_end_str))
    da = datadask_sel[varname]

    monthly_file = monthly_dir / f"{varname}_monthly.nc"
    print(f"Saving monthly climatology to: {monthly_file}")
    month_climatology(da).to_dataset(name=varname).to_netcdf(str(monthly_file), compute=True)

    yearly_file = yearly_dir / f"{varname}_yearly.nc"
    print(f"Saving yearly average to: {yearly_file}")
    weighted_yearly_mean(da).to_dataset(name=varname).to_netcdf(str(yearly_file), compute=True)

    print(f"Done: {varname}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_workers = int(os.environ.get("PBS_NCPUS", os.cpu_count() or 48))
    client = Client(n_workers=n_workers, threads_per_worker=1)
    print(f"Dask client: {client}")

    print("\nLoading intake catalog")
    catalogs = intake.cat.access_nri
    cat = catalogs[EXPERIMENT]

    searched_cat = cat.search(variable=["ty_trans_rho", "ty_trans_rho_gm"])
    print(searched_cat)

    process_variable(searched_cat, "ty_trans_rho", CHUNKS_TY_RHO)
    process_variable(searched_cat, "ty_trans_rho_gm", CHUNKS_TY_RHO_GM)

    print("\n" + "=" * 60)
    print("Density-space variables processed successfully")
    print(f"Monthly output: {monthly_dir}")
    print(f"Yearly output:  {yearly_dir}")
    print("=" * 60)

    client.close()
