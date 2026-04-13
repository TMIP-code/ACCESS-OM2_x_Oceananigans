"""
Unified preprocessing script: compute monthly climatologies and yearly averages
from ACCESS-OM2 output for a given experiment and time window.

Environment variables:
    PARENT_MODEL   — ACCESS-OM2-1 or ACCESS-OM2-025 (default: ACCESS-OM2-1)
    EXPERIMENT     — intake catalog key (e.g. 1deg_jra55_iaf_omip2_cycle6)
    TIME_WINDOW    — year range "YYYY-YYYY" or single year "YYYY" (default: 1960-1979)

Output:
    preprocessed_inputs/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/monthly/*_monthly.nc
    preprocessed_inputs/{PARENT_MODEL}/{EXPERIMENT}/{TIME_WINDOW}/yearly/*_yearly.nc
"""

import sys
import os
import traceback
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

TIME_WINDOW = os.environ.get("TIME_WINDOW", "1960-1979")

# Parse TIME_WINDOW into start/end year strings for xarray slicing
if "-" in TIME_WINDOW:
    year_start_str, year_end_str = TIME_WINDOW.split("-", 1)
else:
    year_start_str = year_end_str = TIME_WINDOW

print(f"PARENT_MODEL = {PARENT_MODEL}")
print(f"EXPERIMENT   = {EXPERIMENT}")
print(f"TIME_WINDOW  = {TIME_WINDOW} (slice {year_start_str}:{year_end_str})")

# ── Output directories ─────────────────────────────────────────────────────

repo_root = Path(__file__).resolve().parent.parent
base_dir = repo_root / "preprocessed_inputs" / PARENT_MODEL / EXPERIMENT / TIME_WINDOW
monthly_dir = base_dir / "monthly"
yearly_dir = base_dir / "yearly"
makedirs(monthly_dir, exist_ok=True)
makedirs(yearly_dir, exist_ok=True)

print(f"Monthly output: {monthly_dir}")
print(f"Yearly output:  {yearly_dir}")

# ── Resolution-dependent chunk sizes ───────────────────────────────────────
# TODO: Chunk sizes could be auto-detected from the NetCDF files themselves
# (e.g. via ds[var].encoding['chunksizes'] or netCDF4.Dataset(path).variables[var].chunking()).
# For now, use hardcoded defaults that match the original per-model scripts.

if PARENT_MODEL == "ACCESS-OM2-1":
    # 1° grid: 360x300
    CHUNKS_2D = {"xt_ocean": 360, "yt_ocean": 300}
    CHUNKS_3D_T = {"time": -1, "xt_ocean": 180, "yt_ocean": 150, "st_ocean": 25}
    CHUNKS_3D_U = {"time": -1, "xu_ocean": 180, "yt_ocean": 150, "st_ocean": 25}
    CHUNKS_3D_V = {"time": -1, "xt_ocean": 180, "yu_ocean": 150, "st_ocean": 25}
    CHUNKS_3D_W = {"time": -1, "xt_ocean": 180, "yt_ocean": 150, "st_ocean": 25}
    CHUNKS_TX = {"time": -1, "xu_ocean": 180, "yt_ocean": 150, "st_ocean": 25}
    CHUNKS_TY = {"time": -1, "xt_ocean": 180, "yu_ocean": 150, "st_ocean": 25}
    CHUNKS_TX_GM = {"time": -1, "xu_ocean": 180, "yt_ocean": 150, "st_ocean": 25}
    CHUNKS_TY_GM = {"time": -1, "xt_ocean": 180, "yu_ocean": 150, "st_ocean": 25}
    CHUNKS_MLD = {"time": -1, "xt_ocean": 360, "yt_ocean": 300}
    CHUNKS_DHT = {"time": -1, "xt_ocean": 180, "yt_ocean": 150, "st_ocean": 25}
    CHUNKS_ETA = {"time": -1, "xt_ocean": 360, "yt_ocean": 300}
    CHUNKS_TY_RHO = {"time": -1, "potrho": 27, "grid_xt_ocean": 120, "grid_yu_ocean": 100}
    CHUNKS_TY_RHO_GM = {"time": -1, "potrho": 27, "grid_xt_ocean": 120, "grid_yu_ocean": 100}
elif PARENT_MODEL == "ACCESS-OM2-025":
    # 0.25° grid: 1440x1080
    CHUNKS_2D = {"xt_ocean": 240, "yt_ocean": 216}
    CHUNKS_3D_T = {"time": -1, "xt_ocean": 120, "yt_ocean": 108, "st_ocean": 25}
    CHUNKS_3D_U = {"time": -1, "xu_ocean": 120, "yt_ocean": 108, "st_ocean": 25}
    CHUNKS_3D_V = {"time": -1, "xt_ocean": 120, "yu_ocean": 108, "st_ocean": 25}
    CHUNKS_3D_W = {"time": -1, "xt_ocean": 120, "yt_ocean": 108, "st_ocean": 25}
    CHUNKS_TX = {"time": -1, "xu_ocean": 120, "yt_ocean": 108, "st_ocean": 25}
    CHUNKS_TY = {"time": -1, "xt_ocean": 120, "yu_ocean": 108, "st_ocean": 25}
    CHUNKS_TX_GM = {"time": -1, "xu_ocean": 120, "yt_ocean": 108, "st_ocean": 25}
    CHUNKS_TY_GM = {"time": -1, "xt_ocean": 120, "yu_ocean": 108, "st_ocean": 25}
    CHUNKS_MLD = {"time": -1, "xt_ocean": 240, "yt_ocean": 216}
    CHUNKS_DHT = {"time": -1, "xt_ocean": 120, "yt_ocean": 108, "st_ocean": 25}
    CHUNKS_ETA = {"time": -1, "xt_ocean": 240, "yt_ocean": 216}
    CHUNKS_TY_RHO = {"time": -1, "potrho": 40, "grid_xt_ocean": 120, "grid_yu_ocean": 108}
    CHUNKS_TY_RHO_GM = {"time": -1, "potrho": 40, "grid_xt_ocean": 120, "grid_yu_ocean": 108}
else:
    print(f"ERROR: Unknown PARENT_MODEL '{PARENT_MODEL}'; cannot determine chunk sizes", file=sys.stderr)
    sys.exit(1)


# ── Helper functions ────────────────────────────────────────────────────────

def select_data(cat, xarray_open_kwargs, **kwargs):
    """Search catalog and return lazy dask-backed dataset."""
    selectedcat = cat.search(**kwargs)
    print(f"\nselectedcat: {selectedcat}")
    xarray_combine_by_coords_kwargs = dict(
        compat="override",
        data_vars="minimal",
        coords="minimal",
    )
    datadask = selectedcat.to_dask(
        xarray_open_kwargs=xarray_open_kwargs,
        xarray_combine_by_coords_kwargs=xarray_combine_by_coords_kwargs,
        parallel=True,
    )
    return datadask


def month_climatology(ds):
    """Compute day-length-weighted monthly climatology (12 months)."""
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby("time.month") / month_length.groupby("time.month").sum()
    np.testing.assert_allclose(weights.groupby("time.month").sum().values, np.ones(12))
    ds_out = (ds * weights).groupby("time.month").sum(dim="time")
    mean_days_in_month = month_length.groupby("time.month").mean()
    ds_out = ds_out.assign_coords(mean_days_in_month=("month", mean_days_in_month.data))
    return ds_out


def weighted_yearly_mean(ds):
    """Compute day-length-weighted mean over the full time window."""
    month_length = ds.time.dt.days_in_month
    weights = month_length / month_length.sum()
    return (ds * weights).sum(dim="time")


def process_variable(searched_cat, varname, chunks, frequency="1mon",
                     is_time_invariant=False):
    """
    Load a variable, compute monthly climatology and yearly average, and save.

    For time-invariant variables (frequency='fx'), just save the raw field.
    Always overwrites existing files to avoid stale/corrupt data from failed runs.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {varname}")
    print(f"{'='*60}")

    datadask = select_data(
        searched_cat,
        dict(chunks=chunks),
        variable=varname,
        frequency=frequency,
    )
    print(f"\ndatadask: {datadask}")

    if is_time_invariant:
        # Time-invariant field: save once to monthly/ (no suffix)
        da = datadask[varname]
        outfile = monthly_dir / f"{varname}.nc"
        print(f"Saving {varname} to: {outfile}")
        da.to_netcdf(str(outfile), compute=True)
        print(f"Done: {varname}")
        return

    # Select time window
    print(f"Slicing for time window {year_start_str}:{year_end_str}")
    datadask_sel = datadask.sel(time=slice(year_start_str, year_end_str))
    da = datadask_sel[varname]
    print(f"\n{varname} (sliced): {da}")

    # Monthly climatology → monthly/
    monthly_file = monthly_dir / f"{varname}_monthly.nc"
    print(f"Computing monthly climatology for {varname}")
    monthly = month_climatology(da)
    print(f"Saving monthly climatology to: {monthly_file}")
    monthly.to_dataset(name=varname).to_netcdf(str(monthly_file), compute=True)

    # Yearly (time-window) average → yearly/
    yearly_file = yearly_dir / f"{varname}_yearly.nc"
    print(f"Computing yearly average for {varname}")
    yearly = weighted_yearly_mean(da)
    print(f"Saving yearly average to: {yearly_file}")
    yearly.to_dataset(name=varname).to_netcdf(str(yearly_file), compute=True)

    print(f"Done: {varname}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dask distributed client is required for parallel NetCDF I/O.
    # Without it, dask falls back to the threaded scheduler and netCDF4
    # segfaults because it is not thread-safe.
    n_workers = int(os.environ.get("PBS_NCPUS", os.cpu_count() or 48))
    client = Client(n_workers=n_workers, threads_per_worker=1)
    print(f"Dask client: {client}")

    # ── Load catalog ────────────────────────────────────────────────────

    print("\nLoading intake catalog")
    catalogs = intake.cat.access_nri
    print(catalogs)
    print(catalogs.keys())
    cat = catalogs[EXPERIMENT]
    print(cat)

    # Search for all required variables
    all_variables = [
        "u", "v", "wt", "tx_trans", "ty_trans", "tx_trans_gm", "ty_trans_gm",
        "ty_trans_rho", "ty_trans_rho_gm",
        "mld", "area_t", "dht", "eta_t",
        "temp", "salt",
    ]
    searched_cat = cat.search(variable=all_variables)
    print(searched_cat)

    # Find config.yaml by walking up from first catalog path
    _p = Path(searched_cat.df.path.iloc[0])
    while _p != _p.parent:
        _config = _p / "config.yaml"
        if _config.exists():
            print(f"\nFound config: {_config}")
            break
        _p = _p.parent
    else:
        print("\nconfig.yaml not found in any parent directory")

    # ── Process each variable ──────────────────────────────────────────

    # Time-invariant field
    process_variable(searched_cat, "area_t", CHUNKS_2D, frequency="fx", is_time_invariant=True)

    # 3D velocity fields
    process_variable(searched_cat, "u", CHUNKS_3D_U)
    process_variable(searched_cat, "v", CHUNKS_3D_V)
    process_variable(searched_cat, "wt", CHUNKS_3D_W)

    # Mass transports
    process_variable(searched_cat, "tx_trans", CHUNKS_TX)
    process_variable(searched_cat, "ty_trans", CHUNKS_TY)
    process_variable(searched_cat, "tx_trans_gm", CHUNKS_TX_GM)
    process_variable(searched_cat, "ty_trans_gm", CHUNKS_TY_GM)

    # Density-space mass transports (for density-space MOC)
    process_variable(searched_cat, "ty_trans_rho", CHUNKS_TY_RHO)
    process_variable(searched_cat, "ty_trans_rho_gm", CHUNKS_TY_RHO_GM)

    # 2D / mixed-layer fields
    process_variable(searched_cat, "mld", CHUNKS_MLD)
    process_variable(searched_cat, "dht", CHUNKS_DHT)
    process_variable(searched_cat, "eta_t", CHUNKS_ETA)

    # Temperature and salinity (T-grid, 3D) for GM-Redi
    process_variable(searched_cat, "temp", CHUNKS_3D_T)
    process_variable(searched_cat, "salt", CHUNKS_3D_T)

    print("\n" + "=" * 60)
    print("All variables processed successfully")
    print(f"Monthly output: {monthly_dir}")
    print(f"Yearly output:  {yearly_dir}")
    print("=" * 60)

    client.close()
