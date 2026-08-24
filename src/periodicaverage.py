"""
Unified preprocessing script: compute monthly climatologies and yearly averages
from ACCESS-OM2 output for a given experiment and time window.

Environment variables:
    PARENT_MODEL   — ACCESS-OM2-1 or ACCESS-OM2-025 (default: ACCESS-OM2-1)
    EXPERIMENT     — intake catalog key (e.g. 1deg_jra55_iaf_omip2_cycle6)
    TIME_WINDOW    — year range "YYYY-YYYY" or single year "YYYY" (default: 1968-1977)

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

import dask
import distributed
from dask.distributed import Client
import intake
import netCDF4
import numpy as np
import xarray as xr


# ── Configuration ───────────────────────────────────────────────────────────
# Everything that reads the environment, prints, or creates directories lives
# in `configure()`, called only under `if __name__ == "__main__"`. It must NOT
# run at import time: dask spawns its workers by re-importing this file (as
# `__mp_main__`), so anything at module scope executes once per worker — which
# used to repeat the config banner 33 times in every log, and would have each
# worker independently re-run the validation and its `sys.exit()` paths.

DEFAULT_EXPERIMENTS = {
    "ACCESS-OM2-1": "1deg_jra55_iaf_omip2_cycle6",
    "ACCESS-OM2-025": "025deg_jra55_iaf_omip2_cycle6",
}


def configure():
    """Read config from the environment and set up output dirs. Main only."""
    global PARENT_MODEL, EXPERIMENT, TIME_WINDOW, VELOCITY_SOURCE
    global BUILD_TOTAL_TRANSPORT, DHT_CHECK, CALENDAR_YEAR_OFFSET
    global year_start_str, year_end_str, sel_start_str, sel_end_str
    global repo_root, base_dir, monthly_dir, yearly_dir

    PARENT_MODEL = os.environ.get("PARENT_MODEL", "ACCESS-OM2-1")
    EXPERIMENT = os.environ.get("EXPERIMENT", DEFAULT_EXPERIMENTS.get(PARENT_MODEL, ""))
    if not EXPERIMENT:
        print(f"ERROR: No default EXPERIMENT for {PARENT_MODEL}; set EXPERIMENT env var", file=sys.stderr)
        sys.exit(1)

    TIME_WINDOW = os.environ.get("TIME_WINDOW", "1968-1977")

    VELOCITY_SOURCE = os.environ.get("VELOCITY_SOURCE", "cgridtransports")
    if VELOCITY_SOURCE not in ("cgridtransports", "totaltransport"):
        print(f"ERROR: VELOCITY_SOURCE must be cgridtransports or totaltransport (got: {VELOCITY_SOURCE})", file=sys.stderr)
        sys.exit(1)
    BUILD_TOTAL_TRANSPORT = VELOCITY_SOURCE == "totaltransport"

    DHT_CHECK = os.environ.get("DHT_CHECK", "no").lower() in ("yes", "true", "1")

    # Parse TIME_WINDOW into start/end year strings. TIME_WINDOW is always given
    # in REAL (calendar) years and also names the output directory tree.
    if "-" in TIME_WINDOW:
        year_start_str, year_end_str = TIME_WINDOW.split("-", 1)
    else:
        year_start_str = year_end_str = TIME_WINDOW

    # CALENDAR_YEAR_OFFSET (Li et al.'s `ny`, default 0): some experiments label
    # the model calendar with a fixed offset from real years (real = labelled +
    # ny). The Qian/Li-et-al ACCESS-OM2-01 runs use ny = 1991 - 2100 = -109. The
    # raw catalog time axis is on the LABELLED calendar, so slice at
    # (real - ny) = labelled while keeping TIME_WINDOW (real) for output paths.
    CALENDAR_YEAR_OFFSET = int(os.environ.get("CALENDAR_YEAR_OFFSET", "0"))
    sel_start_str = f"{int(year_start_str) - CALENDAR_YEAR_OFFSET:04d}"
    sel_end_str = f"{int(year_end_str) - CALENDAR_YEAR_OFFSET:04d}"

    print(f"PARENT_MODEL        = {PARENT_MODEL}")
    print(f"EXPERIMENT          = {EXPERIMENT}")
    print(f"TIME_WINDOW         = {TIME_WINDOW} (real years {year_start_str}:{year_end_str})")
    print(f"CALENDAR_YEAR_OFFSET= {CALENDAR_YEAR_OFFSET} (labelled-calendar slice {sel_start_str}:{sel_end_str})")
    print(f"VELOCITY_SOURCE     = {VELOCITY_SOURCE} (BUILD_TOTAL_TRANSPORT={BUILD_TOTAL_TRANSPORT})")
    print(f"DHT_CHECK           = {DHT_CHECK}")

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

def chunk_sizes(parent_model):
    """Per-model dask chunk dicts. Pure — no printing, no side effects."""
    if parent_model == "ACCESS-OM2-1":
        # 1° grid: 360x300
        c = dict(
            CHUNKS_2D={"xt_ocean": 360, "yt_ocean": 300},
            CHUNKS_3D_T={"time": -1, "xt_ocean": 180, "yt_ocean": 150, "st_ocean": 25},
            CHUNKS_TX={"time": -1, "xu_ocean": 180, "yt_ocean": 150, "st_ocean": 25},
            CHUNKS_TY={"time": -1, "xt_ocean": 180, "yu_ocean": 150, "st_ocean": 25},
            CHUNKS_MLD={"time": -1, "xt_ocean": 360, "yt_ocean": 300},
            CHUNKS_DHT={"time": -1, "xt_ocean": 180, "yt_ocean": 150, "st_ocean": 25},
            CHUNKS_ETA={"time": -1, "xt_ocean": 360, "yt_ocean": 300},
        )
    elif parent_model == "ACCESS-OM2-025":
        # 0.25° grid: 1440x1080
        c = dict(
            CHUNKS_2D={"xt_ocean": 240, "yt_ocean": 216},
            CHUNKS_3D_T={"time": -1, "xt_ocean": 120, "yt_ocean": 108, "st_ocean": 25},
            CHUNKS_TX={"time": -1, "xu_ocean": 120, "yt_ocean": 108, "st_ocean": 25},
            CHUNKS_TY={"time": -1, "xt_ocean": 120, "yu_ocean": 108, "st_ocean": 25},
            CHUNKS_MLD={"time": -1, "xt_ocean": 240, "yt_ocean": 216},
            CHUNKS_DHT={"time": -1, "xt_ocean": 120, "yt_ocean": 108, "st_ocean": 25},
            CHUNKS_ETA={"time": -1, "xt_ocean": 240, "yt_ocean": 216},
        )
    elif parent_model == "ACCESS-OM2-01":
        # 0.1° grid: 3600x2700, st_ocean=75 — match native on-disk chunks:
        # 2D: 720x540 ; 3D: 180x135x19 (as inspected via `ncdump -hs` on cycle4)
        c = dict(
            CHUNKS_2D={"xt_ocean": 720, "yt_ocean": 540},
            CHUNKS_3D_T={"time": -1, "xt_ocean": 180, "yt_ocean": 135, "st_ocean": 19},
            CHUNKS_TX={"time": -1, "xu_ocean": 180, "yt_ocean": 135, "st_ocean": 19},
            CHUNKS_TY={"time": -1, "xt_ocean": 180, "yu_ocean": 135, "st_ocean": 19},
            CHUNKS_MLD={"time": -1, "xt_ocean": 720, "yt_ocean": 540},
            CHUNKS_DHT={"time": -1, "xt_ocean": 180, "yt_ocean": 135, "st_ocean": 19},
            CHUNKS_ETA={"time": -1, "xt_ocean": 720, "yt_ocean": 540},
        )
    else:
        print(f"ERROR: Unknown PARENT_MODEL '{parent_model}'; cannot determine chunk sizes", file=sys.stderr)
        sys.exit(1)
    c["CHUNKS_TX_GM"] = c["CHUNKS_TX"]
    c["CHUNKS_TY_GM"] = c["CHUNKS_TY"]
    return c


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


# ── Verified single-writer NetCDF output ────────────────────────────────────
# NEVER let dask.distributed workers write the output file.
#
# netCDF4/HDF5 (without parallel HDF5) does not support several processes
# holding the same file open for writing. `DataArray.to_netcdf()` on a
# dask-backed array under a distributed cluster does exactly that: the store
# tasks run on the workers, and each worker process re-opens the target file
# and writes its own chunks. HDF5 detects this and refuses the second open
# ("unable to lock file, errno = 11") — until file locking is disabled, at
# which point the writes race silently: whole dask chunks never land (they
# read back as _FillValue = NaN) and the occasional chunk is torn mid-write
# (garbage ~1e308). See docs/periodicaverage_corruption_bug.md.
#
# Instead: compute on the cluster, write from THIS process only, one slab at a
# time, and verify every slab twice —
#   1. mathematically, before writing: a weighted mean whose weights sum to 1
#      cannot exceed max|input|, and cannot be non-finite if the inputs are;
#   2. by reading it back afterwards, which proves the bytes actually landed.
# Any violation raises with the offending indices instead of silently
# producing a corrupt file.

def _verify_slab(name, label, out, in_absmax, in_nonfinite):
    """Check a computed slab against invariants its inputs guarantee."""
    n_bad = int(np.count_nonzero(~np.isfinite(out)))
    if n_bad > in_nonfinite:
        idx = np.argwhere(~np.isfinite(out))[:5].tolist()
        raise ValueError(
            f"{name}: {label} has {n_bad} non-finite values but its inputs had "
            f"{in_nonfinite}. First offending indices within the slab: {idx}"
        )
    finite = np.isfinite(out)
    if not finite.any():
        return
    bound = in_absmax * (1 + 1e-9) + 1e-30
    absmax = float(np.abs(out[finite]).max())
    if absmax > bound:
        idx = np.argwhere(finite & (np.abs(out) > bound))[:5].tolist()
        raise ValueError(
            f"{name}: {label} max|.| = {absmax:.6e} exceeds max|input| = "
            f"{in_absmax:.6e}; a weighted mean cannot do that. "
            f"First offending indices within the slab: {idx}"
        )


def write_verified(out_da, path, name, source_for_slab, slab_dim=None):
    """
    Write `out_da` to `path` under variable `name`, computing on the dask
    cluster but writing from this process only, verifying every slab.

    `source_for_slab(slab_value)` returns the lazy input data that the slab is
    derived from; its |max| and non-finite count are computed in the *same*
    dask call as the slab, so the raw data is read only once.
    """
    path = str(path)
    # Always start from scratch — never append to a stale or partial file.
    if os.path.exists(path):
        os.remove(path)

    # Create the file skeleton (dims, coords, attrs) from this process.
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        for dim, size in zip(out_da.dims, out_da.shape):
            nc.createDimension(dim, size)
        for cname, cvar in out_da.coords.items():
            vals = np.asarray(cvar.values)
            for dim, size in zip(cvar.dims, vals.shape):
                if dim not in nc.dimensions:
                    nc.createDimension(dim, size)
            cv = nc.createVariable(
                cname, vals.dtype, cvar.dims,
                fill_value=np.nan if vals.dtype.kind == "f" else None,
            )
            cv.setncatts({k: v for k, v in cvar.attrs.items() if k != "_FillValue"})
            cv[...] = vals
        # Keep _FillValue = NaN so that any region that is never written stays
        # loudly wrong rather than plausibly zero.
        var = nc.createVariable(
            name, out_da.dtype, out_da.dims,
            fill_value=np.nan if out_da.dtype.kind == "f" else None,
        )
        var.setncatts({k: v for k, v in out_da.attrs.items() if k != "_FillValue"})
        # Match what xarray would emit, so auxiliary coords (e.g.
        # mean_days_in_month) are still recognised as coords on read-back.
        aux = [c for c in out_da.coords if c not in out_da.dims]
        if aux:
            var.setncattr("coordinates", " ".join(aux))

    slab_values = out_da[slab_dim].values if slab_dim else [None]
    for i, slab_value in enumerate(slab_values):
        if slab_dim:
            out_lazy = out_da.isel({slab_dim: i})
            label = f"{slab_dim}={slab_value}"
        else:
            out_lazy = out_da
            label = "whole field"

        src = source_for_slab(slab_value)
        src_finite = np.isfinite(src)
        # Computed together so the shared source chunks are read exactly once.
        out_c, absmax_c, nbad_c = dask.compute(
            out_lazy,
            xr.where(src_finite, np.abs(src), 0.0).max(),
            (~src_finite).sum(),
        )
        out = np.asarray(getattr(out_c, "values", out_c))
        _verify_slab(name, label, out, float(absmax_c), int(nbad_c))

        with netCDF4.Dataset(path, "a") as nc:
            var = nc.variables[name]
            var.set_auto_mask(False)
            if slab_dim:
                var[i, ...] = out
            else:
                var[...] = out

        # Read back from disk to prove the write actually landed.
        with netCDF4.Dataset(path, "r") as nc:
            var = nc.variables[name]
            var.set_auto_mask(False)
            back = var[i, ...] if slab_dim else var[...]
        if not np.array_equal(back, out, equal_nan=True):
            differs = ~((back == out) | (np.isnan(back) & np.isnan(out)))
            idx = np.argwhere(differs)[:5].tolist()
            raise ValueError(
                f"{name}: {label} read back different from what was written "
                f"({int(differs.sum())} cells differ, first: {idx}). The output "
                f"file is being written by more than one process, or the "
                f"filesystem dropped the write."
            )
        print(f"  {label}: verified (max|.| = {float(absmax_c):.4e})")


def process_variable(searched_cat, varname, chunks, frequency="1mon",
                     is_time_invariant=False, save_as=None):
    """
    Load a variable, compute monthly climatology and yearly average, and save.

    For time-invariant variables (frequency='fx'), just save the raw field.
    Always overwrites existing files to avoid stale/corrupt data from failed runs.

    `save_as` overrides the on-disk name (filename prefix and the variable name
    inside the NetCDF). Used to alias e.g. `sea_level` → `eta_t` when the
    source experiment does not save `eta_t` directly.
    """
    save_name = save_as or varname
    print(f"\n{'='*60}")
    print(f"Processing: {varname}")
    print(f"{'='*60}")

    if is_time_invariant:
        # Time-invariant field: open directly from catalog path, bypassing
        # combine_by_coords which fails on static fields without dimension
        # coordinates (xarray >= 2025.03).
        selectedcat = searched_cat.search(variable=varname, frequency=frequency)
        filepath = selectedcat.df.path.iloc[0]
        print(f"Opening static field directly: {filepath}")
        ds = xr.open_dataset(filepath, chunks=chunks)
        # `.load()` first: a numpy-backed array is written by xarray from THIS
        # process (no dask store tasks, so no concurrent writers), which keeps
        # the field's original encoding — area_t marks land with a finite
        # _FillValue of 1e20, not NaN, and downstream readers expect that.
        da = ds[varname].load()
        outfile = base_dir / f"{varname}.nc"
        print(f"Saving {varname} to: {outfile}")
        da.to_netcdf(str(outfile))
        n_bad = int(np.count_nonzero(~np.isfinite(da.values)))
        print(f"Done: {varname} ({n_bad} non-finite; land sentinel is "
              f"{da.encoding.get('_FillValue', 'NaN')})")
        return

    datadask = select_data(
        searched_cat,
        dict(chunks=chunks),
        variable=varname,
        frequency=frequency,
    )
    print(f"\ndatadask: {datadask}")

    # Select time window (slice on the labelled calendar; see CALENDAR_YEAR_OFFSET)
    print(f"Slicing for real years {year_start_str}:{year_end_str} "
          f"(labelled-calendar slice {sel_start_str}:{sel_end_str})")
    datadask_sel = datadask.sel(time=slice(sel_start_str, sel_end_str))
    da = datadask_sel[varname]
    print(f"\n{varname} (sliced): {da}")

    # Monthly climatology → monthly/. Each month is written and verified
    # against only the timesteps of that month, which is also what makes the
    # per-slab bound check tight.
    monthly_file = monthly_dir / f"{save_name}_monthly.nc"
    print(f"Computing monthly climatology for {varname}")
    monthly = month_climatology(da)
    monthly.attrs = dict(da.attrs)
    print(f"Saving monthly climatology to: {monthly_file}")
    write_verified(
        monthly, monthly_file, save_name,
        lambda m: da.isel(time=(da["time.month"] == m).values),
        slab_dim="month",
    )

    # Yearly (time-window) average → yearly/
    yearly_file = yearly_dir / f"{save_name}_yearly.nc"
    print(f"Computing yearly average for {varname}")
    yearly = weighted_yearly_mean(da)
    yearly.attrs = dict(da.attrs)
    print(f"Saving yearly average to: {yearly_file}")
    write_verified(yearly, yearly_file, save_name, lambda _: da)

    print(f"Done: {varname}" + (f" (saved as {save_name})" if save_as else ""))


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configure()
    _chunks = chunk_sizes(PARENT_MODEL)
    CHUNKS_2D = _chunks["CHUNKS_2D"]
    CHUNKS_3D_T = _chunks["CHUNKS_3D_T"]
    CHUNKS_TX = _chunks["CHUNKS_TX"]
    CHUNKS_TY = _chunks["CHUNKS_TY"]
    CHUNKS_TX_GM = _chunks["CHUNKS_TX_GM"]
    CHUNKS_TY_GM = _chunks["CHUNKS_TY_GM"]
    CHUNKS_MLD = _chunks["CHUNKS_MLD"]
    CHUNKS_DHT = _chunks["CHUNKS_DHT"]
    CHUNKS_ETA = _chunks["CHUNKS_ETA"]

    # Record the environment: `conda/analysis3` is a rolling monthly release,
    # so without this the only way to tell which libraries produced a given
    # output is to fish site-packages paths out of incidental warnings.
    print("\n── environment ─────────────────────────────────────────────")
    print(f"  python       {sys.version.split()[0]}")
    print(f"  conda env    {os.environ.get('CONDA_PREFIX', '(unset)')}")
    for mod in (xr, dask, distributed, intake, netCDF4):
        print(f"  {mod.__name__:12s} {getattr(mod, '__version__', '?')}")
    print(f"  libnetcdf    {netCDF4.__netcdf4libversion__}")
    print(f"  libhdf5      {netCDF4.__hdf5libversion__}")
    print(f"  HDF5_USE_FILE_LOCKING={os.environ.get('HDF5_USE_FILE_LOCKING', '(unset)')}")
    print("─" * 60)

    # Dask distributed client is required for parallel NetCDF I/O.
    # Without it, dask falls back to the threaded scheduler and netCDF4
    # segfaults because it is not thread-safe. Workers READ the raw files
    # (many concurrent readers are fine); they must never WRITE the outputs —
    # see write_verified() above.
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

    # Detect the eta source: some experiments (e.g. 01deg_jra55v140_iaf cycles)
    # save `sea_level` = eta_t + patm/(rho0*g) instead of `eta_t`. We fall back
    # to sea_level and save it *as* eta_t — naive substitution; the error is the
    # inverse-barometer (IB) correction, whose spatial anomaly after monthly
    # averaging is ~±5 cm and seasonal cycle ~±2 cm, well below the ~70 cm std
    # of SSH. Small enough for age-tracer / transport / w-diagnostic work.
    # TODO: for an exact conversion, subtract the monthly climatology of JRA55
    # `psl` (at /g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/
    # atmos/3hr/psl/) regridded onto the ocean T grid and divided by rho0*g.
    cat_vars = {v for arr in cat.df["variable"] for v in arr}
    if "eta_t" in cat_vars:
        eta_source = "eta_t"
    elif "sea_level" in cat_vars:
        eta_source = "sea_level"
        print(f"WARNING: eta_t not in catalog for {EXPERIMENT}; "
              f"substituting sea_level (no IB correction).")
    else:
        print(f"ERROR: neither eta_t nor sea_level in catalog for {EXPERIMENT}",
              file=sys.stderr)
        sys.exit(1)

    # Search for all required variables
    all_variables = ["tx_trans", "ty_trans", "mld", "area_t", eta_source, "temp", "salt"]
    if BUILD_TOTAL_TRANSPORT:
        all_variables += ["tx_trans_gm", "ty_trans_gm"]
    if DHT_CHECK:
        all_variables += ["dht"]
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

    # Mass transports (resolved)
    process_variable(searched_cat, "tx_trans", CHUNKS_TX)
    process_variable(searched_cat, "ty_trans", CHUNKS_TY)

    # GM mass transports — only when total transport is requested
    if BUILD_TOTAL_TRANSPORT:
        process_variable(searched_cat, "tx_trans_gm", CHUNKS_TX_GM)
        process_variable(searched_cat, "ty_trans_gm", CHUNKS_TY_GM)

    # 2D / mixed-layer fields
    process_variable(searched_cat, "mld", CHUNKS_MLD)
    process_variable(searched_cat, eta_source, CHUNKS_ETA, save_as="eta_t")

    # dht — only when DHT_CHECK is enabled (sanity check in prep_velocities.jl)
    if DHT_CHECK:
        process_variable(searched_cat, "dht", CHUNKS_DHT)

    # Temperature and salinity (T-grid, 3D) for GM-Redi
    process_variable(searched_cat, "temp", CHUNKS_3D_T)
    process_variable(searched_cat, "salt", CHUNKS_3D_T)

    print("\n" + "=" * 60)
    print("All variables processed successfully")
    print(f"Monthly output: {monthly_dir}")
    print(f"Yearly output:  {yearly_dir}")
    print("=" * 60)

    client.close()
