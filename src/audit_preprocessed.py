"""
Audit preprocessed NetCDF outputs for the missing-dask-chunk corruption
described in docs/periodicaverage_corruption_bug.md.

Symptom being detected: `to_netcdf()` on a dask-backed array under a
dask.distributed cluster let every worker process write the same HDF5 file.
Whole dask chunks never landed and read back as `_FillValue = NaN`; a few were
torn mid-write and read back as garbage (~1e308).

In a healthy file, land is stored as a finite value (0.0 for transports), so
the invariant is simply: **no non-finite values anywhere**, and no value beyond
a generous physical bound.

Usage:
    python3 src/audit_preprocessed.py [ROOT] [--full]

    ROOT    directory to walk (default: preprocessed_inputs/)
    --full  check every level (default samples one level per 19-level z-chunk,
            which is already complete for detecting whole missing chunks)

Exit status is 1 if any file is corrupt, so it can gate a re-run.
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

# Generous per-variable sanity bounds; the real detector is the non-finite
# count. These only need to be loose enough never to fire on good data and
# tight enough to catch manufactured garbage (~1e300).
BOUNDS = {
    "tx_trans": 1e12, "ty_trans": 1e12,      # kg/s   (raw max ~1e8)
    "tx_trans_gm": 1e12, "ty_trans_gm": 1e12,
    "temp": 1e3,                              # degC or K
    "salt": 1e3,                              # psu
    "mld": 1e5,                               # m
    "eta_t": 1e3, "sea_level": 1e3,           # m
    "area_t": 1e12,                           # m^2
}
DEFAULT_BOUND = 1e15

Z_CHUNK = 19  # st_ocean dask chunk used by periodicaverage.py at 0.1 deg


def land_sentinel(var):
    """
    The finite value this variable uses to mark land, or None.

    Raw MOM fields copied straight through (e.g. area_t.nc) mark land with a
    finite sentinel such as 1e20, which must be excluded from the checks.
    Fields written by periodicaverage.py instead carry `_FillValue = NaN` and
    store land as a real number, so a NaN fill value must NOT be masked out —
    NaN is precisely the corruption signal we are looking for.
    """
    for attr in ("_FillValue", "missing_value"):
        if attr in var.ncattrs():
            val = np.asarray(getattr(var, attr)).ravel()
            if val.size and np.isfinite(val[0]):
                return float(val[0])
    return None


def audit_variable(var, name, full=False):
    """Return (n_nonfinite, absmax, first_bad_indices) scanning 2D slices."""
    var.set_auto_mask(False)
    bound = BOUNDS.get(name, DEFAULT_BOUND)
    sentinel = land_sentinel(var)
    shape = var.shape
    n_bad = 0
    absmax = 0.0
    first_bad = []

    # Build the list of leading-index tuples whose 2D slices we will read.
    if len(shape) <= 2:
        leading = [()]
    elif len(shape) == 3:
        leading = [(i,) for i in range(shape[0])]
    elif len(shape) == 4:
        zs = range(shape[1]) if full else range(0, shape[1], Z_CHUNK)
        leading = [(i, k) for i in range(shape[0]) for k in zs]
    else:
        return None  # unsupported rank

    for idx in leading:
        sl = np.asarray(var[idx])
        land = (sl == sentinel) if sentinel is not None else np.zeros(sl.shape, bool)
        finite = np.isfinite(sl)
        bad = ~finite & ~land
        nb = int(np.count_nonzero(bad))
        if nb:
            n_bad += nb
            if len(first_bad) < 5:
                for pos in np.argwhere(bad)[: 5 - len(first_bad)]:
                    first_bad.append(tuple(idx) + tuple(int(p) for p in pos))
        usable = finite & ~land
        if usable.any():
            m = float(np.abs(sl[usable]).max())
            absmax = max(absmax, m)
            if m > bound and len(first_bad) < 5:
                for pos in np.argwhere(usable & (np.abs(sl) > bound))[:2]:
                    first_bad.append(tuple(idx) + tuple(int(p) for p in pos))

    return n_bad, absmax, first_bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    root = Path(args[0]) if args else Path("preprocessed_inputs")

    files = sorted(root.rglob("*.nc"))
    if not files:
        print(f"No .nc files under {root}")
        return 0
    print(f"Auditing {len(files)} file(s) under {root} "
          f"({'full' if full else 'sampled'} z scan)\n")

    n_corrupt = 0
    for path in files:
        try:
            ds = netCDF4.Dataset(path)
        except OSError as exc:
            print(f"[UNREADABLE] {path}: {exc}")
            n_corrupt += 1
            continue
        with ds:
            for name, var in ds.variables.items():
                if var.ndim < 2 or name in ds.dimensions:
                    continue
                result = audit_variable(var, name, full=full)
                if result is None:
                    continue
                n_bad, absmax, first_bad = result
                bound = BOUNDS.get(name, DEFAULT_BOUND)
                bad = n_bad > 0 or absmax > bound
                tag = "CORRUPT" if bad else "ok     "
                rel = path.relative_to(root) if path.is_relative_to(root) else path
                line = (f"[{tag}] {rel}::{name} "
                        f"non-finite={n_bad} max|.|={absmax:.4e}")
                if bad:
                    n_corrupt += 1
                    line += f"\n           first offending indices: {first_bad}"
                print(line)

    print()
    if n_corrupt:
        print(f"FAILED: {n_corrupt} corrupt variable(s) found.")
        return 1
    print("PASSED: all audited variables are finite and within bounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
