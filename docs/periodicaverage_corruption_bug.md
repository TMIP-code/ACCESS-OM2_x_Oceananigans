# Bug — `periodicaverage.py` silently dropped dask chunks from every 0.1° output

**Status:** root cause **proven**; fix implemented in `src/periodicaverage.py` +
`scripts/prepreprocessing/periodicaverage.sh`. Remediation (re-runs) outstanding
— see §6.

## 1. Root cause

`DataArray.to_netcdf()` on a **dask-backed** array under a **`dask.distributed`
cluster** does not write from the client. The store tasks run on the workers,
and **each of the 48 worker processes re-opens the output file for writing**.

netCDF4/HDF5 (without parallel HDF5) does not support that. HDF5 detected it and
refused — prep job `166450245` (2026-04-19) died mid-write on
`tx_trans_monthly.nc` after 55 GB with:

```
H5Fint.c line 1910 in H5F_open(): unable to lock the file
H5FDsec2.c line 941 in H5FD__sec2_lock(): unable to lock file,
    errno = 11, error message = 'Resource temporarily unavailable'
```

`errno 11` (EWOULDBLOCK) from `flock(LOCK_EX|LOCK_NB)` means *another process
already holds the exclusive lock* — i.e. HDF5 correctly catching a second
concurrent writer. This was **not** an NFS problem (outputs go to `/scratch`,
Lustre, where `flock` works).

Commit `e9379e0` (2026-04-20) worked around it with
`export HDF5_USE_FILE_LOCKING=FALSE`, on the stated assumption that there was
"exactly one process per output file". That assumption was wrong, and disabling
the lock **converted a loud crash into silent data loss**: from then on the
worker processes raced freely, and every 0.1° monthly write lost whole dask
chunks. Jobs exited 0 and printed "All variables processed successfully".

## 2. Evidence (measured, not inferred)

Reference file:
`preprocessed_inputs/ACCESS-OM2-01/01deg_jra55v13_ryf9091_qian_wthp/2040-2050/monthly/ty_trans_monthly.nc`
(dask chunks: `st_ocean=19, yu_ocean=135, xt_ocean=180`).

- **Horizontal alignment is exact.** Tiling October, `k=59` by the 135×180 dask
  chunk grid: of 400 tiles, **311 are entirely NaN, 85 entirely finite, 4
  mixed**. Corruption is chunk-shaped, not cell-scattered.
- **Vertical alignment is exact.** The NaN fraction is *identical at every level
  within a z-chunk*: 0.0000 for `k=0–37` (z-chunks 0,1), exactly 0.0300 for all
  of `k=38–56` (z-chunk 2), ~0.78–0.80 for `k=57–74` (z-chunk 3). Whole 3D
  chunks are missing.
- **Garbage only in torn chunks.** The `~1e308` values occur *only* in the 4
  partially-written tiles (e.g. tile (9,4): 23168/24300 finite, max
  `1.566e308`) — a write interrupted mid-flight. Fully-missing chunks read back
  as `_FillValue = NaN`.
- **The arithmetic was never wrong.** A weighted mean cannot exceed max|input|,
  and the raw MOM data is clean (float32, max |ty| ~1e8, 0 non-finite). `1e308`
  is not even representable in float32.

This retires the earlier hypotheses — `compat="override"`, `parallel=True`, a
corrupt raw file, and the `groupby(...).sum()` weights are all **innocent**. So
are the grid/land-mask, the face-area metric, the zstar σ scaling, the B→C copy
kernel, and `create_velocities`/`prep_velocities`.

## 3. Blast radius (wider than first thought)

Corruption tracks **output size**, not experiment — it is not wthp-specific.
Measured with `src/audit_preprocessed.py` over all 52 OM2-01 files:

| Output | Size/var | Status |
|---|---|---|
| OM2-01 `…qian_wthp` / 2040-2050 monthly `temp,salt,tx_trans,ty_trans` | 70 GB | **CORRUPT** |
| OM2-01 `…qian_wthmp` / 2040-2050 monthly (same four) | 70 GB | **CORRUPT** |
| OM2-01 `…iaf_cycle4` / 1968-1977 monthly (same four) | 70 GB | **CORRUPT** |
| OM2-01 `…iaf_cycle4` / 1958-1987 monthly (same four) | 70 GB | **CORRUPT** — deleted, not re-run (non-default window) |
| OM2-01 monthly `eta_t`, `mld` | 0.9 GB | clean |
| OM2-01 `*_yearly.nc` (every level audited) | 5.8 GB | clean |
| OM2-01 `area_t.nc` | 10 MB | clean |
| OM2-025 and OM2-1 monthly + yearly (all) | ≤7.5 GB | clean |

**16 corrupt variables across 4 time-window sets.** Roughly **5–8 % of all
cells** are missing in each (sampling 4 of 75 levels: 25–38 million non-finite
values per variable). Damage is spread over *all twelve months*, not just
Oct/Nov — October–December are simply the worst.

Two corrections to earlier assumptions:
- `…qian_wthmp` was believed to be a clean control. **It is not.**
- `01deg_jra55v140_iaf_cycle4 / 1968-1977` fed the successful `upwind1` NK
  solve, so that result rests on corrupt input.

Only the 0.1° monthly 3D fields are affected: the yearly files (12× smaller,
so 12× fewer chunk writes) and everything at 1°/0.25° came through clean.

The environment (`conda/analysis3`) is **not** the trigger: 26.03 (IAF) and
26.07 (qian) both corrupt. The unpinned rolling module was, however, a real
reproducibility hazard and has been pinned.

## 4. Fix (implemented)

`src/periodicaverage.py` — new `write_verified()` replaces every `to_netcdf()`
call. It **computes on the cluster but writes from the client process only**,
one slab at a time (per month for climatologies, whole field for yearly), and
verifies each slab twice:

1. **Before writing** — a weighted mean with weights summing to 1 cannot exceed
   `max|input|` and cannot be non-finite where its inputs are finite. The input
   `absmax`/non-finite count are computed in the *same* `dask.compute()` call as
   the slab, so the raw data is still read only once.
2. **After writing** — the slab is read back from disk and compared bit-for-bit
   (`equal_nan=True`). This is what proves the bytes actually landed; it would
   have caught this bug on the first run.

Either check raises with the offending indices rather than producing a silently
bad file. `_FillValue = NaN` is kept deliberately so any never-written region
stays loudly wrong instead of plausibly zero.

`scripts/prepreprocessing/periodicaverage.sh` —
- `HDF5_USE_FILE_LOCKING=FALSE` **removed**. With a single writer the lock is
  never contended, and if a concurrent writer is ever reintroduced it will fail
  loudly again instead of corrupting.
- `conda/analysis3` pinned to `conda/analysis3-26.07` (the rolling alias is
  retargeted monthly).
- The script now prints python / xarray / dask / distributed / intake / netCDF4
  / libnetcdf / libhdf5 versions, so an output's provenance no longer has to be
  reconstructed from site-packages paths in incidental warnings.

`src/audit_preprocessed.py` (+ wrapper) — walks `preprocessed_inputs/` and flags
any variable with non-finite values or values beyond a physical bound. Sampling
one level per 19-level z-chunk is *complete* for detecting whole missing chunks;
`--full` checks every level. Exit status 1 on failure, so it can gate a re-run.

## 5. Do NOT

- **Do not** clip/zero corrupt values downstream in `prep_velocities.jl`. Its
  existing `map!(x -> isnan(x) ? 0 : x, …)` is a **land-mask** step (dry faces
  only); repurposing it to launder wet-cell garbage hides the bug and injects
  wrong physics.
- **Do not** re-enable `HDF5_USE_FILE_LOCKING=FALSE` to "fix" a lock error. That
  error is the safety net doing its job — find the second writer instead.

## 6. Remediation checklist (outstanding)

1. ~~Audit `preprocessed_inputs/ACCESS-OM2-01`~~ — done, results in §3. Still
   worth running over OM2-1/OM2-025 to confirm the whole tree.
2. Re-run `prep` (via `scripts/driver.sh`) for the three affected OM2-01 sets
   worth keeping: `…qian_wthp`, `…qian_wthmp` (both 2040-2050,
   `CALENDAR_YEAR_OFFSET=-109`), and `…iaf_cycle4` / 1968-1977 (the default
   window). Submitted 2026-08-19 as jobs `176678772`, `176679803`, `176679805`.
   The corrupt `.nc` files were first moved aside to
   `{TW}/nc_archive_pre_writefix_20260819/` so the new output can be diffed
   against them. `…iaf_cycle4` / 1958-1987 is a non-default window and was
   deleted outright rather than re-run.
3. Re-audit; then rebuild velocities and confirm with
   `scripts/debugging/probe_velocity_extremes.sh` (|u|,|v| ≲ a few m/s,
   |w| ≲ O(1e-2), 0 non-finite).
4. Rebuild the `upwind1` averaged transport matrix
   (`…/TM/…_upwind1_…/avg/M.jld2`) before re-submitting the forward NK for the
   qian experiments — see [Li-etal_simulations.md](../Li-etal_simulations.md) §7.
5. Re-run the IAF `upwind1` NK solve, whose published result currently rests on
   corrupt input.
6. Consider reporting upstream: `to_netcdf()` on a dask array under a
   distributed cluster silently losing chunks when HDF5 locking is disabled is a
   sharp edge others on Gadi will hit (the ACCESS-Hive forum has several threads
   about disabling HDF5 locking as a routine workaround).
