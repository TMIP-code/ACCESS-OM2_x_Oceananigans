# Bug plan — `periodicaverage.py` manufactures ~1e308 / NaN in the climatology

**Status:** root cause localized to `src/periodicaverage.py`; needs a proper fix
(NOT a downstream sanitize). Handoff plan for a focused debugging agent.

## 1. Symptom

The forward-map Newton–Krylov age solve for the OM2-01 experiment
`01deg_jra55v13_ryf9091_qian_wthp` (TIME_WINDOW `2040-2050`,
`CALENDAR_YEAR_OFFSET=-109` → labelled decade 2149-2159) blows up: `v`/`w` from
mass transport reach ~`1e298`, the age tracer goes NaN in Φ! call #1, and the NK
job then hangs (a distributed NaN-desync — separate issue).

Traced to a **corrupt preprocessed transport climatology**: the file
`preprocessed_inputs/ACCESS-OM2-01/01deg_jra55v13_ryf9091_qian_wthp/2040-2050/monthly/ty_trans_monthly.nc`
holds non-physical values at a small cluster of cells.

## 2. Confirmed facts (do not re-derive)

Reference cell (0-based) `xt_ocean=847, yu_ocean=1244, st_ocean=59` (~165°E,
equator, ~2830 m):

- **Preprocessed monthly climatology at that cell** (xarray, `decode_times=False`):
  months 1–9,12 are physical (−3.5e7 … +4.2e7 kg/s); **month 10 (Oct) =
  `1.56637e308`** (≈ Float64 max); **month 11 (Nov) = `NaN`**.
- **Raw MOM `ty_trans` at the same cell over the whole decade** (intake
  `access_nri`, `to_dask`, `.sel(time=slice("2149","2159"))`): 132 timesteps,
  dtype float32, **finite max |ty| = 1.03e8**, **0 non-finite, 0 values >1e12**.
  → the raw is clean; the climatology is not.
- A weighted mean cannot exceed its inputs, so **`periodicaverage.py`
  manufactured** the ~1e308/NaN from inputs ≤1e8. Not a data problem.
- **Cluster, not a single cell:** the velocity probe found **28 cells** with
  `|v|>50 m/s` and 1 catastrophic (`|v|=7.7e298`), all month 10; plus the Nov NaN.
- **Data-dependent:** the sibling experiment `…_qian_wthp`'s counterpart
  `…_qian_wthmp` runs the *identical code + identical grid* and is **completely
  clean** (v max 1.95, w max 0.017). So the bug is triggered by something in
  wthp's raw file/coordinate layout, not by the arithmetic alone.

Evidence scripts already in the tree (read-only probes, useful references):
`src/probe_velocity_extremes.jl`, `src/probe_v_blowup_cell.jl`,
`src/probe_face_area_metrics.jl` (+ wrappers in `scripts/debugging/`). Ruled out
(all confirmed innocent): the grid/land-mask, the Float32-vs-Float64 face-area
metric, the zstar σ scaling, the B→C copy kernel (`_copy_MOM_to_Oceananigans!`),
and `create_velocities`/`prep_velocities` itself.

## 3. Leading hypotheses (ranked)

The clean raw + corrupt climatology + only-Oct/Nov + wthp-only pattern points at
the **read/combine**, not the weighting math:

1. **`compat="override"` mis-combines overlapping/duplicate time coordinates**
   (`select_data`, [periodicaverage.py:136-145](../src/periodicaverage.py)).
   RYF/perturbation runs are often assembled from restart segments; if two raw
   files carry the same (or overlapping) timestamps, `combine_by_coords` with
   `compat="override"` silently picks/aligns without checking, and a
   misaligned/partial chunk can surface as garbage (~uninitialised ≈ 1e308) and
   NaN at specific month-cells. **Most likely, and explains wthp-vs-wthmp.**
2. **`parallel=True` + chunked read** leaving a chunk unfilled / racing
   (same call). Uninitialised dask/output memory reads as ~1e308 + NaN, which
   matches the "huge Oct + NaN Nov, clustered cells" signature.
3. **A specific corrupt/incomplete raw file** for one Oct (and Nov) in the decade
   that periodicaverage's chunked+override+parallel path reads but the default
   `to_dask(chunks={})` path (used in the raw check above) does not. Check file
   integrity for the offending month/year.
4. (Lower) The weighted `groupby("time.month").sum()` itself
   ([periodicaverage.py:149-154](../src/periodicaverage.py)) — unlikely since
   only 2 of 12 month-groups are corrupt and the `assert_allclose` on weights
   passes, but verify the per-group sums at the cell.

## 4. Cheap reproduction (do NOT run the full 12 h job)

Reproduce at a small spatial box so it's seconds, on a login/analysis node:

```python
import intake, numpy as np, xarray as xr
cat = intake.cat.access_nri
c = cat["01deg_jra55v13_ryf9091_qian_wthp"]

# periodicaverage's EXACT read config (chunks/override/parallel) vs a plain read:
CH = {"time": -1, "xt_ocean": 180, "yu_ocean": 135, "st_ocean": 19}
ds_pa = c.search(variable="ty_trans").to_dask(
    xarray_open_kwargs=dict(chunks=CH),
    xarray_combine_by_coords_kwargs=dict(compat="override", data_vars="minimal", coords="minimal"),
    parallel=True)
ds_plain = c.search(variable="ty_trans").to_dask(xarray_open_kwargs=dict(chunks={}))

for tag, ds in [("periodicaverage-config", ds_pa), ("plain", ds_plain)]:
    ty = ds["ty_trans"].sel(time=slice("2149","2159")).isel(xt_ocean=847, yu_ocean=1244, st_ocean=59).load()
    v = ty.values
    print(tag, "max|.|", np.nanmax(np.abs(v[np.isfinite(v)])), "n_nonfinite", np.sum(~np.isfinite(v)))
    # inspect the raw Oct/Nov timesteps specifically
    print("   Oct:", ty.sel(time=ty['time.month']==10).values)
    print("   Nov:", ty.sel(time=ty['time.month']==11).values)

# duplicate/overlap check on the time axis (hypothesis 1):
t = ds_pa["time"].values
print("n_time", t.size, "n_unique", np.unique(t).size, "→ duplicates:", t.size-np.unique(t).size)
```

Interpretation:
- If **periodicaverage-config raw is already corrupt** but **plain is clean** →
  hypothesis 1/2 (the read config). Then narrow: drop `compat="override"`, drop
  `parallel=True`, one at a time, and re-check.
- If **both raw reads are clean** but the **climatology** (apply
  `month_climatology`) is corrupt → hypothesis 4 (the groupby/weights); reproduce
  `(ds*weights).groupby("time.month").sum()` at the cell and bisect.
- If duplicates > 0 on the time axis → strong support for hypothesis 1; find which
  raw files overlap (`c.search(variable="ty_trans").df` → paths + time ranges).

## 5. Fix criteria / acceptance

- The October (and November) climatology at the reference cell must equal the
  **manual day-weighted mean of the clean raw** at that cell (~1e7, sign per the
  raw), and there must be **no wet-cell NaN / no |value| beyond a physical
  transport bound** anywhere in the output.
- Add a **post-condition assertion** in `periodicaverage.py` (fail loudly, do not
  silently clip): after computing each climatology/yearly field, assert
  `all wet-cell values finite` and `max|.| < PHYSICAL_BOUND` (e.g. transports
  < ~1e12 kg/s), erroring with the offending indices if violated. This converts
  the failure mode from silent corruption to a hard, located error.
- Re-run `prep` for wthp and confirm the velocity probe
  (`scripts/debugging/probe_velocity_extremes.sh`) reports physical maxima
  (|u|,|v| ≲ a few m/s, |w| ≲ O(1e-2)) with 0 non-finite.
- Regression-guard: the fix must leave `…_qian_wthmp` (already clean) and the IAF
  experiments bit-unchanged, or changed only in the previously-corrupt cells.

## 6. Do NOT

- **Do not** clip/zero corrupt values in `prep_velocities.jl`. The existing
  `map!(x -> isnan(x) ? 0 : x, …)` there is a **land-mask** step (dry faces only);
  repurposing it to launder wet-cell garbage hides the bug and injects wrong
  physics. Fix the source in `periodicaverage.py`.
- **Do not** re-run the full 12 h OM2-01 `prep` to iterate — reproduce at a small
  spatial box (§4) until the fix is proven, then do one clean full re-run.

## 7. Downstream once fixed

The corrupt velocities already fed **Task E** (the `upwind1` averaged transport
matrix, `…/TM/…_upwind1_…/avg/M.jld2`, built by `TMsnapshot`). After the
`periodicaverage.py` fix + clean `vel` re-run for wthp, **rebuild that avg matrix**
before re-submitting the forward NK. (wthmp is clean and can proceed independently
once its velocities are confirmed.) See the run-tracking table in
[Li-etal_simulations.md](../Li-etal_simulations.md) §7.

## 8. Key references

- `src/periodicaverage.py`: `select_data` (L132-146, the `to_dask` config),
  `month_climatology` (L149-154), `weighted_yearly_mean`, `process_variable`
  (slice + climatology + `to_netcdf`).
- Reproduction cell: `xt_ocean=847, yu_ocean=1244, st_ocean=59` (0-based), month
  10 (Oct) → 1.566e308, month 11 (Nov) → NaN.
- Clean control: `01deg_jra55v13_ryf9091_qian_wthmp` (same code, same grid).
