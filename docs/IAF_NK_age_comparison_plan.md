# IAF NK age comparison — analysis plan

Pickup doc for a fresh session. The four IAF NK periodic-age pipelines all finished with exit 0 on 2026-05-15 (see [IAF_simulations.md](IAF_simulations.md)). This plan turns the outputs into figures comparing time windows within a resolution (Phase 1), across resolutions at the same window (Phase 2), and characterising seasonality (Phase 3).

## Inputs (all confirmed present)

The 12-month periodic FieldTimeSeries is the output of `run1yrNK` (a 1-year simulation initialised from the NK steady solution). Located at:

```
outputs/{PM}/{EXP}/{TW}/periodic/{MC}/1year/Pardiso_LSprec/age_periodic_1year.jld2
```

with `MC = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12`. Specifically:

| Label | File | Size |
|---|---|---|
| OM2-1 / 1968-1977 (A1) | `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 1.4 GB |
| OM2-1 / 1999-2008 (B1) | `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 1.4 GB |
| OM2-025 / 1968-1977 (A025) | `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 19 GB |
| OM2-025 / 1999-2008 (B025) | `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 19 GB |

File format (`jldopen` keys): `grid`, `closure`, `serialized`, `timeseries` (with `timeseries["age"]` = the per-month groups + `timeseries["t"]`). Load as an Oceananigans `FieldTimeSeries` with `backend=InMemory()` for OM2-1 (1.4 GB), `backend=OnDisk()` for OM2-025 (don't put 90 GB in RAM — iterate months).

Age units: seconds. Convert to years with `/(365.25*86400)`. Tripolar grid: `size(grid)` includes the fold (Ny=Ny_interior+1); `interior(field)` excludes it.

## Reusable building blocks (in repo)

All in [src/shared_utils/analysis_and_plotting.jl](../src/shared_utils/analysis_and_plotting.jl):

| Function | Line | Purpose |
|---|---|---|
| `compute_ocean_basin_masks(grid, wet3D)` | 87 | Returns `(; ATL, PAC, IND)` 2D Bool masks via OceanBasins; handles tripolar fold |
| `zonalaverage(x3D, v3D, mask)` | 109 | Volume-weighted zonal average → `(Ny, Nz)`; NaN-safe |
| `zonalaverage!(za, xw, w, x3D, v3D, mask3D)` | 125 | In-place version with preallocated buffers |
| `find_nearest_depth_index(grid, target_depth)` | 145 | k-index for a given depth (m) |
| `plot_age_diagnostics(age_3D, grid, wet3D, vol_3D, out_dir, label; ...)` | 174 | 10 PNGs (4 zonal-avg contour + 6 horizontal slices at 100/200/500/1000/2000/3000 m) for a single field |

Other helpers:
- [src/shared_utils/grid.jl:627](../src/shared_utils/grid.jl#L627) `compute_wet_mask(grid)` → `(; wet3D, idx, Nidx)`
- [src/shared_utils/grid.jl:651](../src/shared_utils/grid.jl#L651) `compute_volume(grid)` → CenterField
- [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) — full reference for loading FTS, computing wet/vol, taking time mean, and calling `plot_age_diagnostics`

There is **no** existing A/B/B−A column comparison plotter, **no** profiles plotter, and **no** seasonality plotter. Add them as new functions in `analysis_and_plotting.jl` (keep public utilities in one file) and call them from a new driver script.

## Output convention

Land all comparison figures under a new tree (don't mix with single-run plots):

```
outputs/comparisons/NK_age/{MC}/
├── phase1_tw_OM2-1/         # within-OM2-1 TW comparison
├── phase1_tw_OM2-025/       # within-OM2-025 TW comparison
├── phase2_resolution_1968-1977/   # OM2-1 vs OM2-025 at 1968-1977
├── phase2_resolution_1999-2008/   # ditto at 1999-2008
├── phase3_seasonality/      # max−min over 12 months, per pipeline + diffs
└── phase3b_phase_shift/     # peak-month maps (deep-water formation regions)
```

Driver script: `src/compare_NK_ages.jl` (new). PBS wrapper if needed: `scripts/plotting/compare_NK_ages.sh` (model the existing `plot_1year_from_periodic_sol.sh`). Single CPU job (express, ~30 min) is plenty for OM2-1 maths; OM2-025 may want `hugemem` and OnDisk FTS iteration.

---

## Phase 1 — Annual mean within resolution (TW comparison)

**Goal**: A vs B diff plots for each resolution, using time-mean of the 12-month FTS as the annual mean age field.

### Two comparisons
1. **OM2-1**: A = TW 1968-1977, B = TW 1999-2008.
2. **OM2-025**: A = TW 1968-1977, B = TW 1999-2008.

### Compute

For each pipeline:
1. Load the 12-month FTS via `FieldTimeSeries(path, "age"; backend)`.
2. Accumulate time mean into a 3D `Array{FT,3}` of size `size(interior(field))` — month-by-month sum then divide by 12 (memory-safe for OM2-025).
3. Convert to years.

### Plots (per comparison)

**(a) Horizontal slices** — at depths 100, 200, 500, 1000, 2000, 3000 m (use `find_nearest_depth_index`). For each depth, one 1×3 figure with columns:

| col 1: A annual mean | col 2: B annual mean | col 3: B − A |
|---|---|---|

A and B share a colourbar (matched colorrange); the diff has its own diverging colourmap (e.g. `:balance`, symmetric range). Save as `slice_{depth}m.png`.

**(b) Basin zonal averages** — for each basin in {Global, ATL, PAC, IND}, one 1×3 figure (cols = A, B, B−A) of zonal-avg age vs (lat, depth). Use `zonalaverage` with the 2D basin mask. Save as `zonal_{basin}.png`.

**(c) Basin profiles** — for each basin in {Global, ATL, PAC, IND}, a single panel with profiles of both A and B overlaid (no diff). One 1×3 figure with one basin per column, or one 1×4 figure if Global is included. (User said 1×3; treat Global as a separate single panel.) Profiles = volume-weighted depth profile over the basin: `sum(age * vol * mask, dims=(1,2)) / sum(vol * mask, dims=(1,2))`. Save as `profiles_basins.png`.

### Suggested function signatures (add to analysis_and_plotting.jl)

```julia
plot_age_comparison_slice(A_3D, B_3D, grid, wet3D, out_dir;
                          label_A, label_B, depth_m, colorrange=:auto, diff_range=:auto)

plot_age_comparison_zonal(A_3D, B_3D, grid, wet3D, vol_3D, basin_mask, basin_label, out_dir;
                          label_A, label_B, colorrange=:auto, diff_range=:auto)

plot_age_profiles_basins(A_3D, B_3D, grid, vol_3D, basins, out_dir;
                         label_A, label_B)
# basins :: NamedTuple{(:ATL,:PAC,:IND), ...}, or :: NamedTuple{(:Global,:ATL,:PAC,:IND), ...}
```

Reuse `compute_ocean_basin_masks`, `zonalaverage`, `find_nearest_depth_index`.

### Verification
- Sanity-check time mean by computing it on a single 12-month FTS and visually comparing to one snapshot — should look similar but smoother.
- For colorrange, use `quantile(age_years[wet3D], [0.02, 0.98])` to avoid outliers.
- Diff range: `±maximum(abs.(B-A)) * 0.9` (or similar) so the diverging colourmap is centred.

---

## Phase 2 — Cross-resolution comparison (OM2-1 vs OM2-025)

**Goal**: Same A/B/B−A column plots for slices and zonal averages, but A = OM2-1 and B = OM2-025 at the same TW. Profiles can overlay without regridding (depths are identical; just plot both basin-mean profiles on the same axis).

### Two comparisons
1. **TW=1968-1977**: A = OM2-1, B = OM2-025.
2. **TW=1999-2008**: A = OM2-1, B = OM2-025.

### Regridding

Use [ConservativeRegridding.jl v0.2.1](https://github.com/JuliaGEO/ConservativeRegridding.jl) (added in commit `f1c108c`).

Direction (per user): regrid OM2-1 → OM2-025 grid (i.e. fine destination). Keep a flag to invert (coarser destination) — that's a more honest "diff at the coarse scale".

**Key references inside the package** (at `/g/data/y99/bp3051/.julia/packages/ConservativeRegridding/ShkeB/`):
- `README.md` — high-level API: `Regridder(dst, src)` then `regrid!(dst_field, R, src_field)`; `transpose(R)` for the reverse direction.
- `ext/ConservativeRegriddingOceananigansExt.jl` — `compute_cell_matrix(field_or_grid)` returns the polygon vertices used to build the regridder. Cells live in lat-lon coordinates; the extension already handles tripolar (incl. fold via `PaddedTreeWrapper`).
- `examples/oceananigans_new_api.jl` — start here for the simplest 2D Oceananigans → Oceananigans regridding workflow.
- `examples/oceananigans_longlat_finer.jl` — lower-level reference using spatial trees (`Trees.CellBasedGrid`, `dual_depth_first_search`). Useful if `Regridder` doesn't work out of the box on tripolar.

**Workflow (2D layer at a time)**:

```julia
using Oceananigans, ConservativeRegridding
import GeometryOps as GO

src_grid = ... # OM2-1 underlying grid (Center, Center, Center)
dst_grid = ... # OM2-025 underlying grid
src_cells = GO.UnitSphereFromGeographic().(compute_cell_matrix(src_grid))
dst_cells = GO.UnitSphereFromGeographic().(compute_cell_matrix(dst_grid))

R = ConservativeRegridding.Regridder(dst_cells, src_cells)  # build once, reuse for every layer

src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)

for k in 1:Nz
    set!(src_field, view(src_age_3D, :, :, k))     # OM2-1 layer
    regrid!(interior(dst_field, :, :, 1), R, interior(src_field, :, :, 1))
    dst_age_3D[:, :, k] .= interior(dst_field, :, :, 1)
end
```

Build the `Regridder` **once** (it's the expensive part — sparse intersection matrix). The actual per-layer `regrid!` is fast (sparse mat-vec).

Memory note: a 0.25° 50-layer 3D field at Float32 is ~310 MB, at Float64 ~620 MB. Two of them plus the regridder is fine on hugemem; even tight on normal.

### Plots

Same scheme as Phase 1, just with A=OM2-1, B=OM2-025:
- **Slices**: same 6 depths, 1×3 columns. Use the OM2-025 grid for the spatial axes.
- **Zonal averages**: regrid the full 3D age first, then run `zonalaverage` on the OM2-025 grid + basin masks computed on the OM2-025 grid.
- **Profiles**: load both 3D fields, compute basin-mean profiles at native resolution (no regridding needed since depths match — confirm `grid.z_faces` is identical for both PMs first), overlay.

### Sanity checks before drawing science conclusions
- After regridding, area-weighted means should be conserved: `sum(src_age * src_area) / sum(src_area) ≈ sum(dst_age * dst_area) / sum(dst_area)` per layer. The package guarantees this; verify on one layer with a printout.
- The tripolar fold can be a regridder corner case. If `Regridder` throws, fall back to the lower-level tree workflow in `oceananigans_longlat_finer.jl`.
- Cache the built `Regridder` to disk (`jldsave`) since rebuilding it is the dominant cost.

---

## Phase 3 — Seasonality

**Goal**: Quantify and map the 12-month variability of age. The user expects this to be small but wants to see how it changes across TW and resolution.

### Metric

For each grid cell: `seasonal_range = max(age[:,:,:,m]) - min(age[:,:,:,m]) for m in 1:12` (in years). This is the simplest and least assumption-heavy quantification of seasonality magnitude.

Optional alternative (mention in plan, don't implement yet): standard deviation over the 12 months, or amplitude of the first annual harmonic (FFT-based). Range is the most readable for a first pass.

### Pipelines and plots

Compute the seasonal-range 3D field once per pipeline (4 fields). Then re-run **all** Phase 1 + Phase 2 plot types with the seasonal-range field substituted for the annual-mean field:

- **Within OM2-1**: 1×3 column plots (TW=1968-1977 range, TW=1999-2008 range, diff). For slices and zonal averages.
- **Within OM2-025**: same.
- **Cross-resolution at each TW**: same (regridded via the same `Regridder` built in Phase 2).
- **Profiles**: overlays of seasonal-range vs depth, same structure as annual-mean profiles.

Place all under `outputs/comparisons/NK_age/{MC}/phase3_seasonality/`.

### Suggested additions

```julia
seasonal_range(fts::FieldTimeSeries) -> Array{FT,3}   # interior-sized, in years
# Implementation: stream months; track running min, max; subtract at end.
```

This is the only new "metric" computation. All plotting reuses Phase 1/2 functions.

---

## Phase 3b — Seasonality phase shift (optional)

**Goal**: Does the seasonal cycle peak earlier or later, depending on TW and resolution?

### Metric

Per cell: `peak_month = argmax_m age[:,:,:,m]` ∈ {1,…,12}. Interpret as month-of-year when age is largest. Use a circular colourmap (e.g. `:cyclic_mygbm_30_95_c78_n256` or a custom phase wheel).

For a more continuous metric: phase of the first annual harmonic via FFT — `angle(rfft(age_ts)[2])` (in radians), mapped to month. More robust to noise but more abstract; pick `peak_month` for the first pass.

### Where to focus

Deep-water formation regions (where seasonality is largest):
- North Atlantic Labrador / Irminger / GIN Seas (lat > 50°N in ATL)
- Southern Ocean Weddell / Ross / coastal Antarctic regions (lat < −60°S, all basins)

Plot:
- Peak-month maps at depth slices 100, 500, 1000 m (the surface signal mixed into the interior).
- Restrict colormap to those latitude bands so other regions don't drown the signal.

Diff = circular subtraction (`mod(peak_B - peak_A + 6, 12) - 6` → in {−6,…,+6}).

### Don't over-engineer

Phase 3b is optional. Implement only if Phase 3 reveals enough seasonal amplitude to make phase analysis interesting. Skip otherwise.

---

## Implementation order (recommended)

1. **Plumbing first**: write `compare_NK_ages.jl` skeleton that loads two FTS, computes time means, calls `plot_age_diagnostics` for each separately (verify loaders work end-to-end before adding new plotters).
2. **Add `plot_age_comparison_slice` and `plot_age_comparison_zonal`** — small, single-purpose. Test on OM2-1 TW pair first (fits in memory, fast).
3. **Add `plot_age_profiles_basins`** — simple overlay.
4. **Run Phase 1 for OM2-1 and OM2-025**. Inspect figures, iterate on colourranges.
5. **Phase 2: build `Regridder` for OM2-1 → OM2-025**. Verify mass conservation on a single layer. Cache to disk. Run cross-resolution Phase 1 plots.
6. **Phase 3: add `seasonal_range`** function. Re-run all Phase 1/2 plotters with this field. Same plot types.
7. **Phase 3b**: only if Phase 3 looks interesting.

Run Phase 1 OM2-1 entirely on the login node (data is small). Use a small PBS job for OM2-025 (hugemem, ~64 GB, ~30 min walltime). Phase 2 regridder build will need ~30–60 min CPU on hugemem.

## Open questions (resolve before implementing)

- **Time mean of FTS vs steady NK snapshot**: the `run1yrNK` FTS is a forward simulation from the NK steady state, so its time mean should be ≈ the NK steady periodic mean by construction. Worth verifying once (compare `mean(FTS)` against the contents of `…/NK/age_Pardiso_LSprec.jld2` for one pipeline). If they match, the steady NK file would be a faster Phase 1 input (single 3D array, no FTS iteration). If they differ meaningfully, the FTS is the right source.
- **Depth axis alignment for cross-resolution profiles**: confirm `grid.z_faces` is identical between OM2-1 and OM2-025 (both should be the standard ACCESS-OM2 50-level grid, but verify before overlaying). If they differ, interpolate one onto the other's depth axis before overlaying.
- **Plot output disk usage**: if seasonal-range diff fields and per-depth slices for OM2-025 produce too many PNGs, consider gating Phase 3 output behind a `SEASONALITY_DEPTHS=...` env var.

## Critical files / paths summary

| Purpose | Path |
|---|---|
| Plot utilities (extend) | [src/shared_utils/analysis_and_plotting.jl](../src/shared_utils/analysis_and_plotting.jl) |
| Reference for FTS load + time mean | [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) |
| Grid / wet mask / volume helpers | [src/shared_utils/grid.jl](../src/shared_utils/grid.jl) (`compute_wet_mask`, `compute_volume`) |
| ConservativeRegridding extension | `/g/data/y99/bp3051/.julia/packages/ConservativeRegridding/ShkeB/ext/ConservativeRegriddingOceananigansExt.jl` |
| Closest usage examples | `…/ConservativeRegridding/ShkeB/examples/oceananigans_new_api.jl`, `…/oceananigans_longlat_finer.jl` |
| New driver | `src/compare_NK_ages.jl` (to be created) |
| New PBS wrapper (optional) | `scripts/plotting/compare_NK_ages.sh` (model on `plot_1year_from_periodic_sol.sh`) |
| Outputs | `outputs/comparisons/NK_age/totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12/phase{1,2,3}/` |
