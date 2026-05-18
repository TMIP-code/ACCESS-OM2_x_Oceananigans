# IAF NK age comparison — analysis plan

Pickup doc for a fresh session. The four IAF NK periodic-age pipelines all finished with exit 0 on 2026-05-15 (see [IAF_simulations.md](IAF_simulations.md)). This plan turns the outputs into figures comparing time windows within a resolution (Phase 1), across resolutions at the same window (Phase 2), and characterising seasonality (Phase 3, with optional 3b).

## Inputs

The periodic `FieldTimeSeries` produced by `run1yrNK` (a 1-year simulation initialised from the NK steady solution):

```
outputs/{PM}/{EXP}/{TW}/periodic/{MC}/1year/Pardiso_LSprec/age_periodic_1year.jld2
```

with `MC = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12`. The four files:

| Label | File | Size |
|---|---|---|
| OM2-1 / 1968-1977 (A1) | `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 1.4 GB |
| OM2-1 / 1999-2008 (B1) | `outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 1.4 GB |
| OM2-025 / 1968-1977 (A025) | `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 19 GB |
| OM2-025 / 1999-2008 (B025) | `outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/$MC/1year/Pardiso_LSprec/age_periodic_1year.jld2` | 19 GB |

Load via `FieldTimeSeries(path, "age"; backend)` — `InMemory()` for OM2-1, `OnDisk()` for OM2-025. Use `Nt = length(fts.times)` (files may hold ~25 half-monthly snapshots, not 12). Age units seconds → years via `/(365.25*86400)`. Tripolar grid: `interior(field)` excludes the fold.

**NK file vs FTS**: the NK file is the t=0 snapshot Φ(x₀)=x₀ (saved by [src/solve_periodic_NK.jl:277](../src/solve_periodic_NK.jl#L277)); the FTS is the cycle starting from it. `mean(FTS) ≠ NK_file` by construction — use the FTS mean for annual means.

## Reusable building blocks (in repo)

All in [src/shared_utils/analysis_and_plotting.jl](../src/shared_utils/analysis_and_plotting.jl):

| Function | Line | Purpose |
|---|---|---|
| `compute_ocean_basin_masks(grid, wet3D)` | 87 | Returns `(; ATL, PAC, IND)` 2D Bool masks via OceanBasins; handles tripolar fold |
| `zonalaverage(x3D, v3D, mask)` | 109 | Volume-weighted zonal average → `(Ny, Nz)`; NaN-safe |
| `zonalaverage!(za, xw, w, x3D, v3D, mask3D)` | 125 | In-place version with preallocated buffers |
| `find_nearest_depth_index(grid, target_depth)` | 145 | k-index for a given depth (m) |
| `plot_age_diagnostics(age_3D, grid, wet3D, vol_3D, out_dir, label; ...)` | 174 | 10 PNGs (4 zonal-avg + 6 horizontal slices at 100/200/500/1000/2000/3000 m) for one field |

Other helpers:
- [src/shared_utils/grid.jl:627](../src/shared_utils/grid.jl#L627) `compute_wet_mask(grid)` → `(; wet3D, idx, Nidx)`
- [src/shared_utils/grid.jl:651](../src/shared_utils/grid.jl#L651) `compute_volume(grid)` → CenterField
- [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) — reference for loading FTS, computing wet/vol, time-mean, and calling `plot_age_diagnostics`

No A/B/B−A column plotter, no profiles plotter, no seasonality plotter exists. Add as new functions in `analysis_and_plotting.jl` and call from a new driver script.

## Output convention

```
outputs/comparisons/NK_age/{MC}/
├── phase1_tw_OM2-1/         # within-OM2-1 TW comparison
├── phase1_tw_OM2-025/       # within-OM2-025 TW comparison
├── phase2_resolution_1968-1977/   # OM2-1 vs OM2-025 at 1968-1977
├── phase2_resolution_1999-2008/   # ditto at 1999-2008
├── phase3_seasonality/      # max−min over cycle, per pipeline + diffs
└── phase3b_phase_shift/     # peak-snapshot maps (deep-water formation regions)
```

Driver: `src/compare_NK_ages.jl` (new). PBS wrapper: `scripts/plotting/compare_NK_ages.sh` (model on `plot_1year_from_periodic_sol.sh`).

---

## Phase 1 — Annual mean within resolution (TW comparison)

A vs B diff plots per resolution, using the FTS time-mean as the annual mean.

**Comparisons**:
1. **OM2-1**: A = TW 1968-1977, B = TW 1999-2008.
2. **OM2-025**: same.

**Compute** (per pipeline): load FTS, sum interior arrays across `Nt = length(fts.times)` snapshots, divide by `Nt`, convert to years. Memory-safe for OM2-025 via OnDisk iteration.

**Plots** (per comparison)

(a) **Horizontal slices** — depths 100, 200, 500, 1000, 2000, 3000 m. One 1×3 figure per depth, columns `A | B | B−A`. A/B share a colourbar; diff uses `:balance` symmetric range. Save as `slice_{depth}m.png`.

(b) **Basin zonal averages** — for {Global, ATL, PAC, IND}, 1×3 (`A | B | B−A`) of zonal-avg age vs (lat, depth). Save as `zonal_{basin}.png`.

(c) **Basin profiles** — single **1×4** figure (Global | ATL | PAC | IND) overlaying A and B per panel. Profile = `sum(age * vol * mask, dims=(1,2)) / sum(vol * mask, dims=(1,2))`. Save as `profiles_basins.png`.

**Suggested signatures** (add to `analysis_and_plotting.jl`):

```julia
plot_age_comparison_slice(A_3D, B_3D, grid, wet3D, out_dir;
                          label_A, label_B, depth_m, colorrange=:auto, diff_range=:auto)

plot_age_comparison_zonal(A_3D, B_3D, grid, wet3D, vol_3D, basin_mask, basin_label, out_dir;
                          label_A, label_B, colorrange=:auto, diff_range=:auto)

plot_age_profiles_basins(A_3D, B_3D, grid, vol_3D, basins, out_dir; label_A, label_B)
# basins :: NamedTuple{(:Global,:ATL,:PAC,:IND), ...}
```

Reuse `compute_ocean_basin_masks`, `zonalaverage`, `find_nearest_depth_index`. Colourrange via `quantile(age_years[wet3D], [0.02, 0.98])`; diff range `±0.9·maximum(abs.(B-A))`.

---

## Phase 2 — Cross-resolution comparison (OM2-1 vs OM2-025)

Same A/B/B−A scheme, A = OM2-1, B = OM2-025 (regridded) at the same TW.

**Comparisons**:
1. TW = 1968-1977.
2. TW = 1999-2008.

**Regrid direction**: **OM2-025 → OM2-1** (coarse destination — honest diff at the scale both grids resolve). Flag to flip if needed.

**Workflow** — canonical pattern from [test/usecases/oceananigans.jl](file:///g/data/y99/bp3051/.julia/packages/ConservativeRegridding/ShkeB/test/usecases/oceananigans.jl) / [examples/oceananigans_new_api.jl:32-38](file:///g/data/y99/bp3051/.julia/packages/ConservativeRegridding/ShkeB/examples/oceananigans_new_api.jl#L32-L38). Pass Oceananigans fields directly to `Regridder`; the extension's `treeify()` handles unit-sphere conversion and the tripolar fold internally.

```julia
using Oceananigans, ConservativeRegridding

# Build once. Expensive sparse-intersection step — jldsave the regridder and reuse across pipelines.
regridder = ConservativeRegridding.Regridder(CenterField(dst_grid),    # OM2-1  (coarse dst)
                                             CenterField(src_grid);    # OM2-025 (fine src)
                                             progress=true)

# Per-layer horizontal regrid. OM2-1 and OM2-025 share Nz=50, so k→k is identity.
Nz = size(src_age_3D, 3)
for k in 1:Nz
    regrid!(vec(view(dst_age_3D, :, :, k)),
            regridder,
            vec(view(src_age_3D, :, :, k)))
end
```

`regrid!` is a fast sparse mat-vec per layer. A 1° 50-level Float32 field is ~7 MB → fits on a normal node.

**Conservation check** (per-layer):

```julia
@assert sum(vec(view(dst_age_3D,:,:,k)) .* regridder.dst_areas) ≈
        sum(vec(view(src_age_3D,:,:,k)) .* regridder.src_areas)  rtol=1e-7
```

Tolerance `1e-7` matches the `RightCenterFolded` test (line 74). Use `transpose(regridder)` for the reverse.

**Plots**: Phase 1 scheme with A=OM2-1, B=OM2-025-regridded-to-OM2-1. Slices/zonal-avg on OM2-1 axes + masks. Profiles at native resolutions, overlay directly (depths identical — see open questions). If `Regridder` chokes on tripolar, fall back to `examples/oceananigans_longlat_finer.jl`.

---

## Phase 3 — Seasonality

**Metric**: per cell, `seasonal_range = max_m age[..,m] − min_m age[..,m]` over `m = 1:Nt`, in years.

Compute once per pipeline, then re-run **all** Phase 1 + Phase 2 plot types with `seasonal_range` substituted for the annual-mean field. Output under `…/phase3_seasonality/`.

```julia
seasonal_range(fts::FieldTimeSeries) -> Array{FT,3}   # interior-sized, in years
# Stream snapshots; track running min/max; subtract at end.
```

---

## Phase 3b — Seasonality phase shift (optional)

**Metric**: per cell, `peak_idx = argmax_m age[..,m]` ∈ {1,…,Nt}. Circular colourmap (e.g. `:cyclic_mygbm_30_95_c78_n256`); circular subtraction for diffs.

Focus on deep-water formation regions (ATL lat > 50°N, lat < −60°S all basins), peak-month maps at 100/500/1000 m. Only if Phase 3 shows meaningful amplitude.

---

## Implementation order

1. Skeleton `compare_NK_ages.jl`: load two FTS, time-mean each, call `plot_age_diagnostics` per pipeline (verify loaders).
2. Add `plot_age_comparison_slice` + `plot_age_comparison_zonal`; test on OM2-1 TW pair.
3. Add `plot_age_profiles_basins` (1×4 overlay).
4. Run Phase 1 for both resolutions; iterate on colourranges.
5. Phase 2: build + cache `Regridder` (OM2-025 → OM2-1); verify conservation on one layer; run cross-resolution plots.
6. Phase 3: add `seasonal_range`; re-run all Phase 1/2 plotters with it.
7. Phase 3b: only if Phase 3 amplitude warrants.

OM2-1 phases run on the login node. OM2-025 wants a small PBS job (normal queue, ~30 min, OnDisk FTS).

## Open questions

- **Depth axis alignment**: confirm `grid.z_faces` identical between OM2-1 and OM2-025 (both should be the standard ACCESS-OM2 50-level grid) before overlaying profiles. If different, interpolate one onto the other.
- **Phase 3 disk usage**: if seasonal-range diff fields at all 6 depths produce too many PNGs, gate behind a `SEASONALITY_DEPTHS=...` env var.

## Critical files / paths summary

| Purpose | Path |
|---|---|
| Plot utilities (extend) | [src/shared_utils/analysis_and_plotting.jl](../src/shared_utils/analysis_and_plotting.jl) |
| Reference for FTS load + time mean | [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) |
| Grid / wet mask / volume helpers | [src/shared_utils/grid.jl](../src/shared_utils/grid.jl) (`compute_wet_mask`, `compute_volume`) |
| ConservativeRegridding canonical test | `/g/data/y99/bp3051/.julia/packages/ConservativeRegridding/ShkeB/test/usecases/oceananigans.jl` |
| ConservativeRegridding canonical example | `…/ConservativeRegridding/ShkeB/examples/oceananigans_new_api.jl` |
| Lower-level fallback if `Regridder` struggles | `…/ConservativeRegridding/ShkeB/examples/oceananigans_longlat_finer.jl` |
| New driver | `src/compare_NK_ages.jl` (to be created) |
| New PBS wrapper (optional) | `scripts/plotting/compare_NK_ages.sh` (model on `plot_1year_from_periodic_sol.sh`) |
| Outputs | `outputs/comparisons/NK_age/totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12/phase{1,2,3}/` |
