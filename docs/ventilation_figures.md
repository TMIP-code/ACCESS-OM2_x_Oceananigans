# Plan: ventilation-figure refactor (annual-mean, contourf, Pasquier 2024 normalisation)

## Context

The first cut of the surface ventilation diagnostic in this repo
([src/compute_ventilation_diagnostic.jl](../src/compute_ventilation_diagnostic.jl),
[src/plot_ventilation.jl](../src/plot_ventilation.jl)) computes
$\mathcal V^\downarrow = V \cdot \Gamma / (\tau\,A)$ from the **NK snapshot**
(`age_Pardiso_LSprec.jld2`), reports it in metres, and plots it with
`scatter!` on a linear colourbar. The user wants this replaced with the
treatment from Pasquier *et al.* 2024 (JGR-Oceans, doi:10.1029/2024JC021043):

1. Diagnostic is computed from the **annual mean** of a 1-year re-run from
   the converged NK solution, not the NK snapshot.
2. Plotting uses **`contourf!`** (not `scatter!`).
3. Units shift from m to **`% vtot / (10,000 km²)`** — fraction of the
   *total* ocean volume ventilated per 10,000 km² of surface — on a
   **semilog (pseudo-log)** colour scale with levels
   `[0, 10, 30, 100, 300, 1000]`.
4. Use the Pasquier 2024 plotting script as the visual reference, but
   adapted to our (2 time-windows × 1 ocean = 2) layout rather than the
   original (3 scenarios × 2 sub-volumes = 6). Skip everything related
   to coastlines, the Ω sub-basin mask, and the zonal-integral side
   panels — just panel (a) of the original layout, one PNG per case.

A later round will add seasonality (per-snapshot maps from the same FTS),
the (1999-2008 − 1968-1977) decadal-difference panel, and any
zonal-integral side panels. **Not in this round.**

## Reference template

[/home/561/bp3051/Projects/MatrixMarineCarbonCycleModel/src/plotting/paper3/plot_paper3_calVdown_maps_ZINT2.jl](/home/561/bp3051/Projects/MatrixMarineCarbonCycleModel/src/plotting/paper3/plot_paper3_calVdown_maps_ZINT2.jl)

Re-use directly:
- The `myscale = ReversibleScale(x -> sign(x) * log10(abs(x/5) + 1), x -> sign(x) * (exp10(abs(x)) - 1) * 5; limits=(0f0, 3f0), name=:myscale)` pseudo-log mapping.
- The `cbarticklabelformatPI` tick formatter.
- The `withwhitelow` colour-bar tweak for the preindustrial-style ramp.
- The `lonticklabel` / `latticklabel` axis-tick formatters.
- The `lon` axis duplication trick (`lonext = [lon; lon .+ 360]; ilonkeep = @. lonlims[1] ≤ lonext ≤ lonlims[2]`) so the map wraps cleanly past the dateline.
- Per the template: `levelsPI = unique(Float32, clamp.([0; kron(10 .^ (1:3), [1, 3])], 0, 1000))` — that gives exactly `[0, 10, 30, 100, 300, 1000]`.

**Skip from the template** (do not port):
- `coastlines.jld2` load and the `lines!(axmap, coastlinesext; ...)` calls (no coastlines this round — placeholder grey background only).
- `Ω2D` / `Ωline` annotations (no sub-basin mask — we plot the whole ocean).
- The zonal-integral side panel (`x1D`, `band!`, `lines!` on `axs`).
- The PI/RCP difference structure (`Delta_v2D`, `extendlow`, the two-colour difference colormap from `PRGn`).
- The `MAT.matread` data path — our inputs are JLD2.
- The "EPAC" / "above500m" / "below2000m" sub-region selection.

What survives in essence is **just panel (a) — a single contourf map of `calVdown_norm` on the pseudo-log scale** with the orange colour ramp from `withwhitelow(ColorSchemes.Oranges)`.

## Diagnostic — new definition

For each of the 8 available `(PM, TW, leg)` combinations
(`PM ∈ {ACCESS-OM2-1, ACCESS-OM2-025}`,
`TW ∈ {1968-1977, 1999-2008}`,
`leg ∈ {forward, adjoint}`), the production 1-year FTS lives at:

```
outputs/{PM}/{EXP}/{TW}/periodic/{MC}{,_traf}/1year/Pardiso_LSprec/age_periodic_1year.jld2
```

with `MC = totaltransport_wdiagnosed_centered2_SRK3_mkappaV_DTx12` (OM2-1)
or `…_DTx9` (OM2-025). All eight FTS files have been verified to exist on
disk.

The annual-mean surface age is

$$\bar\Gamma_{i,j} \;=\; \frac{1}{N-1}\sum_{n=1}^{N-1} \Gamma_{i,j,N_z}^{(n)},$$

averaging over the first $N-1 = 24$ of the 25 half-monthly snapshots
(skip the final snapshot — it is the periodicity check, redundant with
`n=1`; this mirrors [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) lines 159-165).

The raw per-cell ventilation volume per unit area (m³/m²) is then

$$\mathcal V^\downarrow_{i,j}^{\text{raw}} \;=\; \frac{V_{i,j,N_z}\,\bar\Gamma_{i,j}}{\tau\,A_{i,j,N_z}} \;=\; \frac{\Delta z_{N_z}\,\bar\Gamma_{i,j}}{\tau},$$

with $\tau = 3\,\Delta t$ in seconds and $\bar\Gamma_{i,j}$ in seconds (the NK
output is in seconds; `setup_model.jl:351` defines $\tau = 3\Delta t$). Units: m.

Normalisation to `% vtot / (10,000 km²)`:

$$\mathcal V^\downarrow_{i,j} \;=\; \mathcal V^\downarrow_{i,j}^{\text{raw}} \times \underbrace{\frac{10^{10}\,\text{m}^2}{10{,}000\,\text{km}^2}}_{=1} \times \frac{100}{V_{\text{tot}}}\quad\bigl[\% \cdot (10{,}000\,\text{km}^2)^{-1}\bigr]$$

i.e. multiply the raw m³/m² value by `1e10 * 100 / vtot = 1e12 / vtot`,
where $V_{\text{tot}} = \sum_{\text{wet cells}} V_{i,j,k}$ is the total
ocean volume (m³) computed once from the grid.

> **Calibration note for the implementer:** the global-ocean $V_{\text{tot}}$
> is ~1.3 × 10¹⁸ m³, much larger than any sub-basin used in Pasquier 2024,
> so the absolute %-values of `calVdown_norm` here will be much smaller
> than the EPAC values that motivated `[0, 10, 30, 100, 300, 1000]`. Print
> `extrema(calVdown_norm)` and the 50/90/99th percentiles from
> `compute_ventilation_diagnostic.jl` so the user can sanity-check the
> level set. If the data is consistently below the lowest level, leave a
> `@warn` note suggesting alternative level sets such as
> `[0, 0.01, 0.03, 0.1, 0.3, 1]` — but **keep the user-specified
> `[0, 10, 30, 100, 300, 1000]` as the default**.

## Files to modify

### 1. [src/compute_ventilation_diagnostic.jl](../src/compute_ventilation_diagnostic.jl) — rewrite the data source

Currently loads `NK/age_Pardiso_LSprec.jld2` (3-D snapshot). Change to:

- Load `1year/Pardiso_LSprec/age_periodic_1year.jld2` as a
  `FieldTimeSeries` (the loader pattern is in
  [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) line 99:
  `age_fts = FieldTimeSeries(output_filepath, "age")`).
- Time-average snapshots `1:(N-1)` of the surface layer
  (`age_fts[n]` returns a `Field`; use `interior(age_fts[n])[:, :, Nz]`
  inside a NaN-masked accumulator the same way
  [src/plot_periodic_1year_age.jl](../src/plot_periodic_1year_age.jl) lines
  159-165 does it for the whole 3-D volume).
- Compute `vtot = sum(interior(compute_volume(grid))[wet3D])`. The
  `compute_volume`, `compute_wet_mask`, and `load_tripolar_grid` helpers
  are all available via `include("shared_functions.jl")` (see
  [src/solve_matrix_age.jl](../src/solve_matrix_age.jl) line 132-157 for
  the pattern).
- Compute the raw m³/m² field and the normalised
  `% vtot / (10,000 km²)` field. Save both to `ventilation.jld2` so the
  plotting script can choose, with keys:
  - `calVdown_raw` — m³/m² (= m), same definition as the previous round.
  - `calVdown_norm` — `% vtot / (10,000 km²)`.
  - `wet_surf`, `Az_surf`, `vtot`, `tau_seconds`, `units` (descriptive),
    `formula` (string).

The path resolution (matching either the new `NK_Q2x2/` layout or the
legacy `NK/age_*_LSprec.jld2`) currently in
[src/compute_ventilation_diagnostic.jl](../src/compute_ventilation_diagnostic.jl) lines 81-99 stays — apply the same fallback logic to find the **`1year/{solver_tag}/`** dir.

### 2. [src/plot_ventilation.jl](../src/plot_ventilation.jl) — rewrite as a contourf map

Replace the current `heatmap!` / `scatter!` layout with a single contourf
map per (PM, TW, leg) using:

- `levelsPI = unique(Float32, clamp.([0; kron(10 .^ (1:3), [1, 3])], 0, 1000))`
  → `[0, 10, 30, 100, 300, 1000]`.
- `myscale = ReversibleScale(x -> sign(x) * log10(abs(x/5) + 1), x -> sign(x) * (exp10(abs(x)) - 1) * 5; limits=(0f0, 3f0), name=:myscale)`.
- `pseudologlevelsPI = myscale.(levelsPI)`.
- `colormapPI = cgrad(withwhitelow(ColorSchemes.Oranges), length(levelsPI); categorical=true)`; `extendhighPI = colormapPI[end]`; `colormapPI = colormapPI[1:end-1]`.
- `co = contourf!(ax, lonext[ilonkeep], lat, pseudologx2Dext[ilonkeep, :]; levels = pseudologlevelsPI, colormap = colormapPI, nan_color = :lightgray, extendhigh = extendhighPI)`.
- `Colorbar(...; ticks = (pseudologlevelsPI, string.(Int.(levelsPI))), label = rich("% v", subscript("tot"), " / (10,000 km)", superscript("2")))`.

**Tripolar caveat.** Our grid is tripolar — `λᶜᶜᵃ` and `φᶜᶜᵃ` are 2-D
matrices (not vectors), and `contourf!(ax, lon::Vector, lat::Vector, field::Matrix; ...)` won't work directly. Two options for the implementer to pick from, listed in preference order:

1. **Quick win for this round**: re-grid the field to a regular
   `(lon_1d, lat_1d)` grid using a one-shot nearest-neighbour or
   bilinear projection (e.g.
   [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
   or a simple `imresize`-style binning) before calling `contourf!`.
   This loses tripolar accuracy north of the fold but the surface
   ventilation diagnostic is dominated by Southern Ocean and North
   Atlantic features that are well-represented by a regular projection.
2. **Faithful**: keep the tripolar mesh and use
   `Makie.poly!` / `Makie.mesh!` with a per-cell colour, or
   `Makie.tricontourf!` on a Delaunay triangulation of the cell centres.
   This is what the existing
   [src/shared_utils/analysis_and_plotting.jl](../src/shared_utils/analysis_and_plotting.jl)
   helpers already do for some plots — see
   `animate_depth_slices` for the established pattern in this repo.

Recommend (1) for the first cut; revisit (2) if the user wants tripolar
fidelity.

Output filename: `calVdown_{forward,adjoint}_contourf.png`, alongside
the existing `calVdown_{forward,adjoint}_{ij,lonlat}.png` (keep those
for comparison; they can be removed later).

### 3. [scripts/solvers/compute_ventilation.sh](../scripts/solvers/compute_ventilation.sh) and [scripts/plotting/plot_ventilation.sh](../scripts/plotting/plot_ventilation.sh) — no change

Same PBS resources (`express`, 4 CPU, 24 GB, 30 min). The compute script
now loads a larger FTS but still fits comfortably; if memory becomes
tight, bump `mem` to 48 GB at OM2-025.

### 4. [docs/private/cross_resolution_ventilation_paper.md](private/cross_resolution_ventilation_paper.md) — update the `Source*` paths and the figure-caption units to `% vtot / (10,000 km²)`

The 2×2 tables in section 3.3 reference
`calVdown_{forward,adjoint}_lonlat.png`. Point them at the new
`calVdown_{forward,adjoint}_contourf.png` filenames once the
re-plotting is done. Update the caption text where it says "linear in
metres" to the new normalisation. **Defer this edit until after the
plots are regenerated** — the markdown links going stale for a few
hours is OK.

## Verification (for the implementer)

This round writes code only; **do not run the PBS jobs**. The user will
re-run `ventilation` + `plotventilation` via `scripts/driver.sh` in a
follow-up session. Concretely, after editing, the implementer should:

1. **Syntax check**: `julia --project -e 'include("src/compute_ventilation_diagnostic.jl")'`
   on a single test case (set the env vars manually, the same way the
   existing test in this session worked: `PARENT_MODEL=ACCESS-OM2-1
   TIME_WINDOW=1968-1977 VELOCITY_SOURCE=totaltransport TIMESTEPPER=SRK3
   TIMESTEP_MULT=12 MONTHLY_KAPPAV=yes TM_SOURCE=const julia
   --project src/compute_ventilation_diagnostic.jl`). Confirm
   `ventilation.jld2` is written with both `calVdown_raw` and
   `calVdown_norm` keys, and that the printed extrema are finite.
2. **Plot check**: same env vars, run `julia --project src/plot_ventilation.jl`.
   Confirm `calVdown_forward_contourf.png` is written and visually
   has a coherent surface-ventilation pattern (Weddell + Ross + NA peaks).
3. Commit the changes with a message like:
   `ventilation: annual-mean from 1-year FTS, contourf, %vtot/10000km² normalisation`
4. **Stop**; do not submit any PBS jobs and do not edit
   [docs/private/cross_resolution_ventilation_paper.md](private/cross_resolution_ventilation_paper.md).
   The user will pick up from here.

## Out of scope (explicitly deferred)

- Seasonality plots (per-snapshot maps from the same FTS).
- Decadal-difference panel: `(1999–2008) − (1968–1977)` for $\mathcal V^\downarrow$.
- Zonal-integral side panel (the `x1D` band/lines in the template).
- Coastlines.
- Sub-basin Ω masks.
- Tripolar-faithful rendering (use the regular-grid re-grid for now).
- Renaming legacy `LSprec` files to `Q2x2` (separately discussed; on hold).
