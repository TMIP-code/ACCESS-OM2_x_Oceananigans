# Plan: ventilation-figure refactor (annual-mean, 2×2 panel, plotmap! tripolar mesh)

## Context

The surface ventilation diagnostic in this repo
([src/compute_ventilation_diagnostic.jl](../src/compute_ventilation_diagnostic.jl),
[src/plot_ventilation.jl](../src/plot_ventilation.jl)) computes
$\mathcal V^\downarrow = V \cdot \Gamma / (\tau\,A)$ from the annual mean
of a 1-year re-run of the converged NK solution. This second iteration
brings the figure up to the layout used in Pasquier *et al.* 2024
(JGR-Oceans, doi:10.1029/2024JC021043) and Pasquier *et al.* 2025 (GRL),
adapted to our (2 time-windows × 1 ocean) layout:

1. Each output PNG is a **2 × 2 panel** for one (parent-model, leg) pair:
   - `[1,1]` — `calVdown_norm` map at 1968-1977 (Pasquier 2024 orange ramp,
     pseudo-log scale, levels `[0, 10, 30, 100, 300, 1000]`,
     units `% v_tot / (10,000 km)²`).
   - `[1,2]` — zonal-integral side panel, both time windows overlaid as
     lines, units `% v_tot / °lat`.
   - `[2,1]` — decadal-difference map `(1999-2008 − 1968-1977)` on a
     diverging colour scale (`:tol_bu_rd`, symmetric, white at zero).
   - `[2,2]` — zonal integral of the difference, bicolour band split at 0.
2. Use **`plotmap!`** (per-cell quad mesh) from the ACCESS-TMIP plotting
   stack — same MOM5 tripolar grid that ACCESS-ESM1.5 (and ACCESS-OM2-1)
   uses, so no re-grid is needed and the tripolar fold is rendered faithfully.
3. Reuse the Pasquier 2024 colour ramp and `mk_piecewise_linear` scale so
   the map levels and ticks line up cleanly with the contour level set.

A later round will add seasonality (per-snapshot maps from the same FTS)
and any sub-basin Ω masks. **Not in this round.**

## Reference plotting code

The plotting style is lifted from two upstream files:

1. [/home/561/bp3051/Projects/TMIP/ACCESS-TMIP/src/plotting_functions.jl](/home/561/bp3051/Projects/TMIP/ACCESS-TMIP/src/plotting_functions.jl)
   — same MOM5 tripolar grid that ACCESS-OM2-1 uses (via ACCESS-ESM1.5),
   so the helpers port directly. Re-use:
   - `plotmap!(ax, x2D, gridmetrics; colorrange, colormap, highclip, lowclip, levels, colorscale)` — per-cell quad mesh from `(lon_vertices, lat_vertices)` via `mesh!`, plus coastlines.
   - `mk_piecewise_linear(vs)` — `ReversibleScale` whose forward map sends
     the user-supplied levels to integer pixel positions; cleaner than the
     fixed `myscale` from the Pasquier paper template because the colour-bar
     ticks land exactly on the level boundaries.
   - `lonticklabel` / `latticklabel` / `xtickformat` / `ytickformat`.
   - `divergingcbarticklabel` / `divergingcbarticklabelformat` — `+`/`−`
     sign-aware tick formatter for the diff panels.
   - `myhidexdecorations!` / `myhideydecorations!`.

2. [/home/561/bp3051/Projects/MatrixMarineCarbonCycleModel/src/plotting/paper3/plot_paper3_calVdown_maps_ZINT2.jl](/home/561/bp3051/Projects/MatrixMarineCarbonCycleModel/src/plotting/paper3/plot_paper3_calVdown_maps_ZINT2.jl)
   — supplies the colour-ramp recipe, the level set, and the bicolour band
   pattern for the diff zonal integral. Re-use:
   - `withwhitelow` colour-bar tweak (white at the bottom of the orange ramp).
   - `levelsPI = unique(Float32, clamp.([0; kron(10 .^ (1:3), [1, 3])], 0, 1000))` → `[0, 10, 30, 100, 300, 1000]`.
   - `dataforbicolorband(x, y1, y2)` — splits a band crossing zero into
     positive / negative segments so the zonal-integral-of-diff fill can be
     coloured by sign.
   - The mean / diff colour-bar label `% v_tot / (10,000 km)²`.

Skip from the templates:
- The `MAT.matread` data path (our inputs are JLD2).
- The Ω sub-basin mask annotations.
- The "EPAC" / "above500m" / "below2000m" sub-region selection.

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

$$\mathcal V^{\downarrow,\text{raw}}_{i,j} \;=\; \frac{V_{i,j,N_z}\,\bar\Gamma_{i,j}}{\tau\,A_{i,j,N_z}} \;=\; \frac{\Delta z_{N_z}\,\bar\Gamma_{i,j}}{\tau},$$

with $\tau = 3\,\Delta t$ in seconds and $\bar\Gamma_{i,j}$ in seconds (the NK
output is in seconds; `setup_model.jl:347` defines $\tau = 3\Delta t$). Units: m.

Normalisation to `% v_tot / (10,000 km)²` is applied in the **plot script**,
not in the compute script — the saved diagnostic JLD2 stays unit-neutral.
Note the parenthesisation: the normalising area is $(10^4\,\text{km})^2 = 10^{14}\,\text{m}^2$, **not** $10^4\,\text{km}^2 = 10^{10}\,\text{m}^2$:

$$\mathcal V^\downarrow_{i,j} \;=\; \mathcal V^{\downarrow,\text{raw}}_{i,j} \times \underbrace{\frac{10^{14}\,\text{m}^2}{(10{,}000\,\text{km})^2}}_{=1} \times \frac{100}{V_{\text{tot}}}\quad\bigl[\% \cdot (10{,}000\,\text{km})^{-2}\bigr]$$

i.e. multiply the raw m³/m² value by `1e14 * 100 / vtot = 1e16 / vtot`,
where $V_{\text{tot}} = \sum_{\text{wet cells}} V_{i,j,k}$ is the total
ocean volume (m³). $V_{\text{tot}}$ is computed in
`compute_ventilation_diagnostic.jl` and saved into `ventilation.jld2` as
the `vtot` key so the plot script doesn't recompute it.

The level set `[0, 10, 30, 100, 300, 1000]` was validated in Pasquier 2024
across sub-basin volumes spanning 2 orders of magnitude (upper-ETP vs deep-ETP).
Whole-ocean $V_{\text{tot}}$ rescales both numerator (sources like the
Southern Ocean / NA deep-water-formation regions are themselves larger
contributors at the global scale) and denominator proportionally, so the
level set is expected to remain useful for the global normalisation.

## Files to modify

### 1. [src/compute_ventilation_diagnostic.jl](../src/compute_ventilation_diagnostic.jl) — save raw only

The compute script already loads the 1-year FTS and time-averages the
surface layer. The only outstanding change is to **drop the
`% v_tot / (10,000 km)²` normalisation** from the saved file — the plot
script applies it. The `vtot` key stays so the plot script doesn't
recompute it.

- Remove the `calVdown_norm` assignment, NaN-mask, and `norm_factor`
  multiplication.
- Keep printing the normalised extrema / percentiles for logs (compute
  them locally just for the `@info` line).
- Delete the calibration `@warn` block (the level set is fine for global
  `vtot` — see the rationale above).
- `jldsave` keys become: `calVdown_raw, wet_surf, Az_surf, V_surf,
  age_surf, vtot, tau_seconds, n_avg, units, formula`. Update `units` to
  drop the `norm` entry; `formula` describes the raw form only.

### 2. [src/shared_utils/plotting_functions.jl](../src/shared_utils/plotting_functions.jl) — NEW helper file

Port (with minimal edits) the small set of plotting helpers from
[ACCESS-TMIP/src/plotting_functions.jl](/home/561/bp3051/Projects/TMIP/ACCESS-TMIP/src/plotting_functions.jl)
that we need:

- `plotmap!(ax, x2D, gridmetrics; …)` — per-cell quad mesh on a
  curvilinear grid via `mesh!`. `gridmetrics` is a NamedTuple with
  `lon, lat, lon_vertices, lat_vertices` (cell centres and 4-corner
  vertices, all 2-D).
- `lonticklabel`, `latticklabel`, `xtickformat`, `ytickformat`,
  `loninsamewindow`.
- `divergingcbarticklabel`, `divergingcbarticklabelformat`.
- `mk_piecewise_linear(vs)` — `ReversibleScale` mapping `vs[i]` → `i-1`.
- `myhidexdecorations!`, `myhideydecorations!`.
- `withwhitelow`, `withwhitecenter`, `dataforbicolorband` — supporting
  helpers from the Pasquier paper template.

Skip the `GeoMakie.land()` / `LibGEOS` / `GeometryOps` land-poly section
of the upstream file. Add coastlines via `lines!(ax, GeoMakie.coastlines(),
color = :black, linewidth = 0.85)` directly (only depends on `GeoMakie`).

### 3. [src/plot_ventilation.jl](../src/plot_ventilation.jl) — full rewrite as 2 × 2 panel

The script loads **both** time windows' `ventilation.jld2` (1968-1977 and
1999-2008), computes the decadal diff, and produces one PNG per
(parent-model, leg). Outline:

- Read env vars: `PARENT_MODEL`, `EXPERIMENT`, `VS/WF/AS/TS`, `LS`, `TRAF`.
  `TIME_WINDOW` is **not** read — both windows are hard-coded.
- Build path for each TW (mirror the dual-naming fallback `NK_Q2x2` →
  `NK` that the compute script uses); hard-error if either is missing.
- Load both files, assert grid shape and `vtot` agree (same grid, same
  wet mask), then `calV_tw = calVdown_raw_tw .* 1e16 / vtot` for each.
- Build a `gridmetrics` NamedTuple from the Oceananigans grid:
  - `lon = ug.λᶜᶜᵃ[1:Nx, 1:Ny]` (interior, trim halos)
  - `lat = ug.φᶜᶜᵃ[1:Nx, 1:Ny]`
  - `lon_vertices, lat_vertices` of shape `(4, Nx, Ny)` built from
    `ug.λᶠᶠᵃ`, `ug.φᶠᶠᵃ` (the 4 corner indices `(i,j), (i+1,j),
    (i+1,j+1), (i,j+1)`).
- Set up the mean colormap: `cm_mean = cgrad(withwhitelow(:Oranges),
  length(levels_mean); categorical=true)`, then peel off `highclip` and
  trim. Use `scale_mean = mk_piecewise_linear(Float32.(levels_mean))`
  as the `colorscale` so ticks land on level boundaries.
- Set up the diff colormap: `cm_diff` from `:tol_bu_rd` with the
  ACCESS-TMIP white-middle trick (peel off `low/highclip`, drop the
  middle band). `levels_diff` is a symmetric subset of `levels_mean`,
  e.g. `[-300, -100, -30, -10, 0, 10, 30, 100, 300]`.
- Draw the 4 panels into a `(4 rows × 2 cols)` layout where the colour
  bars sit in their own rows (rows 1 & 3 = colour bars, rows 2 & 4 = maps).
- `[1,1]` and `[2,1]` use `plotmap!`; `[1,2]` and `[2,2]` are ordinary
  `Axis` with `lines!` / `band!`. `linkyaxes!` the zonal panels to the
  matching map.
- Zonal integral: bin `(calV[i,j] * Az[i,j])` into 1° latitude bands
  using `lat2D[i,j]` and divide by `(10,000 km)² = 1e14 m²` to keep units
  consistent (`% v_tot / °lat`).
- Diff zonal integral uses `dataforbicolorband` to split into positive /
  negative segments coloured by sign.
- Save to `outputs/{PM}/{EXP}/plots/{MC}/calVdown_{leg}.png` (new
  experiment-level `plots/` dir, sibling to the per-TW `periodic/`,
  `standardrun/`, `TM/` trees). Use `dirname(outputdir)` (where
  `outputdir = outputs/{PM}/{EXP}/{TW}` from `load_project_config()`) to
  get the experiment level.

### 4. [Project.toml](../Project.toml) — add `GeoMakie`

Needed for `GeoMakie.coastlines()`. UUID:
`db073c08-6b98-4ee5-b6a4-5efafb3259c6`. After adding, run `]resolve` to
update `Manifest.toml`.

### 5. [scripts/solvers/compute_ventilation.sh](../scripts/solvers/compute_ventilation.sh) and [scripts/plotting/plot_ventilation.sh](../scripts/plotting/plot_ventilation.sh) — no change

Same PBS resources (`express`, 4 CPU, 24 GB, 30 min). The plot script
now loads two `ventilation.jld2` files (one per TW) rather than one;
still well under memory.

### 6. [docs/private/cross_resolution_ventilation_paper.md](private/cross_resolution_ventilation_paper.md) — update `Source*` paths and figure-caption units (DEFERRED)

The 2×2 tables in section 3.3 reference
`calVdown_{forward,adjoint}_lonlat.png`. Point them at the new
`calVdown_{forward,adjoint}.png` filenames once the re-plotting is done.
Update the caption text where it says "linear in metres" to the new
normalisation. **Defer this edit until after the plots are regenerated.**

## Verification (for the implementer)

This round writes code only; **do not run the PBS jobs**. The user will
re-run `ventilation` + `plotventilation` via `scripts/driver.sh` in a
follow-up session.

1. **Syntax check on `compute_ventilation_diagnostic.jl`** (single TW
   needed): `PARENT_MODEL=ACCESS-OM2-1 TIME_WINDOW=1968-1977
   VELOCITY_SOURCE=totaltransport TIMESTEPPER=SRK3 TIMESTEP_MULT=12
   MONTHLY_KAPPAV=yes TM_SOURCE=const julia --project
   src/compute_ventilation_diagnostic.jl`. Confirm `ventilation.jld2`
   has `calVdown_raw, vtot, …` and **no** `calVdown_norm`. Printed
   extrema finite.
2. **Syntax check on `plot_ventilation.jl`** — needs **both**
   `ventilation.jld2` files present (1968-1977 and 1999-2008). If they
   aren't yet, the script should fail cleanly pointing the user at
   driver.sh. Once both are present, run `julia --project
   src/plot_ventilation.jl` and confirm `calVdown_forward.png` is
   written into `outputs/{PM}/{EXP}/plots/{MC}/` with the 2×2 layout and
   no visible artifacts at the tripolar fold or Antarctic margin.
3. Commit:
   `ventilation: 2×2 layout (both windows), plotmap! tripolar mesh, raw-only save`
4. **Stop**; do not submit any PBS jobs and do not edit
   [docs/private/cross_resolution_ventilation_paper.md](private/cross_resolution_ventilation_paper.md).

## Out of scope (explicitly deferred)

- Seasonality plots (per-snapshot maps from the same FTS).
- Sub-basin Ω masks.
- Renaming legacy `LSprec` files to `Q2x2` (separately discussed; on hold).
