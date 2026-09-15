# Plan — Equilibrium ventilation of the Li et al. (2023) meltwater experiments

Apply the periodic Newton-Krylov (NK) age framework to the two ACCESS-OM2-01
(0.1°) "Qian" perturbation experiments (§3.4 of
[docs/DWF_candidates.md](docs/DWF_candidates.md)) to isolate the **meltwater**
contribution to deep-ocean ventilation. Compute, for each experiment:
**(1) equilibrated ideal age** (forward NK), **(2) time to re-emergence**
(adjoint age via TRAF), and **(3)** the `wthmp − wthp` differences that isolate
meltwater.

| tag | experiment | forcing perturbation |
|-----|------------|----------------------|
| **wthmp** | `01deg_jra55v13_ryf9091_qian_wthmp` | Wind + Thermal + **Meltwater** (full projected AABW decline) |
| **wthp**  | `01deg_jra55v13_ryf9091_qian_wthp`  | Wind + Thermal only (no meltwater) |

Meltwater effect = **`wthmp − wthp`**. The RYF control `01deg_jra55v13_ryf9091`
is *not* required for this comparison (optional extension only).

Reference: Li et al. (2023), *Nature*, doi:10.1038/s41586-023-05762-w — code:
<https://github.com/QianLi-Ocean/Antarctic_MWdriven_Abyssal_Circulation_Change>.

This plan builds directly on the **proven OM2-01 forward recipe** in
[docs/OM2-01_upwind3_NK_solve.md](docs/OM2-01_upwind3_NK_solve.md) (converged at
0.1°, `vol_rms_drift = 3.4e-9`) and the TRAF machinery in
[docs/TRAF_simulations.md](docs/TRAF_simulations.md).

---

## 1. Data situation — verified (2026-08-10)

Both experiments are **already fully wired into the pipeline** — no new
data-location plumbing:

- In the ACCESS-NRI intake catalog (`intake.cat.access_nri`), keyed by the exact
  experiment names → `periodicaverage.py` loads them by `EXPERIMENT`, unchanged.
- Already registered in [`ACCESS-OM2_configs.yaml`](ACCESS-OM2_configs.yaml)
  (lines 9–10) → `create_grid.jl` finds the grid/bathymetry inputs.
- All required vars present at monthly (`1mon`): `tx_trans, ty_trans, temp, salt,
  mld, area_t`, and **`sea_level`** (not `eta_t`). The `sea_level → eta_t`
  fallback already exists ([periodicaverage.py:262-272](src/periodicaverage.py));
  acceptable (no inverse-barometer correction) for age/transport work.
- Raw output under `/g/data/cj50/access-om2/raw-output/access-om2-01/…` (the
  "separate directories", reached transparently via the catalog).
- Grid/bathymetry identical across all RYF-9091 0.1° experiments (forcing-only
  perturbations) → build the grid once and symlink (§3.1).

### 1.1 The calendar offset — the one thing to cater for

The model output is labelled with a shifted calendar. Li et al.'s own
`Figures/Figure_3.ipynb` defines:

```python
ny   = 1991 - 2100      # = -109   (real = labelled + ny)
YEAR = [2041, 2050]     # real years analysed
#   .sel(year=slice(YEAR[0]-ny, YEAR[1]-ny))  → labelled 2150–2159
```

Offset = **−109 yr** (real = labelled − 109). Verified: **every** required
variable in both experiments shares identical labelling (no per-variable glitch —
`tx_trans` is *not* mis-formatted):

| experiment | labelled | **real** |
|------------|----------|----------|
| wthp  | 2100–2159 | **1991–2050** |
| wthmp | 2110–2159 | **2001–2050** |

**Analysis window (decided): the last decade, real 2040–2050 = labelled
2149–2159.** Both experiments cover it, so a single window gives a fair,
identically-timed comparison (≈ Li et al. Fig. 3's real 2041–2050).

`periodicaverage.py` slices raw catalog time with
`.sel(time=slice(year_start_str, year_end_str))`
([periodicaverage.py:213](src/periodicaverage.py)) against the **labelled**
calendar — the *only* place raw years are used (everything downstream reads
calendar-free climatology files).

**→ Task A (enabling):** add an optional `CALENDAR_YEAR_OFFSET` env var
(default `0`, matching Li's `ny`; set `-109` here) so `TIME_WINDOW` is given in
**real** years but the slice targets the labelled calendar:

```python
NY = int(os.environ.get("CALENDAR_YEAR_OFFSET", "0"))   # -109 for Qian
sel_start = f"{int(year_start_str) - NY:04d}"           # real - ny = labelled
sel_end   = f"{int(year_end_str)   - NY:04d}"
datadask_sel = datadask.sel(time=slice(sel_start, sel_end))
```

Thread it through [env_defaults.sh](scripts/env_defaults.sh) +
[driver.sh](scripts/driver.sh) `COMMON_VARS` (qsub `-v`). Then
`TIME_WINDOW=2040-2050 CALENDAR_YEAR_OFFSET=-109` selects labelled 2149–2159 and
lands outputs under a **real-year** path (`.../2040-2050/…`) so all directories
and plots carry correct science labels. (Zero-code fallback: use
`TIME_WINDOW=2149-2159` labelled and relabel plots — rejected, mislabels the
tree.)

---

## 2. Shared configuration

Nothing in the grid/velocity/matrix/solver code is experiment-specific: it is all
parameterized by `EXPERIMENT` + `TIME_WINDOW` + `PARENT_MODEL=ACCESS-OM2-01`. The
per-model defaults (`cgridtransports`, `PARTITION=1x4`, `Q4x4`, κH30,
`gpuhopper`, megamem partition, 48 h NK, `TIMESTEP_MULT=1`) come from
[model_configs/ACCESS-OM2-01.sh](model_configs/ACCESS-OM2-01.sh).

**The proven OM2-01 forward recipe** (decoupled forward map / preconditioner):

- **Forward map** `ADVECTION_SCHEME=upwind3` — less numerical diffusion.
- **Preconditioner** `TM_ADVECTION_SCHEME=upwind1` + `TM_SOURCE=avg` — reuses the
  already-built `upwind1` **averaged** transport matrix (its implicit diffusion
  conditions the Jacobian well). Only `avg` exists at OM2-01; there is no `const`.
- **`GRID_HZ=4`** — `UpwindBiased(order=3)` needs grid z-halo ≥ 3; the default
  `GRID_HZ=2` fails at model build. Rebuild grid → vel → clo → partition at
  `GRID_HZ=4` (harmless larger halo; also serves WENO5).
- `W_FORMULATION=wprescribed` + `PRESCRIBED_W_SOURCE=parent` (model default) → no
  `diagnose_w` step (forward map reads the parent `w`).

Common env block used below:

```bash
PARENT_MODEL=ACCESS-OM2-01
TIME_WINDOW=2040-2050
CALENDAR_YEAR_OFFSET=-109
GRID_HZ=4
ADVECTION_SCHEME=upwind3
TM_ADVECTION_SCHEME=upwind1
TM_SOURCE=avg
```

---

## 3. Forward ideal age — per experiment

For `EXPERIMENT ∈ {…_qian_wthp, …_qian_wthmp}`. The distributed chain is
`grid → vel → clo → partition → NK`; the `upwind1` avg matrix (preconditioner)
must exist before NK.

### 3.1 Preprocess + grid + velocities (once per experiment; grid shared)

```bash
# Python preprocessing (monthly climatology + yearly mean of the analysis decade)
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 CALENDAR_YEAR_OFFSET=-109 \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=prep bash scripts/driver.sh

# Grid + velocities + closures at GRID_HZ=4 (serial; grid built once)
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 CALENDAR_YEAR_OFFSET=-109 GRID_HZ=4 \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=grid-vel-clo bash scripts/driver.sh
```

**Grid reuse.** The two Qian experiments (and the control) share an identical
ocean grid. Build `grid.jld2` for `wthp`, then for `wthmp` either symlink it and
pre-set `GRID_JOB` to skip the grid step (see [CLAUDE.md](CLAUDE.md) § driver.sh
chaining), or just rebuild it (safe, one grid job). Verify the two configs point
at the same bathymetry/vgrid before symlinking.

### 3.2 Build the `upwind1` averaged matrix (preconditioner) — the expensive step

Run **once per experiment** for this window (≈39 GB, megamem; this is the costly
part, not the NK):

```bash
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind1 TM_SOURCE=avg LUMP_AND_SPRAY=4x4 \
TMSNAP_QUEUE=megamem TMSNAP_MEM=2990GB TMSNAP_NCPUS=48 \
WALLTIME_TM_SNAPSHOT=24:00:00 SAVE_INTERMEDIATE_MATRICES=no \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=TMsnapshot bash scripts/driver.sh
```

Produces `…/TM/cgridtransports_wparent_upwind1_AB2_kH30_…_LBS/avg/M.jld2`.

### 3.3 Partition + NK (forward age)

```bash
# partition (megamem) then NK. GRID_HZ=4 forces the partition redo.
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=partition-NK bash scripts/driver.sh
```

**Multi-restart.** Each 48 h GPU job advances ≈1 Newton iteration (`upwind3` ≈2×
the JVPs of `upwind1`), so a full solve spans several restarts.
`TRACE_SOLVER_HISTORY=yes` (default) saves `newton_iterate_NN.jld2`; resume with
`INITIAL_AGE=latest`:

```bash
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
INITIAL_AGE=latest \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=NK bash scripts/driver.sh
```

Restart protocol (from the OM2-01 doc): **walltime kill** → resubmit with
`INITIAL_AGE=latest`; **SIGBUS** (intermittent gpuhopper node fault, no new
iterate) → the same command re-resolves `latest` to the last good iterate and
usually lands on a healthy node. Converged output:
`…/periodic/cgridtransports_wparent_upwind3_AB2_kH30_…_LBS/1x4/NK_Q4x4/age_Pardiso_Q4x4.jld2`.

### 3.4 Post-NK diagnostics + plots

```bash
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 GRID_HZ=4 \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
PLOT_NK_QUEUE=hugemem PLOT_NK_MEM=512GB PLOT_NK_NCPUS=18 WALLTIME_PLOT_NK=12:00:00 \
PARENT_MODEL=ACCESS-OM2-01 \
JOB_CHAIN=run1yrNK-combine1yr-ventilation-plotNK-plotventilation \
bash scripts/driver.sh
```

- **Omit `plotNKtrace`** — it references a nonexistent script and, under `set -e`,
  aborts the chain.
- `plotNK` is heavy at 0.1° (full-res figures + animations from the 25-snapshot
  FTS): hugemem, ≥512 GB, long walltime; animations may not finish in one job.

Then repeat all of §3 with `EXPERIMENT=…_qian_wthmp`.

---

## 4. Time to re-emergence (adjoint age via TRAF) — per experiment

TRAF (`TRAF=yes`) reverses every monthly FTS in time and sign-flips `u, v` (and
the prescribed parent `w`) to integrate the adjoint flow; the adjoint matrix is
synthesized algebraically as `invVMtV = V⁻¹ Mᵀ V` from the forward `M`. At the
model level this is **compatible with OM2-01's `wprescribed`+`parent` recipe**
([setup_model.jl:97](src/setup_model.jl) supports TRAF + prescribed *parent* w).

### 4.1 The only work needed: generalize two hardcoded `const` paths (Task B)

The adjoint matrix is `invVMtV = V⁻¹ Mᵀ V`, constructed on the fly from **any**
forward `M` — exactly as in the `const` case. There is no mathematical obstacle
at OM2-01: hand the synthesis the `upwind1` **avg** matrix and it works
identically. The current code just *assumes* `const` because that is all
OM2-1/025 ever had, in two places:

- [create_matrix.jl:50](src/create_matrix.jl) hardcodes the subdir in the forward-M
  read path: `TM/{fwd_mc}/const/M.jld2`.
- [solve_periodic_NK.jl:86](src/solve_periodic_NK.jl) has a guard that rejects
  `TM_SOURCE=avg` under TRAF ("first cut" limitation).

**→ Task B: point both at the avg matrix.** Numerics-free plumbing so TRAF
mirrors the exact forward recipe (`upwind3` forward, `upwind1`-avg
preconditioner):

1. `solve_periodic_NK.jl`: allow `TM_SOURCE=avg` under TRAF (relax the `== const`
   guard; read `invVMtV.jld2` from the `avg/` dir).
2. `create_matrix.jl` (invVMtV branch): (a) read the forward `M` from
   `TM/{fwd_mc}/avg/M.jld2` when `TM_SOURCE=avg`; (b) apply the same
   advection-token swap the run uses (`TM_ADVECTION_SCHEME=upwind1`) when locating
   `fwd_mc`, so it reads the **upwind1** avg matrix while the forward map runs
   `upwind3`; (c) write `invVMtV.jld2` into the matching `avg/` `_traf` dir.

That is the whole change — the same `V⁻¹ Mᵀ V` construction as `const`, just
reading the avg file. (A no-swap shortcut exists — run the TRAF forward map itself
with `ADVECTION_SCHEME=upwind1` so `fwd_mc` already points at the upwind1 matrix —
but that makes the adjoint age more diffusive than the `upwind3` forward age and
mixes diffusion levels in the forward×adjoint ventilation diagnostic, so prefer
the swap in 2b.)

### 4.2 TRAF run (Task B is now implemented — commit 70d97b2)

Depends on the forward `upwind1` avg matrix from §3.2 (same file, read-only). The
**`TMbuild`** step runs `create_matrix.jl`'s `invVMtV` short-circuit: it loads the
forward `upwind1` avg `M`, algebraically synthesizes `V⁻¹ Mᵀ V`, and writes
`invVMtV.jld2` under the swapped `_traf` config where NK reads it (fast, no
autodiff). Then NK solves the adjoint. **No matrix-age solve / `TMage` warm-start**
— start from `0` and restart from `latest`, exactly like the forward recipe (the
`solve_matrix_age*.jl` warm-start reads `model_config`, not the swapped
`TM_MODEL_CONFIG`, so it cannot locate the invVMtV; and the forward OM2-01 solve
converged from zeros anyway).

Per experiment:

```bash
# TMbuild synthesizes the adjoint matrix (algebraic) + NK solves the adjoint age.
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 GRID_HZ=4 TRAF=yes TRAF_TM_SOURCE=invVMtV \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
INITIAL_AGE=0 \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=TMbuild-NK bash scripts/driver.sh

# restart NK from the latest Newton iterate on walltime/SIGBUS (as in §3.3):
EXPERIMENT=01deg_jra55v13_ryf9091_qian_wthp \
TIME_WINDOW=2040-2050 GRID_HZ=4 TRAF=yes TRAF_TM_SOURCE=invVMtV \
ADVECTION_SCHEME=upwind3 TM_ADVECTION_SCHEME=upwind1 TM_SOURCE=avg \
INITIAL_AGE=latest \
PARENT_MODEL=ACCESS-OM2-01 JOB_CHAIN=NK bash scripts/driver.sh
```

Outputs land under the `_traf`-suffixed `MODEL_CONFIG`
(`…_upwind3_AB2_…_LBS_traf`); adjoint age carries the `age_traf_…` filename infix
and "TRAF age (time to re-emergence)" plot titles.

**TRAF stability risk (real).** At OM2-025 the adjoint solve blew up in the
tripolar fold region for one forcing window until `Δt` was reduced
(SRK3-M=9 / AB2-M=3 tamed it — see
[docs/TRAF_simulations.md](docs/TRAF_simulations.md) §3f). OM2-01 TRAF is
untested; if the forward map diverges near `j≈Ny` (max age → 1e40+), drop
`TIMESTEP_MULT` for the adjoint run (e.g. try the model default first, then
halve) before deeper investigation.

Post-NK diagnostics/plots: same chain as §3.4 with `TRAF=yes` added. The
forward × adjoint pair then gives the surface-ventilation diagnostic 𝒱ꜜ
(`compute_ventilation` / `ventilation` step).

---

## 5. Meltwater-difference plots (`wthmp − wthp`)

Per-experiment single fields come from the existing `plotNK` / `plotventilation`
steps. The **new** deliverable is the same-resolution, same-decade difference:

- **Ideal age**: `age(wthmp) − age(wthp)` — depth slices + basin zonal means.
- **Time to re-emergence**: `age_traf(wthmp) − age_traf(wthp)`.
- **Surface ventilation**: `𝒱ꜜ(wthmp) − 𝒱ꜜ(wthp)`.

Both experiments are on the **identical** OM2-01 grid → a straight cell-by-cell
difference (no regridding). **Task D — done (commit pending):**
[src/plot_Li_etal_meltwater_diff.jl](src/plot_Li_etal_meltwater_diff.jl) +
[scripts/plotting/plot_Li_etal_meltwater_diff.sh](scripts/plotting/plot_Li_etal_meltwater_diff.sh).
It loads each experiment's 1-year periodic age FTS (same tag/partition fallback
as `compute_ventilation_diagnostic.jl`), volume-weighted-time-means both, and
renders A|B|(B−A) depth-slice + basin-zonal + profile panels via the shared
`plot_age_comparison_*` primitives (diff panel = B−A, so A=wthp, B=wthmp gives
`wthmp − wthp` = the meltwater effect, on the `:balance` diverging map).
Forward vs adjoint is selected by `MODEL_CONFIG` (add `TRAF=yes` for the `_traf`
tree). **Untested** — it needs the NK/run1yrNK outputs to exist first. Ventilation
(𝒱ꜜ) difference is not yet in the script (age/adjoint-age only); add it once the
ventilation diagnostic outputs land.

---

## 6. Task list

- [x] **A.** Add `CALENDAR_YEAR_OFFSET` (default 0; `-109` here) to
  `periodicaverage.py` slicing; thread through `env_defaults.sh` + `driver.sh`.
  *(done — commit 1320c4a)*
- [x] **A′.** Add `2040-2050` to `prune_time_windows.jl` allowlist *(done —
  commit 752f15d)*; pre-fetch OceanBasins polygons on a login node *(done)*.
- [x] **B.** Enable TRAF avg-matrix path (`solve_periodic_NK.jl` +
  `create_matrix.jl` + `driver.sh` TMbuild vars) — see §4.1. *(done — commit
  70d97b2; `solve_matrix_age*.jl` intentionally left const-only, no `TMage`
  warm-start for OM2-01 TRAF.)*
- [x] **C.** Per experiment: `prep → grid(once)+vel+clo` at `GRID_HZ=4` *(done —
  both Exit 0; offset verified in-log, `slice 2149:2159` = real 2040–2050)*.
- [x] **D.** `plot_Li_etal_meltwater_diff.jl` (+ `.sh`) for the `wthmp − wthp`
  age/adjoint-age panels *(done, untested pending outputs; ventilation-diff TODO)*.
- [x] **E.** Per experiment: build `upwind1` avg matrix *(done — both `avg/M.jld2`
  41.9 GB, nnz 2 442 298 397)*.
- [x] **F.** Per experiment: `partition → NK` forward ideal age (upwind3 /
  upwind1-avg), multi-restart to convergence (§3.3) *(done — both CONVERGED:
  wthmp 990.7 yr, wthp 921.1 yr; meltwater signal +69.6 yr. wthp needed
  `GMRES_RTOL=1e-3`; see §7.)*.
- [ ] **G.** Per experiment: TRAF NK adjoint age / time to re-emergence (§4.2).
- [ ] **H.** Per experiment: `run1yrNK → combine1yr → ventilation → plotNK →
  plotventilation` (forward and TRAF); then the `wthmp − wthp` difference figures.

Order of first light: A → C → E → F (forward age, both experiments) validated the
whole chain at 0.1° ✅ (via the periodicaverage-corruption recovery, §7). Remaining:
**G** (adjoint) and **H** (diagnostics + differences) — see the next-session plan
[docs/Li-etal_next_steps.md](docs/Li-etal_next_steps.md).

---

## 7. Run tracking (job IDs)

All at `PARENT_MODEL=ACCESS-OM2-01`, `TIME_WINDOW=2040-2050`, `GRID_HZ=4`
(as of commit `297dd74` that is the pipeline default, so the explicit override
is no longer needed — it is kept in the recipe below for clarity).
Forward MODEL_CONFIG = `cgridtransports_wparent_upwind3_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS`;
preconditioner (upwind1 avg) MC = same with `_upwind3_`→`_upwind1_`.

**wthp** — `01deg_jra55v13_ryf9091_qian_wthp` (baseline: wind+thermal, no meltwater)

| Task | step | job | state | notes |
|------|------|-----|-------|-------|
| C | prep       | 176183470 | F ✓ | megamem 4h45; slice 2149:2159 = real 2040–2050 |
| C | grid       | 176183472 | F ✓ | grid.jld2 1.6 GB |
| C | vel        | 176183473 | F ✓ | 7 monthly + 6 yearly velocity files |
| C | clo        | 176183474 | F ✓ | |
| E | TMsnapshot | 176219246 | F ✗ | `avg/M.jld2` 41.9 GB, nnz 2 442 298 397 — **built from corrupt velocities, must be rebuilt** |
| F | partition  | 176466922 | F ✗ | 1×4, 1600 GB — superseded (corrupt velocities) |
| F | NK_a       | 176466923 | qdel | NaN in Φ! call #1 → hung |
| C′ | prep (re-run) | 176678772 | F ✓ | 8h09, 785 GB peak; 78/78 slabs verified |
| C′ | vel        | 177187861 | Q   | |
| C′ | clo        | 177187862 | Q   | |
| C′ | diagnose_w | 177187864 | H   | afterok vel |
| C′ | partition  | 177187865 | H   | afterok vel:diagw:clo |
| C′ | probe      | 177188002 | H   | velocity-extremes acceptance check |

**wthmp** — `01deg_jra55v13_ryf9091_qian_wthmp` (wind+thermal+meltwater)

| Task | step | job | state | notes |
|------|------|-----|-------|-------|
| C | prep       | 176184556 | F ✓ | megamem 4h47; slice 2149:2159 |
| C | grid       | 176184558 | F ✓ | grid.jld2 1.6 GB (byte-identical to wthp) |
| C | vel        | 176184560 | F ✓ | |
| C | clo        | 176184562 | F ✓ | |
| E | TMsnapshot | 176219247 | F ✗ | `avg/M.jld2` 41.9 GB, nnz 2 442 298 397 — **built from corrupt velocities, must be rebuilt** |
| F | partition  | 176466929 | F ✗ | 1×4, 1600 GB — superseded (corrupt velocities) |
| F | NK_a       | 176466932 | qdel | NaN in Φ! call #1 → hung |
| C′ | prep (re-run) | 176679803 | F ✓ | 8h22, 785 GB peak; 78/78 slabs verified |
| C′ | vel        | 177187867 | Q   | |
| C′ | clo        | 177187868 | Q   | |
| C′ | diagnose_w | 177187869 | H   | afterok vel |
| C′ | partition  | 177187870 | H   | afterok vel:diagw:clo |
| C′ | probe      | 177188003 | H   | velocity-extremes acceptance check |

State: F ✓ = finished Exit 0, R = running, H = held on dep, — = not yet submitted.
NK is multi-restart (`INITIAL_AGE=latest` each 48 h until `ReturnCode.Success`);
record restart job IDs + final `vol_rms_drift` here as they complete. TRAF (G)
and post-NK/diagnostics (H) rows to be added when submitted.

### ✅ Cause fixed, recovery in progress — corrupt transport climatologies

The first forward NK jobs (`176466923` wthp, `176466932` wthmp) **both NaN-blew-up
in Φ! call #1** and then hung (distributed NaN desync) — qdel'd, no iterate saved.

**Root cause (proven, fixed in `45baa77`):** not arithmetic — the *write*.
`to_netcdf()` on a dask-backed array under a `dask.distributed` cluster runs its
store tasks on the workers, so all 32–48 worker processes opened the same HDF5
file for writing. HDF5 caught this and failed loudly until
`HDF5_USE_FILE_LOCKING=FALSE` was set in April 2026 to work around it; from then
on the writes raced silently. Whole dask chunks never landed (read back as
`_FillValue = NaN`) and a few were torn mid-write (garbage ~1e308). Jobs exited 0
and printed success. NaN blocks align exactly to the dask chunk grid; the ~1e308
values occur only in partially-written chunks. Full write-up:
[docs/periodicaverage_corruption_bug.md](docs/periodicaverage_corruption_bug.md).

**Two earlier assumptions here were wrong:**
- **wthmp was NOT clean** — it was corrupt too (27.1–29.0 M non-finite per
  variable). It cannot proceed to NK independently.
- The damage was far wider than "~28 deep equatorial cells in `ty_trans`": ~5–8%
  of all cells, across **all twelve months**, in **four** variables (`temp`,
  `salt`, `tx_trans`, `ty_trans`), for wthp, wthmp **and** two IAF windows.
  Only the 0.1° monthly 3D fields were hit — yearly files, `eta_t`/`mld`, and
  everything at 1°/0.25° came through clean (corruption tracked output size).

**Recovery status:**

| step | status |
|---|---|
| `periodicaverage.py` fix | ✅ `45baa77` (single-process write + per-slab verify + readback) |
| re-run `prep` (wthp, wthmp) | ✅ `176678772`, `176679803` — 78/78 slab verifications each |
| verify | ✅ audit `177185182` **PASSED**; ref cell `1.566e308` → `5.710295e+04`, and `max|OLD−NEW| = 0` over all 2 154 284 cells the old file left intact |
| re-run `vel` (both) | ✅ rebuilt Aug 24 from clean prep (full monthly FTS temp/salt/mld/eta/u/v/w) |
| verify `vel` (velocity-extremes probe) | ✅ `177216192` (wthp) / `177216194` (wthmp) Exit 0 — physical maxima (u/v ≲ 2, w ≲ 0.02), **0 non-finite**; the 1e298 v/w are gone |
| **rebuild `upwind1 avg` matrix (Task E) — BOTH** | ✅ `177217170` (wthp) / `177217183` (wthmp) Exit 0, fresh `avg/M.jld2` (41.9 GB) Aug 25; replaced the corrupt-input `176219246/247` |
| partition (both) | ✅ `177217315` (wthp) / `177217316` (wthmp) Exit 0 |
| forward NK — run 1 (`INITIAL_AGE=0`) | ✅ `177365034` (wthp) / `177365035` (wthmp) — **0 NaN (blocker cleared)**; iter 1 done, `vol_rms_drift` 0.983 → **0.0155** (wthp, 42 Φ!) / **0.0159** (wthmp, 44 Φ!); saved `newton_iterate_01`; hit 48 h walltime mid-iter-2 |
| forward NK — restart 2 (`INITIAL_AGE=latest`) | walltime −29. **wthmp** ✅ iter 2 done, `newton_iterate_02`, drift **1.883e-6**. **wthp** ⚠️ no new iterate — iter-2 GMRES needs >41 JVPs to rtol=1e-4 (> one 48 h walltime), so restarts from iterate 01 loop. |
| forward NK — restart 3 | **wthmp** `178099348` ✅ **CONVERGED** `ReturnCode.Success` (exit 0), drift → **6.8e-9**, **vol-weighted mean age = 990.7 yr**, `age_Pardiso_Q4x4.jld2` (14.4 GB). **wthp** `178102620` (`GMRES_RTOL=1e-3`): loop broken — iter 2 done, `newton_iterate_02`, drift **1.93e-5**; walltime. |
| forward NK — restart 4 (wthp only) | `178299686` ✅ **CONVERGED** `ReturnCode.Success`, drift → **2.33e-8**, **vol-weighted mean age = 921.1 yr**, `age_Pardiso_Q4x4.jld2` (14.4 GB). |

**✅ FORWARD IDEAL AGE COMPLETE — BOTH EXPERIMENTS.**

| experiment | forcing | mean periodic steady age | drift | GMRES rtol |
|---|---|---:|---:|---|
| wthmp | wind+thermal+**meltwater** | **990.7 yr** | 6.8e-9 | 1e-4 |
| wthp  | wind+thermal only          | **921.1 yr** | 2.3e-8 | 1e-3 |

**Preliminary meltwater signal:** `wthmp − wthp = +69.6 yr` global-mean ideal age
— meltwater makes the ocean **~70 yr older** on average (caps surface, inhibits
AABW convection → less-ventilated deep water). Physically the expected sign.
(wthp's higher final-drift plateau 2.3e-8 vs wthmp's 6.8e-9 is the `GMRES_RTOL=1e-3`
inner tolerance — same fixed point, negligible for the age field.)

Next: post-NK diagnostics (run1yrNK → combine1yr → ventilation → plotNK) for both,
TRAF adjoint (G), then the `wthmp − wthp` difference plots (H,
`plot_Li_etal_meltwater_diff.jl` — now that both forward ages exist).

The pre-fix `.nc` files are kept for comparison at
`{TW}/nc_archive_pre_writefix_20260819/` (285 GB per set). Do **not** sanitize
downstream: `prep_velocities.jl:287-289` maps `NaN → 0`, which is what let the
missing chunks pass as "no transport" and made the corruption invisible to the
solver in the first place.

| investigation | job(s) | finding |
|---|---|---|
| grid parent-check (`CHECK_AGAINST_PARENT_GRID_OUTPUT`) | 176547316/17 → 176553800/08 | bathymetry matches parent; 390 987 "violations" all ≤0.2 mm Float32-on-face rounding (benign) |
| velocity-extremes probe | 176557072/73 → 176560338/39 | wthp v/w ~1e298 @ (848,1246,16) mo10; **wthmp fully clean** |
| face-area-metric probe | 176562522/23 | `AyCFC` normal (1.97e6) at the cell → not the metric |
| v-blowup cross-check | 176563851 | raw ty_trans there = **1.566e308** → corruption is in the preprocessed climatology |

### TRAF adjoint age (Task G) — submitted 2026-09-13

`GMRES_RTOL` default changed 1e-4 → **1e-3** (commit `158b8dc`) after measuring the
forward solves: same 3 Newton iters / ~same total JVPs, but 1e-3 keeps each
iteration ≤ ~35 JVPs (fits one 48 h walltime) vs 1e-4's >41 (walltime loop). All
OM2-01 NK (forward + TRAF) now walltime-safe by default.

TRAF is **untested at 0.1°** (Task B enabling `70d97b2` unexercised there). Chain
`TMbuild-NK`: TMbuild synthesizes `invVMtV = V⁻¹ Mᵀ V` from the clean `upwind1 avg`
forward M; NK solves the adjoint (`INITIAL_AGE=0`, then `latest` restarts).

| exp | TMbuild (invVMtV) | NK (adjoint) |
|---|---|---|
| wthp  | `178896263` | `178896264` 🔄 |
| wthmp | `178896265` | `178896266` 🔄 |

Watch: first Φ! call clears (no NaN); multi-restart to `ReturnCode.Success`; drop
`TIMESTEP_MULT` if it blows up in the fold region (OM2-025 TRAF precedent).

### Forward post-NK diagnostics + plots (Phase 1) — submitted 2026-09-14

Memory fixes first (commit `c505fcc`): the 1-yr age FTS consumers loaded all 25
snapshots (`InMemory()`, ~160 GB) — the earlier OM2-01 1968-1977 `ventilation`
(172582617, 24 GB) and `plotNK` (172582618, 47 GB) were OOM-killed. Now
`InMemory(2)`. `combine1yr` was at 46/47 GB and 29:20/30:00 there → run on
normal/192 GB/4 h. `plotventilation` now takes `TW1` (default `TIME_WINDOW`)
/ optional `TW2` from ENV (was hard-coded 1968-1977 vs 1999-2008) → single-window
figure `plots/{MC}/calVup_forward_2040-2050.png`.

| step | wthp | wthmp | resources |
|---|---|---|---|
| run1yrNK | `178922589` ✓ (1h23, 422 GB host / 281 GB GPU) | `178922591` | 1×4 H200, 4 h |
| combine1yr | `178922590` | `178922592` | normal 48/192 GB, 4 h |
| ventilation | `178929490` | `178929495` | normal 48/190 GB, 2 h |
| plotNK | `178929491` | `178929496` | normal 48/190 GB, 24 h |
| plotventilation | `178929492` | `178929497` | normal 12/48 GB, 2 h |
| ventseasonal | `178929493` | `178929498` | normal 12/48 GB, 2 h |
| ventmovie | `178929494` | `178929499` | normal 48/190 GB, 12 h |
| meltwater diff (both) | `178929720` (afterok both combine1yr) | | normal 48/190 GB, 4 h |
