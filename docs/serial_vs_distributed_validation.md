# Serial vs distributed validation

How to verify that an Oceananigans-based age simulation produces the same answer
serial (1×1) and distributed (1×N or N×M MPI). The historical motivation is a
recurring symptom: a tracer-side discrepancy that **grows from the rank-rank
seam** in distributed runs while serial is fine.

## Scripts

| Script | What it does |
|---|---|
| [test/run_diagnostic_steps.jl](../test/run_diagnostic_steps.jl) | 10-step age simulation, saves every step. Auto-tags CPU output with `_cpu` so CPU and GPU runs don't clobber each other. |
| [test/compare_runs_across_architectures.jl](../test/compare_runs_across_architectures.jl) | Loads serial + distributed snapshots, prints a per-iter volume-weighted RMS norm table, and emits per-snapshot age/u/v/w/η diff plots — both interior-only and halo-inclusive, plus per-iteration `sigma_cc`/`dt_sigma`/`eta_n` (zstar) diffs that show **first divergence**. |
| [scripts/tests/run_diagnostic_steps.sh](../scripts/tests/run_diagnostic_steps.sh) | PBS wrapper for `run_diagnostic_steps.jl`. |
| [scripts/plotting/compare_runs_across_architectures.sh](../scripts/plotting/compare_runs_across_architectures.sh) | PBS wrapper for the compare script. |
| [scripts/test_driver.sh](../scripts/test_driver.sh) | Driver — orchestrates `diag` / `diagcpu` / `diagcpuserial` / `compare` steps. |

## File-naming convention

CPU and GPU diag runs share `outputdir`, so they used to collide. CPU runs are
now tagged with `_cpu`; GPU is the production default and stays unsuffixed:

```
outputs/{PM}/{EXP}/{TW}/standardrun/{MC}/
├── age_diag.jld2                 # GPU serial (1×1)
├── age_diag_cpu.jld2             # CPU serial (1×1)
└── 1x2/
    ├── age_diag_rank0.jld2       # GPU distributed (1×2) rank 0
    ├── age_diag_rank1.jld2       #   …                  rank 1
    ├── age_diag_cpu_rank0.jld2   # CPU distributed (1×2) rank 0
    └── age_diag_cpu_rank1.jld2   #   …                   rank 1
```

The compare script picks via `DURATION_TAG=diag` (GPU) vs `DURATION_TAG=diag_cpu` (CPU).

## Workflow

### Full matrix (CPU + GPU diag, plus 1-year on GPU)

```bash
# CPU pair (writes *_diag_cpu.jld2)
PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x2 \
    JOB_CHAIN=diagcpuserial-diagcpu bash scripts/test_driver.sh

# GPU pair (writes *_diag.jld2) — must submit twice for serial + distributed
GPU_QUEUE=gpuvolta PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x1 \
    JOB_CHAIN=diag bash scripts/test_driver.sh
GPU_QUEUE=gpuvolta PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x2 \
    JOB_CHAIN=diag bash scripts/test_driver.sh

# 1-year GPU pair (uses run_1year.jl, writes age_1year.jld2)
GPU_QUEUE=gpuvolta PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x1 \
    JOB_CHAIN=run1yr bash scripts/driver.sh
GPU_QUEUE=gpuvolta PARENT_MODEL=ACCESS-OM2-1 PARTITION=1x2 \
    JOB_CHAIN=run1yr bash scripts/driver.sh
```

### Compare jobs (one per pair)

The `compare` step in [test_driver.sh](../scripts/test_driver.sh) submits a
single compare run with no PBS dependency, so for fully automated chaining
you submit it directly via the `submit_job` helper with `--deps`:

```bash
# After runs are submitted, capture their job IDs and chain compare jobs:
source scripts/env_defaults.sh
source scripts/submit_job.sh
COMMON_VARS="PARENT_MODEL=${PARENT_MODEL},..."   # see test_driver.sh for full list

submit_job compare_cpudiag 01:00:00 scripts/plotting/compare_runs_across_architectures.sh \
    --queue express --ngpus 0 --ncpus 12 --mem 47GB \
    --deps "<diagcpu_id>:<diagcpuserial_id>" \
    --vars "GPU_TAG=1x2,DURATION_TAG=diag_cpu"

submit_job compare_gpudiag 01:00:00 scripts/plotting/compare_runs_across_architectures.sh \
    --queue express --ngpus 0 --ncpus 12 --mem 47GB \
    --deps "<diag_1x1_id>:<diag_1x2_id>" \
    --vars "GPU_TAG=1x2,DURATION_TAG=diag"

submit_job compare_1year 01:00:00 scripts/plotting/compare_runs_across_architectures.sh \
    --queue express --ngpus 0 --ncpus 12 --mem 47GB \
    --deps "<run1yr_1x1_id>:<run1yr_1x2_id>" \
    --vars "GPU_TAG=1x2,DURATION_TAG=1year"
```

Each compare job's plots land in
`outputs/{PM}/{EXP}/{TW}/standardrun/{MC}/plots/compare_{GPU_TAG}_{DURATION_TAG}/`.

## What the compare script plots

For both the final snapshot and the first/second iteration:

- `serial_…png`, `distributed_…png`, `diff_…png` — interior age, colorrange auto-tuned to `±3·mean|diff|`.
- `reldiff_…png` — age difference / serial age.
- `{u,v,w,eta}_diff_…png` — velocity & free-surface diffs at first non-zero iter.
- `{field}_rank{R}_diff_halos_…png` — **halo-inclusive** per-rank diffs. This is where rank-seam structure shows up most clearly: misfilled halos that contaminate the interior next step appear in this slice.
- `{sigma_cc,dt_sigma,eta_n}_rank{R}_diff_{DURATION_TAG}_iter{IT}.png` — zstar internal state, per iteration. The compare script also logs the **first divergent iteration** per rank, which is the cleanest forensic signal.

Console output includes a per-snapshot table:
```
iter  time(yr)  vol_norm(yr)  max|diff|(yr)
```

## Open hypotheses for the rank-seam signal

We previously hit a serial-vs-distributed divergence that grew at MPI rank
boundaries on 1-year GPU runs (NaNs on rank 0, w differing by ~0.56 m/s,
age blowing up). A `partition_data.jl` slicing fix removed the most visible
symptoms on the diag run, but it's not clear the underlying mismatch is gone.
The Float32 hypothesis from that session (JLD2Writer's default `Array{Float32}`)
was investigated and abandoned (an attempt to force `Array{Float64}` triggered
an unrelated `ReadOnlyMemoryError` in the implicit vertical diffusion solver).

Current Oceananigans pin: `briochemc/Oceananigans.jl @ bp/offline_ACCESS-OM2_v3`,
sha `91a26ad` (2026-05-14), Oceananigans 0.107.6 — synced with `CliMA/Oceananigans.jl`
main as of the same date. PR #5427 (CommunicationBuffers swap fix) is included.
PR #5564 (conditional-advection on tripolar) is *not* — still open upstream.

### ~~Hypothesis 1 — PR [#5427](https://github.com/CliMA/Oceananigans.jl/pull/5427) "Fix north/south buffer swap in `CommunicationBuffers`"~~ — **ruled out**

Initial reading of [issue #5422](https://github.com/CliMA/Oceananigans.jl/issues/5422)
and PR #5427 looked promising (GPU-only Adapt path, north/south swap on a 1×2
y-partition). But the PR discussion (and Claude's analysis on issue #5422)
makes clear:

- `Adapt.adapt_structure(::OneDBuffer/TwoDBuffer/CornerBuffer) = nothing` — the
  individual buffer types adapt to `nothing`, so `Adapt.adapt_structure(::CommunicationBuffers)`
  always produces an all-`nothing` struct. The swap is invisible.
- `fill_halo_regions!` passes `field.communication_buffers` **directly**, never
  through `Adapt.adapt`, so MPI sends/receives use the correctly-ordered buffers.
- The `on_architecture` half of the swap *is* a real bug, but it only fires on
  explicit architecture transfer (serialization / output paths), which doesn't
  touch the in-loop halo exchange or the saved field data array.

So PR #5427 does not explain the GPU-only seam drift we observe.

### Hypothesis 2 — PR [#5564](https://github.com/CliMA/Oceananigans.jl/pull/5564) "Fix bug in conditional advection for `TripolarGrid`s" *(WENO-only)*

Open in upstream — not in our fork. `LeftConnectedRightCenterFolded` etc. were
incorrectly included in the `BT` (bounded) topology union in
`topologically_conditional_interpolation.jl`. Effect: a distributed rank whose
own y-topology has a fold on top and a rank-rank seam on the bottom was treating
the **seam** as a wall, falling back to a lower-order stencil right there in
distributed mode while serial used the full stencil. For our default
`centered2` advection `required_halo_size = 1`, so the conditional path is
essentially never taken — minimal impact. Would matter for `weno3/weno5` runs.

### Hypothesis 3 — PR [#5489](https://github.com/CliMA/Oceananigans.jl/pull/5489) "Fix show(field) + zipper BC validation for (distributed) tripolar grids" *(probably not relevant for us)*

Merged 2026-05-05 — not in our fork. A user-supplied non-zipper north BC was
silently dropped on distributed and silently used (wrongly) on serial. We do
set `FPivotZipperBoundaryCondition` explicitly everywhere
([src/matrix_setup.jl](../src/matrix_setup.jl), [src/prep_velocities.jl](../src/prep_velocities.jl),
[src/shared_utils/grid.jl](../src/shared_utils/grid.jl)) so we shouldn't trip
this. Worth knowing about anyway.

### Hypotheses we ruled in/out via inspection

- PR #5471 (active-cells map `:xyz`/`:xy` plumbing) — already in our fork, unlikely.
- PR #5435 (skip `fill_corners!` when no corner neighbour) — in our fork, only skips a sync, not data.
- PR #5408 (RightFaceFolded fold-row shift) — in our fork.
- PR #5439 (distributed tripolar fold) — in our fork (verify cherry-pick is complete).
- PR #5565 (halo metrics for TripolarGrid) — affects metrics at j=1, partition-invariant.
- PR #5571 (distributed immersed boundary reconstruction) — only affects global-grid reconstruction (I/O), not run-time halo paths.
- PR #5492 / #5486 (JLD2Writer plumbing) — output-only, doesn't touch physics.

## Observed pattern → narrows the search

Results below (CPU bit-identical, GPU shows seam drift with bit-identical
velocity fields) tell us:

- The bug is in the **GPU code path**, not in the MPI logic, partitioner, or
  generic CPU advection (those are exercised in CPU MPI too).
- It is specific to **tracer halos**, not dynamics — `u`, `v`, `eta` are
  bit-identical between serial and 1×2 distributed on GPU.
- The contamination appears at the rank-rank seam, suggesting the GPU halo
  exchange writes the wrong values into a tracer's halo, or reads its halo
  with a wrong stride / sign convention.

Candidate mechanisms to investigate next:

- GPU halo-fill kernels that branch differently on partition geometry (e.g.,
  fold-aware fill that mishandles the case where the rank-rank seam is well
  south of the fold).
- CUDA-aware MPI / device-buffer paths in `halo_communication.jl` (the CPU
  build skips device-staging entirely).
- A latent kernel bug in `fill_west_and_east_halo!`-style operations under
  Distributed where the rank's local extent is half the global Ny.
- Anything in our fork that diverges from upstream specifically in
  `DistributedComputations/halo_communication.jl`.

## Results

Run on 2026-05-14, OM2-1, defaults `cgridtransports_wdiagnosed_centered2_AB2`,
TW=1968-1977. All numbers below are **relative** (`mean|diff|/mean|serial|`
and pointwise `max|reldiff|`) — absolute magnitudes in seconds-of-age or m/s
are misleading because `w ≈ 10⁻⁶ m/s` is six orders of magnitude smaller than
`u, v ≈ 10⁻² m/s`, so an "FP-roundoff" `w` diff is actually a meaningful
fraction of typical `w`.

Field magnitudes used as the denominator (from serial run, wet cells, NaN-filtered):

| field | mean\|serial\| | max\|serial\| |
|---|---|---|
| `u` | 2.92e-2 m/s | 1.01 m/s |
| `v` | 1.31e-2 m/s | 5.06e-1 m/s |
| `w` | 2.01e-6 m/s | 8.68e-4 m/s |
| `eta` | 5.82e-1 m | 1.81 m |
| `age` (diag, end) | 5.22e4 s (~0.6 d) | 5.49e4 s |
| `age` (1year, end) | 6.7e-1 yr | 2.08 yr |

### 1-year: GPU 1×2 vs 1×1 — **clear rank-seam contamination**

Compare job: 168312952 (exit 0). Plots:
[outputs/.../plots/compare_1x2_1year/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_1year).

Final-snapshot age statistics (wet cells only):

| metric | value | interpretation |
|---|---|---|
| `mean\|reldiff\|` | 8.37e-4 | ~0.08% typical relative age error after 1 yr |
| `max\|reldiff\|` | 3.88e+1 | huge in cells where serial age ≈ 0 (near sources) — pointwise outliers |
| `mean\|diff\|/mean\|serial\|` | 1.5e-4 | ~0.015% bulk relative error |
| `RMS(diff)/RMS(serial)` | 5.7e-4 | ~0.06% RMS relative error |

The full-domain interior diff at z≈1030 m shows a clear horizontal stripe of
"distributed is less-aged" along the rank-rank seam, mostly visible across the
Pacific:

![age diff at 1000 m, end of year 1](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_1year/diff_1x2_1year_centered2_iter5844_slice_1000m.png)

Global zonal-average diff at end of year 1 — the contamination is **surface-trapped
near the equator** (the j=150 partition boundary on OM2-1 sits at the equator) and
penetrates ~3500 m down in a narrow plume:

![zonal avg diff, end of year 1](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_1year/diff_1x2_1year_centered2_iter5844_zonal_avg_global.png)

Per-rank halo-inclusive age diff confirms the diff is localised right at the
rank's interface with its neighbour — rank 0 (southern) has its diff along its
top edge (j≈Ny/2+halos), rank 1 (northern) has its diff along its bottom edge.
Mirror images, perfectly aligned at the seam:

| rank 0 (southern) | rank 1 (northern) |
|---|---|
| ![rank 0 age halo diff](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_1year/age_rank0_diff_halos_1year_k57.png) | ![rank 1 age halo diff](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_1year/age_rank1_diff_halos_1year_k57.png) |

**Velocities & free surface at iter 487 (first non-zero saved snapshot)** —
relative diffs, denominator is per-field magnitude from the table above:

| field | `mean\|diff\|/mean\|serial\|` | `max\|diff\|/max\|serial\|` |
|---|---|---|
| `u` surface | 0 | 0 |
| `v` surface | 0 | 0 |
| `w` k=51 (top) | 1.44e-13 / 2.01e-6 ≈ **7.2e-8** | 2.58e-10 / 8.68e-4 ≈ **3.0e-7** |
| `w` k=50 | 1.36e-13 / 2.01e-6 ≈ **6.8e-8** | 2.17e-10 / 8.68e-4 ≈ **2.5e-7** |
| `eta` 2D | 0 | 0 |

`u` and `v` are bit-identical at the surface. `w` differs only at machine-epsilon
levels — consistent with diagnostic-`w` recomputation order rather than transport-
field divergence. So the divergence does NOT show up in the velocity fields
themselves — it shows up in the **tracer** that the (bit-equal) velocities
advect through a (suspectedly mis-swapped) halo.

**zstar fields (`sigma_cc`, `dt_sigma`, `eta_n`) — halo-fill artefact in the
diagnostic save, NOT a real model divergence.** The compare script reports
"FIRST DIVERGENCE at iter 0" for `sigma_cc` (max|diff|≈5.4e-2) and `eta_n`
(max|diff|≈9.7e-1) on **both** CPU and GPU compares, with byte-identical
values. Cross-checked directly against the JLD2 files via
[scripts/debugging/check_zstar_locations.jl](../scripts/debugging/check_zstar_locations.jl):

| field | max\|diff\| over full saved array | max\|diff\| over interior only |
|---|---|---|
| `sigma_cc` rank 0 | 5.36e-2 | **0.00e+00** |
| `sigma_cc` rank 1 | 5.02e-2 | **0.00e+00** |
| `eta_n` rank 0 | 9.73e-1 | **0.00e+00** |
| `eta_n` rank 1 | 9.18e-1 | **0.00e+00** |

All diffs live in **halo rows** at `j=1` (rank 1's south halo) and
`j=Ny_rank` (rank 0's north halo). Pattern: the rank's saved halo cells hold
the placeholder values (`sigma=1.0`, `eta=0.0`), while the serial global
array has the actual values filled by the BC there. `save_zstar_fields`
dumps raw `parent(...)` data without re-filling halos in the distributed
case. Bug is in the *diagnostic save*, not in the *model state*. **CPU
interior is genuinely bit-identical** — consistent with the age=0 result.

### Diag (10 steps): CPU 1×2 vs 1×1 — **bit-identical**

Compare job: 168312950 (exit 0). Plots:
[outputs/.../plots/compare_1x2_diag_cpu/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag_cpu).

| metric | value |
|---|---|
| `mean\|reldiff\|` (age) | 0 |
| `max\|reldiff\|` (age) | 0 |
| u/v/w/eta surface relative diff | 0 |

On pure CPU MPI the run is **bit-identical** between 1×1 and 1×2 across all
10 iterations.

### Diag (10 steps): GPU 1×2 vs 1×1 — **rank-seam already visible after 10 steps**

Compare job: 168312951 (exit 0). Plots:
[outputs/.../plots/compare_1x2_diag/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag).

Age stats:

| metric | value | interpretation |
|---|---|---|
| `mean\|reldiff\|` | 2.54e-4 | ~0.025% mean relative age error after 10 steps |
| `max\|reldiff\|` | 9.85e-1 | pointwise outlier in cells where age ≈ 0 |
| `mean\|diff\|/mean\|serial\|` | ~5e-7 | bulk relative drift very small at 10 steps |

Velocity / free-surface relative diffs at iter 1 (one timestep after t=0):

| field | `mean\|diff\|/mean\|serial\|` | `max\|diff\|/max\|serial\|` |
|---|---|---|
| `u` surface | 0 | 0 |
| `v` surface | 0 | 0 |
| `w` k=51 (top) | 1.28e-13 / 2.01e-6 ≈ **6.4e-8** | 4.54e-11 / 8.68e-4 ≈ **5.2e-8** |
| `w` k=50 | 1.25e-13 / 2.01e-6 ≈ **6.2e-8** | 4.01e-11 / 8.68e-4 ≈ **4.6e-8** |
| `eta` 2D | 0 | 0 |

Interior age diff at z=1030 m, iter 10 — the faint blue stripe at j≈150 across
the Pacific is already aligned at the rank seam and is the same structure that
grew to the bold blue stripe in the 1-year plot above:

![GPU diag, iter 10, 1000 m slice](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag/diff_1x2_diag_centered2_iter10_slice_1000m.png)

### Halo-depth sweep — all fields, all halo trim levels

To verify that the bug is purely in tracer halos (and not in dynamics or
silent halo placeholders), we ran [scripts/debugging/halo_diff_sweep.jl](../scripts/debugging/halo_diff_sweep.jl)
on every saved JLD2 variable, reporting max|diff| at four trim levels:
**interior** (trim by full Hx,Hy), **+1 halo**, **+2 halos**, and **full halos**.
Detected `Hx=Hy=13` from `age` (Center-Center). 1×2 split gives 150 Center-y
cells per rank → rank 0 covers global parent y=1..176, rank 1 covers y=151..326.

GPU diag (iter 10 for age/zstar; iter 1 for u/v/w/eta):

| field   | rank | interior   | +1 halo    | +2 halos   | full halos   |
|---------|------|------------|------------|------------|--------------|
| age     | 0    | 3.42e+3 s  | 4.81e+4 s  | 4.81e+4 s  | 4.81e+4 s    |
| age     | 1    | **4.81e+4 s** | 4.81e+4 s  | 4.81e+4 s  | 4.81e+4 s    |
| u       | 0/1  | 0          | 0          | 0          | 0            |
| v       | 0    | 0          | 0          | 0          | 0            |
| v       | 1    | 1.38e-1*   | 1.38e-1*   | 1.38e-1*   | 1.38e-1*     |
| w       | 0/1  | ~3–5e-11   | ~3–5e-11   | ~3–5e-11   | ~6e-5–2e-4   |
| eta     | 0/1  | 0          | 0          | 0          | 0            |
| sigma_cc/dt_sigma/eta_n | 0/1 | 0     | 0          | 0          | 0            |

CPU diag — same sweep:

| field   | rank | interior | +1 halo | +2 halos | full halos |
|---------|------|----------|---------|----------|------------|
| age, u, eta | 0/1 | 0     | 0       | 0        | 0          |
| v       | 0    | 0        | 0       | 0        | 0          |
| v       | 1    | 1.38e-1* | 1.38e-1*| 1.38e-1* | 1.38e-1*   |
| w       | 0/1  | 0        | 0       | 0 / (18200 NaN in rank 0 z-halos) | 6.5e-5 / 2.2e-4 |
| sigma_cc/dt_sigma/eta_n | 0/1 | 0 | 0 | 0   | 5.4e-2 / 1.2e-9 / 9.7e-1 |

`*` v rank-1 diff is at the **tripolar fold row** (global Face-y = Ny+1 = parent
y=314), not at the rank-rank seam — see "Known save-side artefacts" below.

### Seam y-profile — exactly where does the diff first appear?

[scripts/debugging/seam_profile.jl](../scripts/debugging/seam_profile.jl)
scans `max|diff|` row-by-row across global parent y in the seam band (parent
y=156..170, i.e. Center-y=143..157), GPU diag:

| global parent y | global Center y | age `max\|diff\|` (s) | note |
|---|---|---|---|
| 158 | 145 | 0 | |
| 159 | 146 | 3.91e-3 | |
| 160 | 147 | 5.47e-2 | |
| 161 | 148 | 2.79 | |
| 162 | 149 | 1.50e+2 | |
| 163 | 150 | **3.42e+3** | rank 0's last interior cell |
| 164 | 151 | **4.81e+4** | rank 1's first interior cell ← **SEAM** |
| 165 | 152 | 3.64e+3 | |
| 166 | 153 | 2.60e+2 | |
| 167 | 154 | 1.68e+1 | |
| 168 | 155 | 0.875 | |
| 169 | 156 | 3.52e-2 | |
| 170 | 157 | 3.91e-3 | |

**Bell-shaped peak exactly at global Center-y=151** (rank 1's first interior
cell, immediately above the seam). 10-step diag has spread the contamination
~5 cells in either direction, consistent with centered2 advection's one-cell
stencil propagating over 10 timesteps. CPU diag is **0 across the entire band**
for every field — confirms the seam contamination is GPU-only.

### Known save-side artefacts (NOT real model divergence)

These were initially mistaken for real seam signals; they're side effects of
how diagnostic fields are saved, and don't reflect the model's runtime state.

1. **zstar halo cells (`sigma_cc`, `eta_n`, `dt_sigma`) on CPU.** The compare
   script's "FIRST DIVERGENCE at iter 0" lines (max|diff|≈5.4e-2, 9.7e-1,
   1.2e-9) come entirely from the outermost halo row of each rank — the
   `save_zstar_fields` callback writes `parent(field.data)` without filling
   halos in distributed mode, so the rank's halo cells hold placeholder
   values (`sigma=1`, `eta=0`) while serial has BC-filled values there.

   Visual confirmation: `sigma_cc` rank-0 vs rank-1 diffs at iter 10 of CPU
   diag — a thin row at j≈175 (rank 0's north halo = seam) and j≈1 (rank 1's
   south halo = seam), interior completely zero. Mirror images of each other:

   | rank 0 (south halo at top) | rank 1 (south halo at bottom) |
   |---|---|
   | ![sigma_cc rank 0 CPU](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag_cpu/sigma_cc_rank0_diff_diag_cpu_iter10.png) | ![sigma_cc rank 1 CPU](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag_cpu/sigma_cc_rank1_diff_diag_cpu_iter10.png) |

   Same for `eta_n` at iter 10:

   | rank 0 | rank 1 |
   |---|---|
   | ![eta_n rank 0 CPU](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag_cpu/eta_n_rank0_diff_diag_cpu_iter10.png) | ![eta_n rank 1 CPU](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag_cpu/eta_n_rank1_diff_diag_cpu_iter10.png) |

   **Resolved — only the outermost halo row is stale; the seam-adjacent
   halo is correctly filled on both CPU and GPU.** Probed directly via
   [scripts/debugging/zstar_halo_values.jl](../scripts/debugging/zstar_halo_values.jl)
   at three depths in rank 0's north halo (Hy=13, so 13 halo cells total):

   | rank 0 parent y | depth into halo | `eta_n` values | matches serial? |
   |---|---|---|---|
   | 164 | 1 (seam-adjacent) | real eta values (0.44, 0.59, 0.25, …) | **YES** (diff=0) |
   | 170 | 7 (middle)        | real eta values (0.44, 0.61, 0.20, …) | **YES** (diff=0) |
   | 176 | 13 (outermost)    | 0.0 in every cell (placeholder)        | NO (diff ≈ -0.5) |

   So `fill_halo_regions!` *is* running on the zstar fields in distributed
   mode, but only fills 12 of the 13 halo rows — leaves the outermost cell
   at its init value. The compare-script diff is entirely at that single
   outermost row. Same fill pattern on CPU and GPU.

   **Implication for the seam tracer bug:** centered2 advection reads only
   `Hy=1` cell beyond the interior; that cell is correctly filled on both
   CPU and GPU. So **stale zstar halos do NOT explain the GPU seam tracer
   drift** — the cell-face heights at the seam are computed from the same
   (correctly-filled) zstar values on both arches. Some other GPU-only
   mechanism is producing the seam tracer diff.

   Extending the same probe to *every* saved field at GPU rank 0's
   seam-adjacent halo row (parent y=164) gives:

   | field | rank 0 vs serial @ y=164, GPU diag iter 10 |
   |---|---|
   | `u` | 0 at every probed `i` |
   | `v` | 0 at every probed `i` |
   | `w` | 0 at every probed `i` |
   | `eta` | 0 at every probed `i` |
   | `sigma_cc` | 0 at every probed `i` |
   | `eta_n` | 0 at every probed `i` |
   | `dt_sigma` | 0 at every probed `i` |
   | `age` | 0 at most `i`; −0.92 s at `i=106` (rel ~2e-5) |

   Every state field the seam-flux kernel reads from the halo is
   bit-identical to serial. The tiny `age` diff in rank 0's seam halo at
   `i=106` is not a halo-fill bug — that halo is filled by MPI exchange
   from rank 1's interior, and rank 1's interior at global y=151 already
   carries the GPU seam drift. So rank 0's halo is just faithfully copying
   rank 1's already-drifted age.

   **What this tells us about the GPU bug:** halo exchange is working;
   all velocity/zstar/eta state at the seam-adjacent halo is correct on
   GPU. The seam flux kernel is therefore being fed correct inputs yet
   produces a different tendency than serial. The next places to look:
   - the **tracer tendency / advection kernel** for a GPU-specific
     code path near the rank boundary
   - **static / cached inputs** the kernel reads: grid metrics
     (`Δxᶜᶜᵃ`, `Δyᶜᶜᵃ`, `Azᶜᶜᵃ`), `bottom_height`, the wet/active mask,
     `MutableVerticalDiscretization` z-coordinate state
   - **reduction / sum operations** across the seam that might compile
     differently on GPU (e.g., free-surface barotropic substep aggregating
     across ranks).

### Iter timeline at the seam: bug fires on iter 1, confined to one row

Probed `age` on GPU diag at iter 0, 1, 2, ... 10 over the full 3D rank 1
array (every i, j, k), to pinpoint when and where the bug first fires:

| iter | rows with \|diff\| > 1e-10 in rank 1 | peak row | peak max\|diff\| |
|---|---|---|---|
| 0  | (none — initial state identical) | — | 0 |
| 1  | **j=14 only** (2896 cells) | j=14 | **4.44e+3 s** |
| 2  | j=13, **14**, 15 (5832 cells)      | j=14 | 9.36e+3 s |
| 3  | j=12–16 plus a few outliers        | j=14 | grows |
| 10 | j=10–20 (bell-shape, 19000 cells)  | j=14 | 4.81e+4 s |

**The bug fires on the very first step (Euler bootstrap), and at iter 1 is
confined to exactly one row: rank 1's first interior row** (parent y=14 =
global Center-y = 151, immediately above the seam). Every other row is
bit-identical to serial at iter 1. From iter 2 onward, centered2 advection
propagates the iter-1 contamination one cell per step, producing the
bell-shape seen at iter 10.

Inspecting the actual values in rank 1 at iter 1 (rank-1-parent-y = 1..20):

| j (rank 1 parent) | global j | rank 1 [min, max] | serial [min, max] | max\|diff\| |
|---|---|---|---|---|
| 1–13 (south halo) | 151–163 | [0, **5400**] | [0, 5400] | **0** ← halos correct |
| **14 (first interior)** | 164 | [0, **8718**] | [0, 5400] | **4443** ← bug here |
| 15–20 (deeper interior) | 165–170 | [0, 5400] | [0, 5400] | 0 |

**The key tell**: rank 1's max age at j=14 is **8718 s — exceeding `Δt = 5400 s`**.
Age physically cannot grow by more than `Δt` in one timestep (it's accumulating
"time since immersion"). The model is over-aging some cells in row j=14
(and under-aging others — the cell I sampled, i=62 k=50, has 747 s vs serial 5190 s).

Halo rows 1–13 are bit-identical to serial → MPI exchange is delivering
correct values. Rows 15+ are bit-identical → the tendency kernel is fine
everywhere except at j=14. So **the bug is specifically in how the tendency
is computed at rank 1's first interior row**, not in the halo exchange and
not in the broader kernel.

Strong candidates for "what's different at exactly j=14":

- **Cell metrics** (`Δyᶜᶜᵃ`, `Δxᶜᶜᵃ`, `Azᶜᶜᵃ`) — constructed at partition
  time, possibly with off-by-one or wrong sign for the row at the rank's
  southern boundary.
- **Vertical-coordinate cell-face heights** computed from `sigma_cc` /
  `eta` at j=14's south face — the south face of row j=14 reads `sigma_cc`
  from both j=14 itself and j=13 (the halo). If that interpolation goes
  wrong only at this exact row, we'd see exactly this signal.
- **Tracer-tendency kernel** branching on a "near-southern-boundary" flag
  that's set wrong for rank 1's first interior row (e.g., a kernel that
  applies a one-sided derivative at the *true* domain south boundary
  mistakenly applying it at the rank's south boundary).

The CPU run does not exhibit this (rank 1's j=14 is bit-identical to serial),
so whichever of those candidates is responsible, the issue lives in a
GPU-only or KernelAbstractions-launch branch of the code path.

**CPU vs GPU side-by-side, iter 1, rank 1's first interior row** (j=14, the row where the GPU bug fires):

| metric | CPU | GPU | serial |
|---|---|---|---|
| `j=14` max age | **5400 s (= Δt, physical)** | **8718 s (> Δt, non-physical)** | 5400 s |
| rows with diff vs serial | **none** (bit-identical) | j=14 only (2896 cells) | — |
| max\|diff\| at j=14 | 0 | 4.44e+3 s | — |

Same `centered2 + AB2` model, same partition, same MPI launch, same input grid.
The only difference is `arch = CPU()` vs `arch = GPU()`. So the bug must be in
a code path where the kernel compilation / launch behaviour diverges between
the two backends — almost certainly a kernel that has an indexing or boundary
condition that gets specialised differently on the GPU.

**Cross-check — does serial GPU match serial CPU at j=164?** Yes, bit-identically
at every iter (0, 1, 2, 10): max\|diff\| = 0 over row j=164. (Full-array there's
some FP-roundoff scatter — 10 cells at iter 2 differing by ~1e-3 s, 64 cells at
iter 10 by ~4e-3 s, all far from the seam.) So serial GPU has the right value at
the row where the *distributed* GPU goes wrong.

**The 2×2 table — bug only fires under GPU AND distributed:**

| | serial (1×1) | distributed (1×2) |
|---|---|---|
| **CPU** | reference (correct) | bit-identical to serial CPU |
| **GPU** | bit-identical to serial CPU at j=164 | **broken at j=14 (rank 1) / j=164 (global)** |

That intersection — GPU backend + MPI partition — is exactly the necessary
condition. The suspect kernel must (a) only run in distributed mode (otherwise
serial GPU would already differ from serial CPU) and (b) have a GPU-specific
specialisation (otherwise distributed CPU would also break). That's a much
narrower search target than "the entire GPU code path".

**Only `age` is affected — all other fields are bit-identical at iter 1
across the seam.** Per-field probe at GPU rank 1, parent y=13..16 (south
halo + first three interior rows), iter 1:

| field | j=13 (halo) | **j=14 (1st interior)** | j=15 | j=16 |
|---|---|---|---|---|
| `u` | 0 | 0 | 0 | 0 |
| `v` | 0 | 0 | 0 | 0 |
| `w` | 5e-12 (FP) | 2.5e-12 (FP) | 2.7e-12 (FP) | 9.9e-12 (FP) |
| `eta`, `sigma_cc`, `dt_sigma`, `eta_n` | 0 | 0 | 0 | 0 |
| **`age`** | 0 | **4.44e+3 s** (2896 cells) | 0 | 0 |

So dynamics is bit-identical, zstar is bit-identical, halos are bit-identical;
**only the tracer (`age`) has a wrong tendency, and only at the first interior
row of rank 1**. That isolates the bug to the tracer-only code path: tracer
advection or the `age` forcing term, GPU + distributed only, at exactly the
row that abuts the rank-rank seam. Candidates worth looking at:

1. **The age forcing** (`dage/dt = 1 [s/s]`) — if its kernel range is wrong
   under Distributed on GPU, the forcing could fire twice or with wrong stride
   at the rank's south boundary row.
2. **Tracer advection at the rank's south boundary** — the kernel for the
   south face of row j=14 reads (j=13 halo, j=14 interior). Both inputs are
   bit-identical to serial, so any wrongness has to come from the kernel's
   arithmetic / indexing at that specific row.
3. **Implicit vertical diffusion solver** applied to the tracer at this row,
   if it uses any across-y operation.

   **Still a save-side fix to do**: `save_zstar_fields` should either trim
   the outermost halo or call `fill_halo_regions!` immediately before
   saving, so the compare script doesn't report this as a divergence.

2. **v at the tripolar fold row (rank 1, global Face-y = Ny+1 = parent y=314).**
   Rank 1's saved `v` at the fold row has the **opposite sign** to serial
   (`v_rank = -v_serial` exactly, per [scripts/debugging/locate_v_diff.jl](../scripts/debugging/locate_v_diff.jl)).
   This is documented in `compare_runs_across_architectures.jl` itself
   (lines 599-606): JLD2Writer wraps fields in anonymous ComputedFields whose
   `fill_halo_regions!` dispatches to default `sign=+1` BCs because the wrapper
   isn't named `:u`/`:v`. Serial applies the wrong (positive) sign and stores
   the wrong fold-row value; distributed retains the true `sign=-1` value.
   **This needs a fix** (in `JLD2Writer` or in our save callback) — the
   *saved* `u`/`v` data near the fold has the wrong sign in serial mode.
   It does not affect the simulation itself (the model uses its own internal
   field with the correct BCs), only the diagnostic output.

### Jobs submitted (2026-05-14, all exit 0)

| Job | ID |
|---|---|
| diagcpuserial (CPU 1×1) | 168312669 |
| diagcpu (CPU 1×2) | 168312668 |
| diag (GPU 1×1) | 168312680 |
| diag (GPU 1×2) | 168312681 |
| run1yr (GPU 1×1) | 168312682 |
| run1yr (GPU 1×2) | 168312683 |
| compare_cpudiag (afterok …668:669) | 168312950 |
| compare_gpudiag (afterok …680:681) | 168312951 |
| compare_1year (afterok …682:683) | 168312952 |

## Interim conclusion

The 1×2 GPU runs produce a clear, localised tracer mismatch at the rank-rank
seam — already visible after 10 steps in `diag` (mean relative age error
~0.025%) and growing to ~0.08% mean / ~0.06% RMS relative age error after
1 year, with Pacific seam stripe structure at z≈1000 m. The CPU 1×2 run is
**bit-identical** to CPU 1×1 across all 10 diag iterations.

- The bug is **GPU-specific**, not in the partitioner, MPI logic, or generic
  CPU advection.
- Velocity fields are bit-identical between serial and distributed (`u`, `v`,
  `eta` exactly; `w` relative diff ~10⁻⁷, i.e. ~10⁻¹³ m/s on a typical
  `w ~ 10⁻⁶ m/s` — FP roundoff). The dynamics is fine; the **tracer halo
  exchange / fold-fill** on the GPU path is the load-bearing suspect.
- PR #5427 was initially hypothesised but is ruled out (Hypothesis 1 above —
  the `Adapt.adapt_structure` swap is a no-op because individual buffer types
  adapt to `nothing`).

Next steps:

1. **Find the GPU-only mechanism** (the real bug).
   - The seam-adjacent halo probe rules out halo-fill bugs for **all** state
     fields (u, v, w, eta, zstar). Halo exchange is working. So look at
     what the tracer kernel reads *besides* halos: grid metrics, bottom
     topography, wet mask, the `MutableVerticalDiscretization` state.
   - Compare `bottom_height`, `Δxᶜᶜᵃ`, `Δyᶜᶜᵃ`, `Azᶜᶜᵃ` between serial
     and distributed grid files near the seam — these are loaded once
     at start of simulation, so any partition-construction bug would
     leave a permanent diff that affects every timestep.
   - Look at GPU-only kernel branches in the tracer tendency / advection
     code paths under Distributed.
   - Run `weno5` instead of `centered2` and rerun GPU diag — if the seam
     grows a lot more, PR #5564 (conditional-advection treatment of fold
     topologies) is contributing too.
   - Try a 2×1 (x-partition) instead of 1×2 (y-partition) to confirm the
     seam follows the partition direction.

2. **Fix the v fold-row sign artefact in the save path.** Per the docstring
   in `compare_runs_across_architectures.jl` lines 599–606: `JLD2Writer`
   wraps `u`/`v` in anonymous `ComputedField`s, so the `fill_halo_regions!`
   on the wrapper can't dispatch to `sign=-1` for the zipper BC; it defaults
   to `sign=+1` and stores the wrong fold-row value in serial. Confirmed via
   [locate_v_diff.jl](../scripts/debugging/locate_v_diff.jl): exact sign-flip
   at global Face-y=314 (the fold row). Either name the wrapper `:u`/`:v` so
   it dispatches correctly, or save with `with_halos=false` for `u`/`v` and
   use the existing manual-callback path that bypasses the wrapper.

3. **Fix the zstar diagnostic save** (`sigma_cc`, `eta_n`, `dt_sigma`).
   Halo cells on each rank hold placeholder values (sigma=1, eta=0) while
   serial has BC-filled values there. Either call `fill_halo_regions!` on
   the zstar fields before saving, or trim the saved arrays to the interior.
   (Interior is bit-identical — confirmed via [check_zstar_locations.jl](../scripts/debugging/check_zstar_locations.jl).)

4. **Document, then iterate.** Subsequent runs (WENO sweep, 2×1 partition,
   bisect candidates) should add rows to the Results section here.
