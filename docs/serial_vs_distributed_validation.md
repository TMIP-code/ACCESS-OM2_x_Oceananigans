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

**zstar fields (`sigma_cc`, `dt_sigma`, `eta_n`) per-iter** — the script reports
"first divergence" iteration for each zstar field per rank. The recorded
non-zero diffs at iter 0 (e.g. `sigma_cc` ≈ 5.36e-2 ≈ 5.4% of unity; `eta_n`
≈ 9.73e-1 ≈ ~unity) reproduce **byte-identically** between the CPU and GPU
compares — strongly suggesting these are a **compare-script slicing artefact**
(serial saved with halos, distributed saved without, or vice versa) rather
than a real iter-0 mismatch. This needs follow-up before relying on these
plots; the **tracer-side seam signal** above is the load-bearing evidence.

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

1. **Find the actual GPU-only mechanism.** Suggested:
   - Run `weno5` instead of `centered2` and rerun GPU diag — if the seam grows
     a lot more, PR #5564 (conditional-advection treatment of fold topologies)
     is contributing too.
   - Bisect / inspect GPU-only kernel branches in `fill_halo_regions!` for
     tracers in `src/DistributedComputations/` between the v3 base and the
     last known-good Oceananigans version.
   - Try a 2×1 (x-partition) instead of 1×2 (y-partition) to confirm the seam
     follows the partition direction (eliminates "it's always near the
     equator" interpretations).
2. **Sanity-check the zstar iter-0 diff.** `sigma_cc` and `eta_n` show
   *byte-identical* non-zero diff at iter 0 in CPU vs GPU compares — almost
   certainly a compare-script slicing artefact (serial saved with halos,
   distributed saved without halos, or vice versa). Worth fixing before
   relying on those zstar plots forensically.
3. **Document, then iterate.** This doc is the workflow + first-pass results;
   subsequent runs (WENO sweep, 2×1 partition, bisect candidates) should add
   rows to the Results section here.
