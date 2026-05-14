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
sha `91a26ad` (2026-05-14), Oceananigans 0.107.6. Synced with `CliMA/Oceananigans.jl`
main as of the same date. PR #5427 (CommunicationBuffers swap fix) is included.
PR #5564 (conditional-advection on tripolar) is *not* — still open upstream.

The runs documented below were executed under v3. Earlier (v2 sha `62147c3`,
2026-04-13) runs gave essentially the same result — see "v2 vs v3" notes inline.

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
- Anything in our fork that diverged from upstream specifically in
  `DistributedComputations/halo_communication.jl` between the fork base and
  cutoff 2026-04-13.

## Results

Run on 2026-05-14, OM2-1, defaults `cgridtransports_wdiagnosed_centered2_AB2`,
TW=1968-1977. Both `v2` (sha `62147c3`, 2026-04-13) and `v3` (sha `91a26ad`,
2026-05-14, synced with main) were tested; v3 numbers are reported below and
v2 differences (where any) are flagged inline. **No upstream PR merged between
v2 and v3 closes the GPU rank-seam bug.**

### 1-year: GPU 1×2 vs 1×1 — **clear rank-seam contamination**

Compare jobs: 168312952 (v3, exit 0) / 168304605 (v2, exit 0). Plots in
[outputs/.../plots/compare_1x2_1year/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_1year)
(overwritten in place; current PNGs are v3).

Summary stats from final snapshot (wet cells only):

| metric | v3 | v2 |
|---|---|---|
| `max\|diff\|` | 3.83e-01 years | 2.05e-01 years |
| `mean\|diff\|` | (similar order) | 1.02e-04 years |
| RMS diff | (similar order) | 1.43e-03 years |
| `mean\|reldiff\|` | (similar order) | 8.60e-04 |

v3's slightly larger 1-year `max\|diff\|` is most likely roundoff amplified
through 5844 timesteps — the *structure* (Pacific seam stripe) and the
**10-step `max\|diff\|` are bit-identical** between v2 and v3 (see GPU diag
below). The bug is not closed by any v2→v3 upstream change.

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

**Velocities & free surface at iter 1** (v3, immediately after first timestep):

| field | max\|diff\| | mean\|diff\| | RMS |
|---|---|---|---|
| `u` surface | 0 | 0 | 0 |
| `v` surface | 0 | 0 | 0 |
| `w` k=51 (top) | 2.58e-10 | 1.44e-13 | 2.15e-12 |
| `w` k=50 | 2.17e-10 | 1.36e-13 | 1.87e-12 |

`u` and `v` are bit-identical at the surface. `w` differs only at machine-epsilon
levels — consistent with diagnostic-`w` recomputation order rather than transport-
field divergence. So the divergence does NOT show up in the velocity fields
themselves — it shows up in the **tracer** that the (bit-equal) velocities
advect through a (suspectedly mis-swapped) halo.

**zstar fields (`sigma_cc`, `dt_sigma`, `eta_n`) per-iter** — the script reports
"first divergence" iteration for each zstar field per rank:

| field | rank 0 first-diverge iter | rank 1 first-diverge iter | overall max\|diff\| |
|---|---|---|---|
| `sigma_cc` | 0 | 0 | 5.36e-02 |
| `dt_sigma` | 487 | 487 | 2.78e-09 |
| `eta_n` | 0 | 0 | 9.73e-01 |

`sigma_cc` and `eta_n` showing non-zero diff at iter 0 looks suspicious (the
initial state should be bit-identical). The most likely explanation is that the
compare script slices the serial state with even-partition assumptions and the
fields are saved with halos — so the "diff at iter 0" is really an artifact of
comparing serial-with-halos against distributed-without-halos (or vice versa).
This needs follow-up before drawing zstar conclusions; the **tracer-side seam
signal** above is the load-bearing evidence.

### Diag (10 steps): CPU 1×2 vs 1×1 — **bit-identical (v2 and v3)**

Compare jobs: 168312950 (v3) / 168306896 (v2), both exit 0. Plots:
[outputs/.../plots/compare_1x2_diag_cpu/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag_cpu).

```
max|diff|    = 0.00e+00 years
mean|diff|   = 0.00e+00 years
RMS diff     = 0.00e+00 years
u/v/w/eta surface: all zero
```

On pure CPU MPI the run is **bit-identical** between 1×1 and 1×2 across all
10 iterations, on both v2 and v3. The "FIRST DIVERGENCE at iter 0" entries
for `sigma_cc` and `eta_n` reproduce *byte-identical* values across CPU/GPU
**and** v2/v3 compares — so they are a comparison-script artefact, not a real
signal.

### Diag (10 steps): GPU 1×2 vs 1×1 — **rank-seam already visible after 10 steps; v2 and v3 bit-identical**

Compare jobs: 168312951 (v3) / 168306897 (v2), both exit 0. Plots:
[outputs/.../plots/compare_1x2_diag/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag).

```
max|diff|    = 1.52e-03 years     (~ 13 hours of mis-aging in 10 steps)
RMS diff     = 1.31e-05 years
u/v/eta surface: bit-identical
w surface:    max|diff| = 4.54e-11 years (FP roundoff, v3)
                       = 4.56e-11 years                  (v2)
```

The **v2 and v3 `max\|diff\|` values match to the last printed digit (1.52e-03 yr)**.
That's strong evidence the bug is deterministic and unchanged by the upstream
merge — the v2→v3 sync brings no fix for it.

Interior age diff at z=1030 m, iter 10 — colorscale is ±5e-7 yr; the faint
blue stripe at j≈150 across the Pacific is already aligned at the rank seam
and is the same structure that grew to the bold blue stripe in the 1-year
plot above:

![GPU diag, iter 10, 1000 m slice](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/plots/compare_1x2_diag/diff_1x2_diag_centered2_iter10_slice_1000m.png)

### Jobs submitted

v3 run (2026-05-14, sha `91a26ad`), all exit 0:

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

v2 run for reference (2026-05-14, sha `62147c3`), all exit 0:
168304275, 168304276, 168306870, 168306873, 168304301, 168304352,
168306896, 168306897, 168304605.

## Interim conclusion

The 1×2 GPU runs produce a clear, localised tracer mismatch at the rank-rank
seam — already visible after 10 steps in `diag` and large enough to be
prominent in the 1-year run (max `|diff|` ≈ 0.2–0.4 yr in deep wet cells, with
structure visible across the Pacific at z≈1000 m). The CPU 1×2 run is
**bit-identical** to CPU 1×1 across all 10 diag iterations.

- The bug is **GPU-specific**, not in the partitioner, MPI logic, or generic
  CPU advection.
- Velocity fields are bit-identical between serial and distributed (`u`/`v`
  exactly; `w` at FP roundoff). The dynamics is fine; the **tracer halo
  exchange / fold-fill** on the GPU path is the load-bearing suspect.
- PR #5427 was initially hypothesised but is ruled out (Hypothesis 1 above —
  the `Adapt.adapt_structure` swap is a no-op because individual buffer types
  adapt to `nothing`).
- **v2 → v3 upstream sync does not fix the bug.** The GPU diag 10-step
  `max\|diff\|` is bit-identical between v2 and v3, so none of the PRs merged
  into CliMA main between 2026-04-13 and 2026-05-14 close it.

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
   *byte-identical* non-zero diff at iter 0 in CPU vs GPU and v2 vs v3
   compares — almost certainly a compare-script slicing artefact (serial
   saved with halos, distributed saved without halos, or vice versa). Worth
   fixing before relying on those zstar plots forensically.
3. **Document, then iterate.** This doc is the workflow + first-pass results;
   subsequent runs (WENO sweep, 2×1 partition, bisect candidates) should add
   rows to the Results section here.
