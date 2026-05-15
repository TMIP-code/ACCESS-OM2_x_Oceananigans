# Next probes: GPU seam tracer bug localised to `implicit_step!`

Plan file for the next debugging session. The bug we are chasing is
documented in [serial_vs_distributed_validation.md](serial_vs_distributed_validation.md);
this file picks up where that one left off.

## TL;DR — what we know as of 2026-05-15

**The bug is in `implicit_step!` (implicit vertical-diffusion tridiagonal
solve), on rank 1 only.**

Evidence (from [test/probe_tracer_tendency.jl](../test/probe_tracer_tendency.jl)
manual decomposition of `time_step!`, CPU 1×2 vs GPU 1×2, OM2-1 1968-1977,
`centered2_AB2` default config):

| stage in the iter 0 → iter 1 step | rank 0 `age` max\|diff\| | rank 1 `age` max\|diff\| | rank 1 seam row (parent y=14) |
|---|---|---|---|
| iter 0 inputs (age, u, v, w, η, σ, κV, metrics, bottom, …) | 0 (except `w` at 1 Float64 ULP) | 0 (except `w` at 1 Float64 ULP) | 0 |
| `compute_tracer_tendencies!` → `Gⁿ.age` | 0 | 0 | **0** ← bit-identical |
| `_ab2_step_tracer_field!` → age (`post_explicit`) | 1.8e-12 s (Float64 ULP, 120 cells) | 1.8e-12 s (Float64 ULP, 130 cells) | **0** ← bit-identical |
| `implicit_step!` → age (`post_implicit`) | 3.3e-11 s (Float64 ULP, 388K cells — clean) | **4534.677 s (250K cells)** | **4534.677 s in 3769 cells** ← BUG |
| `update_state!` → iter 1 age | 4534.677 s (494K cells, ULP-spread elsewhere) | 4534.677 s (340K cells) | 4534.677 s in 4266 cells |

So:

1. Everything **upstream** of `implicit_step!` is CPU↔GPU bit-identical
   to within Float64 ULP — including `Gⁿ.age`, the freshly computed
   tendency.
2. **`implicit_step!` itself** produces the 4535 s seam-row divergence on
   rank 1.
3. Rank 0 is clean (only Float64-ULP scatter from a slightly different
   tridiagonal-solver rounding path on GPU — harmless).
4. The asymmetry between rank 0 and rank 1 is the key. Rank 1 has
   y-topology `LeftConnectedRightFaceFolded` (south rank-rank seam +
   north tripolar fold). Rank 0 has `RightConnected` (south wall + north
   rank-rank seam).

## Probe artefacts on disk

Per-rank JLD2 in
`outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wdiagnosed_centered2_AB2/1x2/probe/`:

- `probe_tendency_{cpu|gpu}_iter{0,1}_rank{0,1}.jld2` — full state, ~1 GB
  each, includes `age`, `u_fts`/`v_fts`/`w`, `eta_fts`, `Gn_age`/`Gm_age`,
  `sigma_cc`/`eta_n`/`dt_sigma`, 9 grid-metric variants, `bottom_height`,
  `κV`, `closure_fields` (currently `(nothing, nothing)` for our setup).
- `probe_age_{cpu|gpu}_iter0_{post_explicit|post_implicit}_rank{0,1}.jld2` —
  lightweight age + Gⁿ_age snapshots straddling `implicit_step!`.

Total ~9 GB. **Safe to delete after each round** — they're regenerable in
~6 min wallclock via `JOB_CHAIN=probetend-probetendcpu`.

## Where the suspect code lives

The implicit-vertical-diffusion solver path used by our model:

- Top-level dispatcher: [Oceananigans `src/TurbulenceClosures/vertically_implicit_diffusion_solver.jl`](https://github.com/briochemc/Oceananigans.jl/blob/bp/offline_ACCESS-OM2_v3/src/TurbulenceClosures/vertically_implicit_diffusion_solver.jl)
  - `implicit_step!(field::Field, implicit_solver::BatchedTridiagonalSolver, closure, closure_fields, tracer_id, clock, model_fields, Δt)` — builds the per-column tri-diagonal coefficients (`κ` from the closure, mask, metrics), then dispatches to `BatchedTridiagonalSolver`.
- The actual solve: [Oceananigans `src/Solvers/batched_tridiagonal_solver.jl`](https://github.com/briochemc/Oceananigans.jl/blob/bp/offline_ACCESS-OM2_v3/src/Solvers/batched_tridiagonal_solver.jl)
  - `BatchedTridiagonalSolver` runs one Thomas-algorithm sweep per (i, j) column. GPU launches one kernel over (i, j) with serial vertical recursion inside each thread.

Two reasons the GPU+rank-1-only divergence might arise:

1. **Coefficient construction** reads diffusivity `κᶜᶜᶠ(i, j, k, grid, ...,
   closure, K, id)` and possibly grid metrics at face locations. If any
   of these inputs has a fold-row / partition-aware specialisation that
   misfires on a rank whose y-topology is
   `LeftConnectedRightFaceFolded`, the coefficients themselves end up
   different from the serial-CPU baseline.
2. **The solver kernel launch** parametrises over the active-cells map
   on GPU. If the active-cells map for rank 1 includes / excludes a
   different set of columns than CPU does, some columns get
   solved-with-junk on GPU.

## Probes for the next session

In rough priority order. All probes should keep the existing
PARTITION=1x2 CPU + GPU baseline from
[probe_tracer_tendency.jl](../test/probe_tracer_tendency.jl) as the
control.

### Probe A — `κV` halo at the seam (cheap; 0 new code)

The existing dumps already include `κV`. Read the rank-1 GPU vs CPU
`κV` parent-array at parent y = 13 (south halo), 14 (first interior),
and the matching north-halo rows; print min/max/n_differ. If CPU and
GPU disagree on `κV` at the seam-adjacent halo, the bug is upstream of
the solver, in the κV halo-fill path.

Expected outcome: clean. (`κV` is loaded once, partitioned by
`partition_data.jl`, and the existing probe-iter-0 dump showed all
inputs identical except `w`.) If it _is_ clean, this rules out
"wrong κV halo on rank 1" as the upstream cause.

Skeleton: `scripts/debugging/probe_kV_at_seam.jl` — open the iter-0
dump, slice `κV[:, [13, 14, ...], :]` on rank 1 for both arches, print
side-by-side.

### Probe B — disable implicit vertical diffusion (direct confirmation)

Toggle `κVField` (the closure's κ) to zero, or temporarily replace
`implicit_vertical_diffusion = VerticalScalarDiffusivity(...)` with
`nothing` for the age tracer. Rerun the 10-step diag pair (GPU 1×2 vs
CPU 1×2). If the seam tracer drift vanishes, that **confirms**
`implicit_step!` is the sole cause; if it persists, there is a second
mechanism we haven't found.

Cheapest path: add a `IMPLICIT_KAPPAV=no` env flag to
[src/setup_model.jl](../src/setup_model.jl) that drops the implicit
closure, tag outputs with a `_noKV` suffix so they don't clobber the
default runs, and submit via the existing `diag` step.

### Probe C — solver coefficients dump

Extend the probe to dump the **per-column tri-diagonal coefficients**
that `implicit_step!` constructs before the Thomas sweep, for every
(i, j) in rank 1's parent-y = 14 row. This requires either:

  - calling `implicit_step!`'s coefficient-construction kernels
    directly with the same arguments and dumping their outputs; or
  - monkey-patching `implicit_step!` to also write the LHS / RHS arrays
    to JLD2 before calling the BatchedTridiagonalSolver.

If the coefficients differ CPU vs GPU on rank 1 row 14 → the bug is in
the coefficient builder (probably a fold-row-aware κ lookup or
metric). If coefficients match but the solve output differs → the bug
is in `BatchedTridiagonalSolver` itself.

This is the most invasive probe but the most decisive one.

### Probe D — try a 2×1 partition (x-direction split)

If we partition along x instead of y, the rank-rank seam is purely in
i and the y-topology on each rank stays `Bounded` (no
`LeftConnectedRightFaceFolded` issue). If the seam bug **also** appears
on 2×1, the trigger is "any MPI partition", not the fold-row topology.
If 2×1 is clean, the trigger is specifically the fold-row topology
that rank 1 of 1×2 has.

Submit:
```bash
GPU_QUEUE=gpuvolta PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x1 \
    JOB_CHAIN=probetend-probetendcpu bash scripts/test_driver.sh
PARENT_MODEL=ACCESS-OM2-1 PARTITION=2x1 \
    JOB_CHAIN=compareprobe bash scripts/test_driver.sh
```

This relies on partitioned FTS data existing for 2×1. Check
`preprocessed_inputs/.../1968-1977/partitions/2x1/` first; if missing,
run the `partition` step.

### Probe E — substitute `BatchedTridiagonalSolver` with a CPU-only stub

For the GPU run, route the age tracer's `implicit_step!` through a CPU
solve (copy field to host, solve, copy back). If this fixes the seam
diff, the bug is in the GPU implementation of the solver itself
(probably in the kernel that reads/writes column data). If the diff
persists, the bug is upstream of the solver.

Lowest-cost expression of this idea is `IMPLICIT_KAPPAV=no` (Probe B),
which removes the call site entirely. Probe E is the "remove the
suspect, keep the rest of the physics" experiment; B is "remove the
suspect _and_ its contribution to physics."

## Stop conditions

The probe loop is done when one of:

- A specific Oceananigans upstream PR or local change makes the
  seam-row diff in `step 0→1, post_implicit, rank1` drop to Float64-ULP
  level (currently 4534.677 s, expected < 1e-10 s).
- We have a reproducible MWE we can hand off as an Oceananigans
  upstream issue.

## Hygiene

Probe artefacts are ~9 GB per round. The existing probe always writes
to the same path so reruns clobber old files. **Do not** push probe
JLD2s into the repo; the existing `.gitignore` already excludes
`outputs/`.
