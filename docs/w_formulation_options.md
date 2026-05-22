# Vertical velocity (`w`) formulation choices

The age tracer simulation requires a 3D vertical velocity `w` consistent
with the (u, v, η) inputs. This project exposes three options for how
to provide `w`, controlled by the `W_FORMULATION` and `PRESCRIBED_W_SOURCE`
environment variables. They differ in **what `w` is**, **where it
comes from**, and **how it interacts with runtime cost and reproducibility**.

| `model_config` tag | `W_FORMULATION` | `PRESCRIBED_W_SOURCE` | `w` source |
|---|---|---|---|
| `wdiagnosed` | `wdiagnosed` | — | Recomputed from continuity at every model step (Oceananigans `DiagnosticVerticalVelocity`). |
| `wparent`    | `wprescribed` | `parent`     | Read from the parent MOM mass-transport, preprocessed monthly into `w_from_mass_transport_monthly.jld2` and interpolated cyclically. |
| `wprediag`   | `wprescribed` | `diagnosed`  | Read from a pre-computed file `w_diagnosed_monthly.jld2` that was produced by an earlier `diagnose_w` run (one-time `wdiagnosed` simulation with output writers enabled). |

The tag becomes part of the unified `MODEL_CONFIG` directory name (see
[shared_utils/config.jl::build_model_config](../src/shared_utils/config.jl)),
so outputs from different `w` choices never collide on disk.

---

## What each option does

### `wdiagnosed` — recompute every step

`HydrostaticFreeSurfaceModel` is built with a
`DiagnosticVerticalVelocity` field. At every iteration, the continuity
kernel solves `∂w/∂z = −(∂u/∂x + ∂v/∂y) − ∂η/∂t` and writes `w` into a
persistent Field. The age tracer then advects with the freshly diagnosed
`w`.

- **Pros**: `w` is exactly consistent with the (u, v, η) seen by the
  tracer; no extra files; no preprocessing.
- **Cons**: the continuity kernel is one of the more expensive pieces of
  the per-step cost. On OM2-1 4× V100 it was historically ~65% of GPU
  time, motivating the `wprescribed` shortcut
  (see [plans/prescribe_w_for_performance.md](../plans/prescribe_w_for_performance.md)).

### `wparent` — read from MOM mass transport

The `prep_mass_transport` step preprocesses MOM's saved (u·dz·dy, v·dz·dx)
monthly mass-transport into Oceananigans (u, v, w) Fields, with `w`
reconstructed from divergence of the horizontal transports (i.e. solving
the same continuity equation MOM did, but offline).

The simulation loads `w_from_mass_transport_monthly.jld2` as a
`FieldTimeSeries` and interpolates it cyclically; the continuity kernel
is never launched inside the tracer loop.

- **Pros**: cheap inside the tracer loop (one `interp_fts!` per step);
  `w` is exactly what MOM saw.
- **Cons**: requires the parent MOM `mass_transport` outputs to be
  available; preprocessing time + disk space.

### `wprediag` — read from a previously diagnosed `w` snapshot

`src/diagnose_w.jl` performs one `wdiagnosed` run with an output writer
that saves the diagnosed `w` to `w_diagnosed_monthly.jld2`. Subsequent
tracer runs with `W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=diagnosed`
read that file instead of re-diagnosing.

- **Pros**: cheap at run time (same `interp_fts!` cost as `wparent`);
  reproduces the diagnosed-w solution without re-running continuity.
- **Cons**: requires a one-time `diagnose_w` step; **and the saved `w`
  must be consistent with the consuming config** — see "Reusability
  caveat" below.

---

## Cost comparison

### Per-step

`wdiagnosed` carries the continuity kernel on every iteration;
`wparent` and `wprediag` replace it with a single `interp_fts!(target,
w_ts, t)` call (broadcast over the parent storage; see
[shared_utils/data_loading.jl::interp_fts!](../src/shared_utils/data_loading.jl)).
On OM2-1 4× V100, `wprescribed`-class (parent / prediag) used to win by
factors of ~2×–3× — that win came mostly from skipping the continuity
solve.

### End-to-end 1-year benchmark (no I/O, `run1yrfast`)

OM2-1 was the cheapest cross-check; here we report the OM2-01
`mkappaV` case where we recently shrank a separate per-step allocation
hot spot:

| Config | Job | 1-yr wall (timed) | Notes |
|---|---|---|---|
| OM2-01 `wparent` `mkappaV` (no alloc-fix) | 169009208 | 4368 s | Per-step FTS-derived κV callback allocated a fresh Field each step. |
| OM2-01 `wparent` (no `mkappaV`)           | 169009209 | 1713 s | Same code, MLD-derived κV callback off; lower bound for this config. |
| OM2-01 `wparent` `mkappaV` (alloc-fix)    | 169027661 | **1815 s** | After [6ed0c55](https://github.com/) `interp_fts!` skips the per-step Field allocation. |

→ The alloc-fix recovered ~99% of the κV-callback gap. The remaining
~100 s of the 1815 vs 1713 spread is the real κV broadcast work
(non-allocating but non-zero).

---

## Correctness: NK age side-by-side (OM2-1, 1968-1977, AB2 mkappaV DTx4)

Newton–Krylov periodic-steady-state age for the three configurations on
the same model + experiment + time window. Run via `JOB_CHAIN=TMbuild-
TMsolve-NK-run1yrNK-plotNK` per config; analysis via
[test/probe_nk_age_diff.jl](../test/probe_nk_age_diff.jl).

| Config | NK age — global wet-cell mean (yr) | min | max | NK solve wall | Job |
|---|---|---|---|---|---|
| `wdiagnosed` | **390.37** | −110.87 | 2371.98 | 21m 52s | 169044444 |
| `wparent`    | **390.38** | −110.89 | 2371.96 | **17m 38s** | 169044450 |
| `wprediag`   | **390.35** | −110.83 | 2372.15 | 18m 36s | 169044457 |

NK solve = the `NK_c` step alone (Newton–Krylov GMRES on the
periodic-time-marching residual; gpuvolta, 1× V100, Pardiso linear
solver, LUMP_AND_SPRAY preconditioner). The `wdiagnosed` solve is
~24% slower than `wparent` because each Φ! evaluation runs the
continuity kernel that `wparent`/`wprediag` skip; `wprediag` lands
between the two (continuity skipped, but the saved `w` requires the
same per-step FTS interpolation as `wparent`).

(Slight negative ages are the well-known centered2-advection
undershoots common to all three.)

### Pairwise differences (yr, all wet cells)

| Pair (A − B) | mean(A − B) | RMS(A − B) | max\|A − B\| |
|---|---|---|---|
| `wdiagnosed − wparent`  | −0.0065 | **0.028** | 0.37 |
| `wdiagnosed − wprediag` | +0.0222 | **0.141** | 2.86 |
| `wparent − wprediag`    | +0.0287 | **0.147** | 2.84 |

**Interpretation**:

- `wparent` and `wdiagnosed` agree to **0.028 yr RMS** — ~0.007% of
  mean age. For practical purposes they are indistinguishable on this
  configuration. This is the expected outcome: both ultimately solve
  the same continuity equation, one inside MOM, one inside Oceananigans.
- `wprediag` differs from both by **~5× more** (0.14 yr RMS, max ~2.8 yr).
  See the next section.

---

## Reusability caveat for `wprediag`

`w_diagnosed_monthly.jld2` is saved at monthly cadence from a
`wdiagnosed` run. In principle the diagnosed `w` depends only on
`(u, v, η)` (continuity is independent of κV and Δt), so the file
should be re-usable across `mkappaV`/`DTx<N>` variants of the same
`(model, experiment, time_window)`.

The OM2-1 result above (`wdiagnosed − wprediag` ~5× larger than
`wdiagnosed − wparent`) suggests the saved file we used was generated
under a slightly different config than the consuming run. Concretely the
file was produced by an earlier `diagnose_w` invocation whose
provenance isn't recorded in the JLD2 (`GIT_COMMIT` of the `diagnose_w`
job is not stored alongside `w`).

If you need bit-exact `wprediag ≡ wdiagnosed` for a given consuming
config, re-run `JOB_CHAIN=diagnose_w` with the matching env immediately
before the comparison. Otherwise treat `wprediag` as "close to
diagnosed, but not identical".

---

## Required files on disk

| Option | Required `preprocessed_inputs/{PM}/{EXP}/{TW}/monthly/` files |
|---|---|
| `wdiagnosed` | `u_*_monthly.jld2`, `v_*_monthly.jld2`, `eta_monthly.jld2` |
| `wparent`    | + `w_from_mass_transport_monthly.jld2` |
| `wprediag`   | + `w_diagnosed_monthly.jld2` (from a prior `diagnose_w` run) |

Mixed runs that toggle `MONTHLY_KAPPAV=yes` additionally need
`kappa_v_monthly.jld2` (built by `JOB_CHAIN=clo`); none of that depends
on the `w` choice.

---

## Choosing between them

- **Quick exploration / debugging / single rank**: `wdiagnosed`.
  No preprocessing dependencies, exact consistency with the active
  (u, v, η) FTS.
- **Distributed GPU production / NK solves**: `wparent` (cheapest
  in the tracer loop; agrees with `wdiagnosed` to ≤ 0.03 yr RMS NK age).
- **Reproducing a prior `wdiagnosed` solution without re-running
  continuity**: `wprediag`, but only if `w_diagnosed_monthly.jld2`
  was produced under the same config you're consuming under.

For new work in this project, **`wparent` is the recommended default**:
it sidesteps the per-step continuity kernel cost and produces age
fields that are statistically identical to `wdiagnosed`.

---

## Related

- [plans/prescribe_w_for_performance.md](../plans/prescribe_w_for_performance.md)
  — original motivation (continuity ≈ 65% of GPU time on OM2-1 4× V100)
- [.claude/plans/i-want-to-run-ethereal-feigenbaum.md](../.claude/plans/i-want-to-run-ethereal-feigenbaum.md)
  — comparison plan executed to produce the numbers above
- [src/diagnose_w.jl](../src/diagnose_w.jl) — the one-time
  `w_diagnosed_monthly.jld2` producer
- [src/setup_model.jl](../src/setup_model.jl) (search for `W_FORMULATION`)
  — branching logic for the three options
- [test/probe_nk_age_diff.jl](../test/probe_nk_age_diff.jl) — script
  that produces the per-cell-pair age diff statistics in this doc
