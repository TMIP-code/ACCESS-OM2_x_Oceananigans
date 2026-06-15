# First converged OM2-01 periodic Newton–Krylov solve

**Headline:** switching the age tracer's advection scheme from `centered2` to
`UpwindBiased(order=1)` ("`upwind1`") cuts the periodic-NK Jacobian–vector-product
(JVP) count by ~3–6× and makes it roughly **resolution-independent**. This
unlocked the **first converged periodic-NK solution at 0.1° (ACCESS-OM2-01)** —
a configuration that with `centered2` had never completed even a single Newton
iteration within a 48 h GPU job.

Window: `1968-1977`. Date: 2026-06.

---

## 1. Background: what a JVP is here

The periodic NK solver drives `G(x) = Φ(x) − x = 0`, where `Φ` (`Φ!` in the logs)
is one year of GPU forward integration of the age tracer through the archived
monthly velocities. `Φ!` is called in two roles:

- **once per Newton iteration** to evaluate the residual `G` (logged as
  `Φ! call #k (source_rate=1.0)`, followed by a `G! residual` block with the
  drift norms), and
- **once per GMRES inner iteration** as a **JVP** (the linearised tracer, logged
  `Φ! call #k (source_rate=0.0)`).

So "JVPs per Newton iteration" = the GMRES iterations needed to compute that
Newton step, and the total JVP count is the dominant cost (each `Φ!` is a full
1-year integration). The preconditioner is a coarsened transport matrix
(`LUMP_AND_SPRAY` / `Q{di}x{dj}`); it only changes *how fast* GMRES converges,
not the fixed point. The interleaving is reconstructed by
[`scripts/analysis/nk_jvp_sequence.py`](../scripts/analysis/nk_jvp_sequence.py).

Across every converged solve the per-iteration JVP count follows the same shape
`[a, ~1.5–2a, ~a, 0, 0, 0]` (peaks at Newton iter 2, collapses to 0 once the
residual is below tolerance), while `vol_rms_drift` converges quadratically.

---

## 2. JVP cost vs advection scheme and resolution

All `1968-1977`, AB2, Pardiso preconditioner. ΣJVP = total JVPs to convergence;
final/min `vol_rms_drift` in years.

| Model | κH | scheme | precond | JVP / Newton iter | ΣJVP | min drift |
|---|---|---|---|---|---|---|
| OM2-1 | 300 | centered2 | const | [17, 28, 17] | ~62 | ~1e-7 |
| OM2-1 | 300 | **upwind1** | const | [12, 19, 11] | **42** | 6.2e-8 |
| OM2-1 | 300 | **upwind1** | avg | [10, 15, 8] | **33** | 6.4e-8 |
| OM2-025 | 75 | centered2 | const | [60, 123, 100] | ~283 | ~1e-7 |
| OM2-025 | 75 | **upwind1** | const | [13, 21, 13] | **47** | 2.5e-8 |
| OM2-025 | 75 | **upwind1** | avg | [11, 15, 10] | **36** | 2.5e-8 |
| OM2-01 | 30 | centered2 | const | never completed Newton iter 1 in 48 h | — | did not converge |
| OM2-01 | 30 | **upwind1** | avg | [13, 20, 9] then restart [13] | **~55** | **3.0e-9 ✓** |

Two robust patterns:

1. **upwind1 is much cheaper than centered2** at fixed resolution and diffusivity
   — ~30% fewer JVPs at OM2-1 (42 vs ~62), and **~6× fewer** at OM2-025 (47 vs
   ~283).
2. **upwind1 makes the JVP cost ~resolution-independent.** With `centered2` the
   cost blew up ~4.5× from 1° → 0.25° (≈62 → ≈283); with `upwind1` the two are
   nearly equal (42 vs 47), and even 0.1° lands in the same band (~13–20 JVPs per
   Newton iteration).

The averaged-matrix preconditioner ("avg") consistently beats the constant one
at every resolution (e.g. OM2-1 33 vs 42; OM2-025 36 vs 47) — it is a better
match to the seasonally-varying forward operator.

> Note on diffusivity: separately, *reducing* OM2-025's diffusivity from the
> OM2-1-matched values (κH300) to κH75 had ~2.5×'d the `centered2` JVP cost
> (≈110 → ≈283). `upwind1` removes the need to compensate with higher diffusion
> at finer resolution — it converges at the low κH75/κH30 values directly.

---

## 3. Why upwind1 helps (mechanism)

`UpwindBiased(order=1)` carries implicit numerical diffusion. That diffusion
better-conditions the Jacobian `∂G/∂x`, so each GMRES solve needs far fewer JVPs
to reach the requested tolerance. `centered2` is energy-conserving but leaves a
stiffer, worse-conditioned operator — at fine resolution the first GMRES solve
alone needs more JVPs than fit in a 48 h job.

`upwind1` needs only halo ≥ 1 (vs WENO5's ≥ 4), so no grid rebuild is required.

---

## 4. How upwind1 unlocked OM2-01

**With `centered2` (the prior state):** the OM2-01 NK solves were submitted
repeatedly on 4×H200 GPUs with 48 h walltime and a range of preconditioner
coarsenings. They all hit the walltime (exit `-29`) having reached only ~44–46
`Φ!` calls and **one** `G! residual` block — i.e. still inside the **first**
Newton iteration's GMRES (below the `gmres_restart = 50` cycle). The coarsest
preconditioners (Q5x5, Q4x4, Q4x3/Q3x4) factorized fine; the finest, Q3x3,
failed factorization (`Φ!` never reached 2). **No OM2-01 solve ever completed a
single Newton iteration.** This is the "OM2-01 NK currently blocked" caveat in
the cross-resolution paper.

**With `upwind1` + averaged-matrix preconditioner (Q4x4), κH30:**

- First run (`170345012`, 48 h, walltime-killed mid-`Φ!`): completed **3 Newton
  iterations** — `vol_rms_drift` fell `9.80e-1 → 1.20e-2 → 1.01e-6` with JVP/iter
  `[13, 20, 9]`. Saved `newton_iterate_01`, `_02`.
- Restart (`170893330`, `INITIAL_AGE=latest` from `newton_iterate_02`, exit 0 in
  19 h): residual dropped `1.01e-6 → 3.0e-9` and held. **Converged.**

So upwind1 turns "can't finish one Newton iteration in 48 h" into "converges in
3 Newton iterations + a short restart" — purely through better conditioning, at
unchanged (low) diffusivity. The converged steady-state age is saved at
`…/periodic/{MC}/1x4/NK_Q4x4/age_Pardiso_Q4x4.jld2` (14 GB);
`MC = cgridtransports_wparent_upwind1_AB2_kH30_kVML25e-3_kVBG75e-7_mkappaV_LBS`.

---

## 5. Enabling infrastructure: run1yr-free averaged-matrix build

The averaged-matrix preconditioner used to require a prior 1-year forward run to
supply per-month velocity snapshots. New
[`src/create_monthly_matrices.jl`](../src/create_monthly_matrices.jl) instead
reuses the constant-matrix build (`matrix_setup.jl`) one month at a time over the
**preprocessed monthly velocity FieldTimeSeries**, averaging the 12 Jacobians —
**no 1-year run needed**. Selected via `TMAVG_METHOD=monthly` (default) in
`build_TMavg.sh`. It inherits the const build's zero sea-surface-tendency
approximation (coarser than the snapshot method) but builds from preprocessing
alone.

At OM2-01 this is a serial 12-Jacobian job: sparsity detection once (~3.2 h) +
12 × ~34 min = **~10.2 h**, peak **986 GB** (megamem), producing a 39 GB
`avg/M.jld2`. `SAVE_INTERMEDIATE_MATRICES=no` skips the 12 per-month matrices
(12 × 39 GB) to fit disk.

---

## 6. Pipeline / tooling changes made this session

- `upwind1` advection scheme added (`config.jl` parser + whitelist;
  `ADVECTION_SCHEME=upwind1` → `UpwindBiased(order=1)`).
- `create_monthly_matrices.jl` + `TMAVG_METHOD` dispatch in `build_TMavg.sh`.
- Driver `TMSNAP_QUEUE/MEM/NCPUS` overrides and overridable
  `WALLTIME_TM_SNAPSHOT` (OM2-01 needs megamem/24 h, not the hardcoded
  normal/192 GB/8 h); `SAVE_INTERMEDIATE_MATRICES` toggle.
- Driver post-NK steps (`run1yrNK`/`combine1yr`/`ventilation`) collapsed to run
  **once** regardless of `TM_SOURCE` (the solution is preconditioner-independent),
  fixing the avg-branch plot dependencies and the `TM_SOURCE=both` output
  collision.
- `scripts/analysis/nk_jvp_sequence.py` — reconstructs the `Φ!`/`G!` interleaving
  per PM, groups by advection scheme + diffusivity config, `--converged-only`.

---

## 7. Key job IDs (1968-1977, reproducibility)

| Job | What | Result |
|---|---|---|
| 170302528 / 529 | OM2-1 upwind1 NK const / avg | converged, ΣJVP 42 / 33 |
| 170302535 / 536 | OM2-025 upwind1 NK const / avg | converged, ΣJVP 47 / 36 |
| 170345011 | OM2-01 upwind1 monthly avg-matrix build | exit 0, 10:13 h, 986 GB |
| 170345012 | OM2-01 upwind1 NK (avg, Q4x4) — first run | 3 Newton iters, drift→1.01e-6, walltime-killed |
| 170893330 | OM2-01 upwind1 NK restart (`INITIAL_AGE=latest`) | **converged, drift 3.0e-9** |
| 171174889–894 | OM2-01 run1yrNK → combine1yr → plotNK / ventilation → plotventilation | post-NK diagnostics |

Reproduce (matrix build → NK → restart):

```bash
# averaged-matrix build (megamem, run1yr-free) + NK on the avg preconditioner
PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=1968-1977 \
  ADVECTION_SCHEME=upwind1 TM_SOURCE=avg LUMP_AND_SPRAY=4x4 \
  TMSNAP_QUEUE=megamem TMSNAP_MEM=2990GB TMSNAP_NCPUS=48 \
  WALLTIME_TM_SNAPSHOT=24:00:00 SAVE_INTERMEDIATE_MATRICES=no \
  JOB_CHAIN=TMsnapshot-NK bash scripts/driver.sh

# if the NK solve hits walltime mid-solve, restart from the latest Newton iterate
PARENT_MODEL=ACCESS-OM2-01 TIME_WINDOW=1968-1977 \
  ADVECTION_SCHEME=upwind1 TM_SOURCE=avg LUMP_AND_SPRAY=4x4 \
  INITIAL_AGE=latest JOB_CHAIN=NK bash scripts/driver.sh
```

---

## 8. Open questions / next steps

- Confirm the OM2-01 post-NK diagnostics (run1yrNK → combine1yr → plotNK /
  ventilation → plotventilation) and inspect the age field for physical sanity.
- Does `upwind1` change the *simulated* ventilation enough to matter vs
  `centered2`, or only the solver cost? (Cross-resolution comparison.)
- Extend to the second forcing window (`1999-2008`) at OM2-01.
- Whether the avg-vs-const preconditioner edge persists / grows at 0.1°.
