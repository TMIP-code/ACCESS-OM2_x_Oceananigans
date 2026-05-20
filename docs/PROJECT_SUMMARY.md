# Project Summary — ACCESS-OM2 × Oceananigans

*Last updated: 2026-05-13. Companion narrative to [PROGRESS.md](../PROGRESS.md) (status tables) and [AGENTS.md](../AGENTS.md) (technical reference).*

## 1. Motivation

The deep ocean ventilates on timescales of centuries to millennia. The **age** of a water parcel — the time since it last touched the surface — and its **adjoint age** — the expected time until it next reaches the surface — are two of the cleanest summary diagnostics of that ventilation. Combined, they yield directly meaningful quantities such as the **surface ventilation flux per unit area** (Pasquier et al. 2024, *JGR-Oceans*, doi:10.1029/2024JC021043), which says where and how strongly the deep ocean is being refreshed. Source-region decompositions (**water-mass fractions**, e.g. what share of a grid cell originated as North Atlantic Deep Water vs. Antarctic Bottom Water) are a closely related target, not yet computed in this pipeline. All of these quantities respond to changes in dense-water formation around Antarctica, which is sensitive to Southern Ocean meltwater, winds, and stratification.

The long-term science target of this project is to ask: **how do altered meltwater forcings around Antarctica change the long-term ventilation of the global ocean?** The natural setting for this question is ACCESS-OM2-01 (0.1° eddy-resolving), where a set of meltwater perturbation experiments already exists in the ACCESS-NRI catalog (e.g. `01deg_jra55v13_ryf9091` control and its `weddell_up1` / `weddell_down2` / `qian_wthmp` perturbations; see `notebooks/ACCESS_simulations_summary.md`).

The work also sits inside the broader **Transport Matrix Intercomparison Project (TMIP)** programme, whose central object is the sparse **transport matrix** $M$ assembled offline from an archived circulation — once built, it gives the steady tracer field through a single direct linear solve $M\,x = b$. Transport matrices are extraordinarily efficient for any *given* circulation, but they have a resolution problem: at finer grids the matrix becomes hard to assemble and, more critically, hard to factorise. This pipeline is a natural extension. The Newton-Krylov solve does not require a tractable factorisation of the *full*-resolution transport matrix — only of a **coarsened** version, used as a preconditioner inside the inner GMRES iterations. That is what made the periodic-steady-state problem tractable at 0.25° here, where a pure transport-matrix approach was not.

## 2. Method

Equilibrating a deep-ocean tracer by brute force requires integrating a tracer-transport model for **~3000 years** to reach steady state. That is prohibitive at high resolution. The core idea of this project is to bypass that integration.

Let $\phi$ denote the one-year-periodic operator that advances a tracer field $x$ through one year of a yearly-repeating circulation,

$$\phi(x(t)) = x(t + 1\,\text{yr}).$$

A periodic steady state is a fixed point of $\phi$, equivalently a zero of

$$G(x) = \phi(x) - x.$$

Instead of iterating $x \leftarrow \phi(x)$ for thousands of years, we solve $G(x) = 0$ directly with **Newton-Krylov** (Newton's method, with GMRES used to invert the Jacobian-vector products). Convergence typically requires only $\mathcal{O}(100)$ evaluations of $\phi$ rather than $\mathcal{O}(3000)$ — a 25–40× saving.

For this to be practical, each evaluation of $\phi$ must itself be fast. We exploit two facts:

1. The ocean dynamics are **prescribed** from archived ACCESS-OM2 output (monthly climatologies of velocities, free surface, mixing coefficients). We only advect a passive tracer, so the dynamical CFL constraint that limits ACCESS-OM2 does not bind — the timestep can be considerably larger.
2. The forward integration runs **offline on GPUs** through [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), which is far cheaper than rerunning ACCESS-OM2 itself.

The Jacobian needed by Newton-Krylov is built once at the start of the solve, on CPU, by automatic differentiation through a sparsified transport matrix $M$ assembled from year-averaged velocity fields. The same $M$ also gives a useful steady-state age estimate on its own (via a single direct linear solve), and serves as a preconditioner inside the periodic solve.

The same pipeline also solves the **adjoint** problem through a single configuration flag (`TRAF=yes`, *Time-Reversed Adjoint Flow*): every monthly velocity FieldTimeSeries is reversed in time, $u$ and $v$ are sign-flipped, and the adjoint transport matrix is constructed analytically as $V^{-1} M^\top V$ from the forward $M$ instead of being rebuilt from scratch. Newton-Krylov is then run on this reversed problem to obtain the **adjoint age**, and combined with the forward age this yields the surface ventilation diagnostic above. Implementation details and the converged-run inventory are in [docs/TRAF_simulations.md](TRAF_simulations.md).

## 3. Progress

The pipeline is now end-to-end functional at two of the three ACCESS-OM2 resolutions.

**What works end-to-end:**

| | ACCESS-OM2-1 (1°) | ACCESS-OM2-025 (0.25°) | ACCESS-OM2-01 (0.1°) |
|---|---|---|---|
| Tripolar grid built offline | ✓ | ✓ | ✓ |
| B-grid → C-grid velocity regrid (monthly climatology + yearly average) | ✓ | ✓ | ✓ |
| 1-year offline simulation on GPU | ✓ (~60 s) | ✓ (~480 s on H200) | ✓ (~1.7 h on 4× H200) |
| Transport-matrix build (CPU, sparse AD) | ✓ | ✓ (~1 h 13 min) | not yet |
| Steady-state age from $M$ (direct solve, CPU/GPU) | ✓ | ✓ | not yet |
| **Newton-Krylov periodic solve (forward age)** | ✓ (~70–73 $\phi$ calls) | ✓ (~116–120 $\phi$ calls) | not yet |
| **Newton-Krylov periodic solve (adjoint age via TRAF)** | ✓ (2 TWs) | ✓ (2 TWs) | not yet |
| Plots: AABW / MOC / zonal-mean age | ✓ | ✓ | partial |

**Pipeline components delivered along the way** (each shared across all three resolutions):

- A model-agnostic PBS driver (`scripts/driver.sh`) wiring together ~15 pipeline steps (preprocessing → grid → velocities → run → transport-matrix build → NK solve → plots), parameterised by parent model, experiment, time window, and a four-part configuration tag covering velocity source, $w$ formulation, advection scheme (centered-2 / WENO-3 / WENO-5), and timestepper (AB2 / split Runge-Kutta of order 2–5).
- Distributed multi-GPU support via MPI with explicit socket binding (modules `cuda/12.9.0` + `openmpi/5.0.8`).
- Three direct linear-solver backends for the Newton inner solve (Pardiso, ParU, UMFPACK on CPU; CUDSS on GPU), and an optional "lump-and-spray" preconditioner coarsening.
- Profiling infrastructure (Nsight Systems / NCU) and a documented workflow in `docs/profiling_workflow.md` + `docs/profiling_results.md`.

## 4. Where we are blocked, and what to do next

The honest status on ACCESS-OM2-01 is: **the 1-year forward operator runs**, but the cost is currently the binding constraint on doing Newton-Krylov on this grid. On 4× H200 GPUs one $\phi$ evaluation takes ~1.7 h, and going to 8 GPUs across a node boundary gives no speed-up (see `docs/OM2_01_NODE_SCALING_INVESTIGATION.md`). A 100-iteration NK solve would therefore cost ~170 GPU-hours per experiment, and we have not yet resolved the cross-node scaling issue.

Recent work on the `TIMESTEP_MULT` flag (see [docs/timestep_multiplier.md](timestep_multiplier.md) and [docs/timestep_multiplier_NK.md](timestep_multiplier_NK.md)) has paid off, with one important caveat: the NK fixed point can amplify per-step truncation error over the ~900-yr ventilation timescale, so 1-year forward-map stability is necessary but not always sufficient for NK stability. Current per-resolution NK defaults:

- **OM2-1**: **SRK3-M=12** (Δt = 18 h, 4× theoretical speedup), NK-validated for both the forward IAF and adjoint TRAF runs. AB2-M=4 also passes NK (2.6× wall-clock speedup) and is the standalone `run1yr` default.
- **OM2-025**: **SRK3-M=9** (Δt = 4.5 h, 3× speedup), lowered from the initial SRK3-M=12 candidate after that turned out to be NK-unstable for the 1999-2008 forcing window under both forward IAF and adjoint TRAF. M=9 converges cleanly across both computed time windows for both. AB2-M=3 is a conservative fallback.
- **OM2-01**: in the 1-year forward map each integrator has exactly one stable multiplier (AB2-M=2, SRK3-M=3, SRK5-M=6); AB2-M=2 is the production choice (2× speedup). NK has not yet been attempted at 0.1°.

Another option, not yet implemented, is to **precompute $w$** instead of diagnosing it on the fly. Diagnosing $w$ from continuity is currently a substantial fraction of per-step cost (estimated ~30%), and prescribing it from a preprocessed FieldTimeSeries would remove that overhead at the price of an extra preprocessing step. A plan exists in [plans/prescribe_w_for_performance.md](../plans/prescribe_w_for_performance.md).

**Near-term plan:**

1. **Cross-resolution comparison** of periodic-steady-state forward age, adjoint age, and the resulting surface ventilation diagnostic between ACCESS-OM2-1 and ACCESS-OM2-025 across the two computed time windows (1968-1977 and 1999-2008). The forward and adjoint NK solves have converged; the diagnostic and plotting passes are the remaining work.
2. Settle on a stable, validated `TIMESTEP_MULT` and partition layout for OM2-01.

**Medium-term plan:**

3. Bring OM2-01 control through the forward IAF NK pipeline.
4. Run TRAF on OM2-01 control to produce the adjoint age field.
5. Repeat (3)–(4) for at least one meltwater-perturbation experiment (e.g. `weddell_up1` or `qian_wthmp`) and compare ventilation against control.
6. Extend to water-mass-fraction tracers (multi-tracer NK).

## Thoughts on the OM2-01 "setback"

The OM2-01 setback is not a failure — it is a quantitative result about the cost of high-resolution offline-tracer modelling. The 1° and 0.25° results validate that the periodic-NK approach works, converges in tens to ~120 $\phi$ calls, and produces sensible ventilation diagnostics. Scaling that approach to 0.1° is a separate engineering problem (cross-node MPI, larger timesteps, possibly different preconditioning), and the cross-resolution comparison at 1° vs. 0.25° is itself a meaningful answer to *"how much does resolution change our picture of deep ventilation?"* — a question the meltwater study would have had to answer anyway.
