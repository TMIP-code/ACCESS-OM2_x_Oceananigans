# Pipeline Progress Tracker

## ACCESS-OM2-1 / 1deg_jra55_iaf_omip2_cycle6

### TIME_WINDOW: 1960-1979

| w | Advection | Timestepper | Redi-GM | κV | Prep | 1yr Run (wall) | NK Solve (Φ calls) | NK Plots |
|---|-----------|-------------|---------|-----|------|----------------|-------------------|----------|
| diagnosed | centered2 | AB2 | - | - | grid + vel | ~1.8 min | 63 | yes |
| diagnosed | centered2 | AB2 | yes | - | grid + vel | 11.3 min | crashed (segfault after 1 Φ) | no |
| diagnosed | centered2 | AB2 | yes | yes | grid + vel | 11.7 min | crashed (segfault after 1 Φ) | no |
| diagnosed | centered2 | AB2 | - | yes | grid + vel | 3.0 min | 71 | no |
| parent | centered2 | AB2 | - | - | grid + vel | 1.8 min | 63 | no |
| prediag | centered2 | AB2 | - | - | grid + vel | 1.8 min | 63 | no |
| diagnosed | centered2 | SRK2 | - | - | grid + vel | - | - | - |
| diagnosed | centered2 | SRK3 | - | - | grid + vel | - | - | - |
| diagnosed | centered2 | SRK4 | - | - | grid + vel | - | - | - |
| diagnosed | centered2 | SRK5 | - | - | grid + vel | - | - | - |

---

## ACCESS-OM2-025 / 025deg_jra55_iaf_omip2_cycle6

### TIME_WINDOW: 1960-1979

| w | Advection | Timestepper | Redi-GM | κV | Prep | 1yr Run (wall) | NK Solve (Φ calls) | NK Plots |
|---|-----------|-------------|---------|-----|------|----------------|-------------------|----------|
| diagnosed | centered2 | AB2 | - | - | grid + vel | ~8.8 min | 110 | yes |

---

## Multi-TIME_WINDOW runs (wprediag + κV)

Config: `cgridtransports_wprediag_centered2_AB2_mkappaV`
Full plan: `.claude/plans/compiled-seeking-mountain.md`

> **TODO:** Also run with Redi-GM enabled (`GMREDI=true`). Additionally, test a config *without* Redi-GM but with mass transports preprocessed to include GM velocities (i.e., GM contribution baked into the velocity field rather than parameterized online).

### ACCESS-OM2-1 (1deg)

| # | TIME_WINDOW | Label | prep | vel | clo | diagw | 1yr Run (wall) | TM Build | NK Solve (Φ calls) | NK Plots |
|---|-------------|-------|------|-----|-----|-------|----------------|----------|-------------------|----------|
| 1 | `1958-1987` | First 30yr | ✓ | ✓ | ✓ | ✓ | ✓ (10 min) | ✓ | ✓ (70 Φ) | ✓ |
| 2 | `1989-2018` | Last 30yr | ✓ | ✓ | ✓ | ✓ | ✓ (10 min) | ✓ | ✓ (73 Φ) | ✓ |
| 3 | `1958-1977` | First 20yr | ✓ | ✓ | ✓ | ✓ | ✓ (10 min) | ✓ | ✓ (71 Φ) | ✓ |
| 4 | `1999-2018` | Last 20yr | ✓ | ✓ | ✓ | ✓ | ✓ (8 min) | ✓ | ✓ (73 Φ) | ✓ |
| 5 | `1958-1967` | First 10yr | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (72 Φ) | ✓ |
| 6 | `2009-2018` | Last 10yr | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (74 Φ) | ✓ |
| 7 | `1963-1972` | Max AABW 10yr (-16.6 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (73 Φ) | ✓ |
| 8 | `2006-2015` | Min AABW 10yr (-11.7 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (75 Φ) | ✓ |
| 9 | `1964-1966` | Max AABW 3yr (-17.9 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (8 min) | ✓ | ✓ (76 Φ) | ✓ |
| 10 | `1998-2000` | Min AABW 3yr (-10.7 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (75 Φ) | ✓ |
| 11 | `1964` | Max AABW 1yr (-18.7 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (80 Φ) | ✓ |
| 12 | `2010` | Min AABW 1yr (-9.3 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (9 min) | ✓ | ✓ (80 Φ) | ✓ |

### ACCESS-OM2-025 (0.25deg)

| # | TIME_WINDOW | Label | prep | vel | clo | diagw | 1yr Run (wall) | TM Build | NK Solve (Φ calls) | NK Plots |
|---|-------------|-------|------|-----|-----|-------|----------------|----------|-------------------|----------|
| 1 | `1958-1987` | First 30yr | ✓ | ✓ | ✓ | ✓ | Q | ✓ (1h13m) | Q | - |
| 2 | `1989-2018` | Last 30yr | ✓ | ✓ | ✓ | ✓ | ✓ (20 min) | ✓ (1h13m) | Q | - |
| 3 | `1958-1977` | First 20yr | ✓ | ✓ | ✓ | ✓ | Q | ✓ (1h14m) | Q | - |
| 4 | `1999-2018` | Last 20yr | ✓ | ✓ | ✓ | ✓ | ✓ (21 min) | ✓ (1h12m) | Q | - |
| 5 | `1958-1967` | First 10yr | ✓ | ✓ | ✓ | ✓ | Q | ✓ (1h15m) | Q | - |
| 6 | `2009-2018` | Last 10yr | ✓ | ✓ | ✓ | ✓ | ✓ (21 min) | ✓ (1h19m) | Q | - |
| 7 | `1963-1972` | Max AABW 10yr (-10.3 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (21 min) | FAIL (OOM) | - | - |
| 8 | `2006-2015` | Min AABW 10yr (-6.8 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (20 min) | R (18m) | H | - |
| 9 | `1980-1982` | Max AABW 3yr (-11.2 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (20 min) | R | H | - |
| 10 | `1998-2000` | Min AABW 3yr (-6.4 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (22 min) | Q | H | - |
| 11 | `1980` | Max AABW 1yr (-12.8 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (20 min) | Q | H | - |
| 12 | `1998` | Min AABW 1yr (-5.3 Sv) | ✓ | ✓ | ✓ | ✓ | ✓ (22 min) | Q | Q | - |

AABW metric: min(ψ_depthspace) for lat ≤ 50°S. Plots: `outputs/{model}/{experiment}/AABW/`

---

## Legend

- **Prep**: `grid` = grid.jld2 built, `vel` = monthly + yearly velocity fields preprocessed
- **1yr Run (wall)**: Wall-clock time for `run_1year.jl` on a single GPU (V100 for 1deg, H200 for 025deg)
- **NK Solve (Φ calls)**: Number of 1-year forward simulations for Newton-GMRES to converge (Pardiso_LSprec)
- **NK Plots**: Plots from 1-year run initialized from periodic NK solution
- **κV**: Prescribed vertical diffusivity from parent model
