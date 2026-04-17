# Pipeline Progress Tracker

The six TIME_WINDOWs tracked here are derived from the parent forcing period 1958–2018:
- `1958-1987` — First 30yr
- `1989-2018` — Last 30yr
- `1968-1977` — Mid 10yr of first 30yr
- `1999-2008` — Mid 10yr of last 30yr
- `1972` — Mid 1yr of first 30yr
- `2003` — Mid 1yr of last 30yr

## AABW Timeseries

Depth-space AABW transport timeseries with selected TIME_WINDOWs highlighted.
Plots from `outputs/{model}/{experiment}/AABW/`.

### ACCESS-OM2-1 (1deg)

![min ψ, lat ≤ 50°S](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/AABW/AABW_50S_depthspace_timeseries.png)
![min ψ, lat ≤ 60°S](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/AABW/AABW_60S_depthspace_timeseries.png)
![min ψ, lat ≤ 0°, depth ≥ 3000m](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/AABW/AABW_deep_depthspace_timeseries.png)

### ACCESS-OM2-025 (0.25deg)

![min ψ, lat ≤ 50°S](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/AABW/AABW_50S_depthspace_timeseries.png)
![min ψ, lat ≤ 60°S](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/AABW/AABW_60S_depthspace_timeseries.png)
![min ψ, lat ≤ 0°, depth ≥ 3000m](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/AABW/AABW_deep_depthspace_timeseries.png)

---

## Total MOC (resolved + GM)

Plots from `outputs/{model}/{experiment}/{TW}/MOC/MOC_total_mean.png`.

### ACCESS-OM2-1 (1deg)

| # | TIME_WINDOW | Label | Total MOC |
|---|-------------|-------|-----------|
| 1 | `1958-1987` | First 30yr | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/MOC/MOC_total_mean.png) |
| 2 | `1989-2018` | Last 30yr | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/MOC/MOC_total_mean.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/MOC/MOC_total_mean.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/MOC/MOC_total_mean.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/MOC/MOC_total_mean.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/MOC/MOC_total_mean.png) |

### ACCESS-OM2-025 (0.25deg)

| # | TIME_WINDOW | Label | Total MOC |
|---|-------------|-------|-----------|
| 1 | `1958-1987` | First 30yr | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/MOC/MOC_total_mean.png) |
| 2 | `1989-2018` | Last 30yr | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/MOC/MOC_total_mean.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/MOC/MOC_total_mean.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/MOC/MOC_total_mean.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/MOC/MOC_total_mean.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/MOC/MOC_total_mean.png) |

---

## Resolved MOC (no GM)

Plots from `outputs/{model}/{experiment}/{TW}/MOC/MOC_resolved_mean.png`.

### ACCESS-OM2-1 (1deg)

| # | TIME_WINDOW | Label | Resolved MOC |
|---|-------------|-------|--------------|
| 1 | `1958-1987` | First 30yr | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/MOC/MOC_resolved_mean.png) |
| 2 | `1989-2018` | Last 30yr | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/MOC/MOC_resolved_mean.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/MOC/MOC_resolved_mean.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/MOC/MOC_resolved_mean.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/MOC/MOC_resolved_mean.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/MOC/MOC_resolved_mean.png) |

### ACCESS-OM2-025 (0.25deg)

| # | TIME_WINDOW | Label | Resolved MOC |
|---|-------------|-------|--------------|
| 1 | `1958-1987` | First 30yr | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/MOC/MOC_resolved_mean.png) |
| 2 | `1989-2018` | Last 30yr | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/MOC/MOC_resolved_mean.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/MOC/MOC_resolved_mean.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/MOC/MOC_resolved_mean.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/MOC/MOC_resolved_mean.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/MOC/MOC_resolved_mean.png) |

---

## GM MOC

Plots from `outputs/{model}/{experiment}/{TW}/MOC/MOC_gm_mean.png`.

### ACCESS-OM2-1 (1deg)

| # | TIME_WINDOW | Label | GM MOC |
|---|-------------|-------|--------|
| 1 | `1958-1987` | First 30yr | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/MOC/MOC_gm_mean.png) |
| 2 | `1989-2018` | Last 30yr | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/MOC/MOC_gm_mean.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/MOC/MOC_gm_mean.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/MOC/MOC_gm_mean.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/MOC/MOC_gm_mean.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![MOC](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/MOC/MOC_gm_mean.png) |

### ACCESS-OM2-025 (0.25deg)

| # | TIME_WINDOW | Label | GM MOC |
|---|-------------|-------|--------|
| 1 | `1958-1987` | First 30yr | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/MOC/MOC_gm_mean.png) |
| 2 | `1989-2018` | Last 30yr | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/MOC/MOC_gm_mean.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/MOC/MOC_gm_mean.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/MOC/MOC_gm_mean.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/MOC/MOC_gm_mean.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![MOC](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/MOC/MOC_gm_mean.png) |

---

## Multi-TIME_WINDOW runs (wprediag + κV)

Config: `cgridtransports_wprediag_centered2_AB2_mkappaV`
Full plan: `.claude/plans/compiled-seeking-mountain.md`

> **TODO:** Also run with Redi-GM enabled (`GMREDI=true`). Additionally, test a config *without* Redi-GM but with mass transports preprocessed to include GM velocities (i.e., GM contribution baked into the velocity field rather than parameterized online).

### ACCESS-OM2-1 (1deg)

| # | TIME_WINDOW | Label | prep | vel | clo | diagw | 1yr Run | Benchmark (sim) | TM Build | NK Solve (Φ calls) | NK Plots |
|---|-------------|-------|------|-----|-----|-------|---------|-----------------|----------|-------------------|----------|
| 1 | `1958-1987` | First 30yr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (60s) | ✓ | ✓ (70 Φ) | ✓ |
| 2 | `1989-2018` | Last 30yr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (61s) | ✓ | ✓ (73 Φ) | ✓ |
| 3 | `1968-1977` | Mid 10yr (first 30) | - | - | - | - | - | - | - | - | - |
| 4 | `1999-2008` | Mid 10yr (last 30) | - | - | - | - | - | - | - | - | - |
| 5 | `1972` | Mid 1yr (first 30) | - | - | - | - | - | - | - | - | - |
| 6 | `2003` | Mid 1yr (last 30) | - | - | - | - | - | - | - | - | - |

### ACCESS-OM2-025 (0.25deg)

| # | TIME_WINDOW | Label | prep | vel | clo | diagw | 1yr Run | Benchmark (sim) | TM Build | NK Solve (Φ calls) | NK Plots |
|---|-------------|-------|------|-----|-----|-------|---------|-----------------|----------|-------------------|----------|
| 1 | `1958-1987` | First 30yr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (478s) | ✓ (1h13m) | ✓ (120 Φ) | ✓ |
| 2 | `1989-2018` | Last 30yr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (472s) | ✓ (1h13m) | ✓ (116 Φ) | ✓ |
| 3 | `1968-1977` | Mid 10yr (first 30) | - | - | - | - | - | - | - | - | - |
| 4 | `1999-2008` | Mid 10yr (last 30) | - | - | - | - | - | - | - | - | - |
| 5 | `1972` | Mid 1yr (first 30) | - | - | - | - | - | - | - | - | - |
| 6 | `2003` | Mid 1yr (last 30) | - | - | - | - | - | - | - | - | - |

---

## Basin Zonal Mean Age (periodic mean)

Plots from `{TW}/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/`

### ACCESS-OM2-1 (1deg)

| # | TIME_WINDOW | Label | Global | Atlantic | Pacific | Indian |
|---|-------------|-------|--------|----------|---------|--------|
| 1 | `1958-1987` | First 30yr | ![global](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 2 | `1989-2018` | Last 30yr | ![global](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![global](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![global](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![global](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![global](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |

### ACCESS-OM2-025 (0.25deg)

| # | TIME_WINDOW | Label | Global | Atlantic | Pacific | Indian |
|---|-------------|-------|--------|----------|---------|--------|
| 1 | `1958-1987` | First 30yr | ![global](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1958-1987/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 2 | `1989-2018` | Last 30yr | ![global](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1989-2018/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 3 | `1968-1977` | Mid 10yr (first 30) | ![global](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1968-1977/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 4 | `1999-2008` | Mid 10yr (last 30) | ![global](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1999-2008/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 5 | `1972` | Mid 1yr (first 30) | ![global](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/1972/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| 6 | `2003` | Mid 1yr (last 30) | ![global](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![atlantic](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![pacific](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![indian](outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/2003/periodic/cgridtransports_wprediag_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |

---

## Legend

- **Prep**: `grid` = grid.jld2 built, `vel` = monthly + yearly velocity fields preprocessed
- **1yr Run (wall)**: Wall-clock time for `run_1year.jl` on a single GPU (V100 for 1deg, H200 for 025deg)
- **NK Solve (Φ calls)**: Number of 1-year forward simulations for Newton-GMRES to converge (Pardiso_LSprec)
- **NK Plots**: Plots from 1-year run initialized from periodic NK solution
- **κV**: Prescribed vertical diffusivity from parent model
