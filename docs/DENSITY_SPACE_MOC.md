# Density-space MOC exploration

Full-timeseries global density-space MOC for all three model resolutions,
produced from `ty_trans_rho + ty_trans_rho_gm` (GM not available for OM2-01 —
eddy-resolving). See [plan](../.claude/plans/density-space-moc-windows.md) and
[README § Model notes](../README.md#model-notes).

Scripts:
- [src/compute_MOC_rho_timeseries.py](../src/compute_MOC_rho_timeseries.py) — NetCDF export
- [src/animate_MOC_rho_timeseries.jl](../src/animate_MOC_rho_timeseries.jl) — mp4 animations + time-mean PNG
- [src/plot_AABW_timeseries.jl](../src/plot_AABW_timeseries.jl) — AABW timeseries (`SPACE=rho`)

AABW metric: **min ψ for lat < 0° and σ₀ ≥ 1036 kg/m³**.

Each model section below contains:
- Day-weighted time-mean MOC (PNG)
- AABW transport timeseries (PNG)
- Monthly animation (mp4, ~720 frames, ~30 s at 24 fps)
- 12-month rolling-mean animation (mp4, smoother)

### 1-year AABW picks

| Model | Max AABW 1yr | Min AABW 1yr |
|-------|---------------|---------------|
| OM2-1 | 1980 | 2008 |
| OM2-025 | 1975 | 1986 |
| OM2-01 | 1967 | 2015 |

10-yr picks agree between OM2-025 and OM2-01 (`1967–1976`); OM2-1's 10-yr
window is `1962–1971` — common overlap is `1967–1971`.

---

## ACCESS-OM2-1 (1°)

### Time-mean MOC

![OM2-1 time-mean MOC rho](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/MOC_rho_global_mean.png)

### AABW transport timeseries

![OM2-1 AABW rho](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/AABW/AABW_rho_lt0_1036_rhospace_timeseries.png)

### Animations

- [MOC_rho_global_timeseries.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/MOC_rho_global_timeseries.mp4)
- [MOC_rho_global_rollingyear.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/MOC_rho_global_rollingyear.mp4)

---

## ACCESS-OM2-025 (0.25°)

### Time-mean MOC

![OM2-025 time-mean MOC rho](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/MOC_rho_global_mean.png)

### AABW transport timeseries

![OM2-025 AABW rho](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/AABW/AABW_rho_lt0_1036_rhospace_timeseries.png)

### Animations

- [MOC_rho_global_timeseries.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/MOC_rho_global_timeseries.mp4)
- [MOC_rho_global_rollingyear.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/MOC_rho_global_rollingyear.mp4)

---

## ACCESS-OM2-01 (0.1°)

### Time-mean MOC

![OM2-01 time-mean MOC rho](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/MOC_rho_global_mean.png)

### AABW transport timeseries

![OM2-01 AABW rho](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/AABW/AABW_rho_lt0_1036_rhospace_timeseries.png)

### Animations

- [MOC_rho_global_timeseries.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/MOC_rho_global_timeseries.mp4)
- [MOC_rho_global_rollingyear.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/MOC_rho_global_rollingyear.mp4)
