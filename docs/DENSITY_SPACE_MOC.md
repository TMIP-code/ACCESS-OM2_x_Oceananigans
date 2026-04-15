# Density-space MOC exploration

Full-timeseries global density-space MOC for all three model resolutions,
produced from `ty_trans_rho + ty_trans_rho_gm` (GM not available for OM2-01 —
eddy-resolving). See [plan](../.claude/plans/density-space-moc-windows.md) and
[README § Model notes](../README.md#model-notes).

Scripts:
- [src/compute_MOC_rho_timeseries.py](../src/compute_MOC_rho_timeseries.py) — NetCDF export
- [src/animate_MOC_rho_timeseries.jl](../src/animate_MOC_rho_timeseries.jl) — mp4 animations
- [src/plot_AABW_timeseries.jl](../src/plot_AABW_timeseries.jl) — AABW timeseries (`SPACE=rho`)

Metric for AABW picks below: **min ψ for lat < 0° and σ₀ ≥ 1036 kg/m³**.

---

## AABW timeseries

### ACCESS-OM2-1 (1°)
![OM2-1 AABW rho](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/AABW/AABW_rho_lt0_1036_rhospace_timeseries.png)

### ACCESS-OM2-025 (0.25°)
![OM2-025 AABW rho](../outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/AABW/AABW_rho_lt0_1036_rhospace_timeseries.png)

### ACCESS-OM2-01 (0.1°)
![OM2-01 AABW rho](../outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle3/AABW/AABW_rho_lt0_1036_rhospace_timeseries.png)

### 1-year picks per model

| Model | Max AABW 1yr | Min AABW 1yr |
|-------|---------------|---------------|
| OM2-1 | 1980 | 2008 |
| OM2-025 | 1975 | 1986 |
| OM2-01 | 1967 | 2015 |

10-yr picks agree between OM2-025 and OM2-01 (`1967–1976`); OM2-1's 10-yr
window is `1962–1971` — common overlap is `1967–1971`.

---

## Global density-space MOC animations

Each model has two mp4s:
- `…_timeseries.mp4` — raw monthly frames (~720 frames, ~30 s at 24 fps)
- `…_rollingyear.mp4` — 12-month day-weighted rolling mean (same length, smoother)

### ACCESS-OM2-1 (1°)

- [MOC_rho_global_timeseries.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/MOC_rho_global_timeseries.mp4)
- [MOC_rho_global_rollingyear.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/MOC_rho_global_rollingyear.mp4)

### ACCESS-OM2-025 (0.25°)

- [MOC_rho_global_timeseries.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/MOC_rho_global_timeseries.mp4)
- [MOC_rho_global_rollingyear.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-025/025deg_jra55_iaf_omip2_cycle6/MOC_rho_global_rollingyear.mp4)

### ACCESS-OM2-01 (0.1°)

- [MOC_rho_global_timeseries.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle3/MOC_rho_global_timeseries.mp4)
- [MOC_rho_global_rollingyear.mp4](/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans/outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle3/MOC_rho_global_rollingyear.mp4)
