# MLD × Transport time-window comparison

A test set running OM2-1 NK periodic-steady-state age across the four
combinations of transport-window × MLD-window from `{1968-1977, 1972}`.
The aim is to isolate how much of the periodic age signal comes from the
MLD (vertical-mixing structure) versus the transport circulation.

## Setup

- `PARENT_MODEL = ACCESS-OM2-1`
- `EXPERIMENT  = 1deg_jra55_iaf_omip2_cycle6`
- `MODEL_CONFIG = cgridtransports_wdiagnosed_centered2_AB2_mkappaV` (MONTHLY_KAPPAV=yes)
- `TM_SOURCE = const`, `LINEAR_SOLVER = Pardiso`, `LUMP_AND_SPRAY = yes`
- Decoupled via `MLD_TIME_WINDOW` env var (see [README.md](../README.md#tracking-submissions))

Case files: [scripts/runs/cases/OM2-1_TR*_MLD*.sh](../scripts/runs/cases/).

| Case                | TR_TW     | MLD_TW    | Output dir                                                                 |
|---------------------|-----------|-----------|----------------------------------------------------------------------------|
| TR1968-1977_MLD1968-1977 | 1968-1977 | 1968-1977 | [TR1968-1977_MLD1968-1977/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977) |
| TR1968-1977_MLD1972      | 1968-1977 | 1972      | [TR1968-1977_MLD1972/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972) |
| TR1972_MLD1968-1977      | 1972      | 1968-1977 | [TR1972_MLD1968-1977/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977) |
| TR1972_MLD1972           | 1972      | 1972      | [TR1972_MLD1972/](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972) |

How to read each 2×2 table below:

- **Rows** — transport time window (`TR_TW`).
- **Columns** — MLD time window (`MLD_TW`).
- **Diagonal** (TR=MLD) — internally consistent runs.
- **Off-diagonal** (TR≠MLD) — mixed runs that isolate MLD vs transport effects.

All static plots are time-mean fields from the 1-year run-from-periodic
solution. Animations of the seasonal cycle are linked under each section
(GitHub doesn't render `.mp4` inline; click to download).

[plot_dir]: ./../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test
[mc]: cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots

## Horizontal slices (time-mean age)

### 100 m

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_100m.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_100m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_100m.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_100m.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_100m.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_100m.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_100m.mp4)

### 200 m

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_200m.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_200m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_200m.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_200m.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_200m.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_200m.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_200m.mp4)

### 500 m

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_500m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_500m.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_500m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_500m.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_500m.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_500m.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_500m.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_500m.mp4)

### 1000 m

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_1000m.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_1000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_1000m.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_1000m.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_1000m.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_1000m.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_1000m.mp4)

### 2000 m

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_2000m.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_2000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_2000m.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_2000m.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_2000m.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_2000m.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_2000m.mp4)

### 3000 m

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_3000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_3000m.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_3000m.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_slice_3000m.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_3000m.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_3000m.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_3000m.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_slice_3000m.mp4)

## Zonal averages (time-mean age)

### Global

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_global.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_global.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_global.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_global.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_global.mp4)

### Atlantic

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_atlantic.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_atlantic.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_atlantic.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_atlantic.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_atlantic.mp4)

### Pacific

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_pacific.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_pacific.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_pacific.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_pacific.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_pacific.mp4)

### Indian

|              | MLD = 1968-1977 | MLD = 1972 |
|--------------|-----------------|------------|
| **TR = 1968-1977** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |
| **TR = 1972** | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) | ![](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_mean_centered2_zonal_avg_indian.png) |

Animations: [TR=68-77,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_indian.mp4) · [TR=68-77,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1968-1977_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_indian.mp4) · [TR=72,MLD=68-77](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1968-1977/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_indian.mp4) · [TR=72,MLD=72](../outputs/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/test/TR1972_MLD1972/periodic/cgridtransports_wdiagnosed_centered2_AB2_mkappaV/1year/Pardiso_LSprec/plots/age_periodic_centered2_zonal_avg_indian.mp4)

## Reproducing

```bash
# 1. preprocess 1972 once (1968-1977 already preprocessed)
TIME_WINDOW=1972 JOB_CHAIN=preprocessing bash scripts/driver.sh

# 2. submit the 4 cases
for c in scripts/runs/cases/OM2-1_TR{1968-1977,1972}_MLD{1968-1977,1972}.sh; do
    bash scripts/runs/run_case.sh "$c"
done

# 3. once finished, refresh the submissions index
bash scripts/runs/reconcile_submissions.sh
```

PBS resources per chain: TMbuild ~7 min CPU + NK ~1h10 GPU + run1yrNK ~8 min GPU + plotNK ~7 min CPU.
