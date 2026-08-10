# Candidate simulations for exploring changes in deep water formation (DWF)

Each section covers one physical mechanism, ordered by availability at lower resolution. Within each table, rows are ordered by resolution (coarsest first).


## 1. Resolution effect

Same forcing at different ocean resolutions. Tests whether eddies fundamentally change DWF representation.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 1° | 1deg_jra55_ryf9091_gadi | — | OM2 | [Kiss et al. (2020)](https://doi.org/10.5194/gmd-13-401-2020) | RYF baseline |
| 0.25° | 025deg_jra55_ryf9091_gadi | 1deg_jra55_ryf9091_gadi | OM2 | [Kiss et al. (2020)](https://doi.org/10.5194/gmd-13-401-2020) | Same RYF; eddy-permitting |
| 0.1° | 01deg_jra55v13_ryf9091 | 025deg_jra55_ryf9091_gadi | OM2 | [Kiss et al. (2020)](https://doi.org/10.5194/gmd-13-401-2020) | Same RYF; eddying |
| 1° | 1deg_jra55_iaf_omip2_cycle6 | — | OM2 | | IAF baseline |
| 0.25° | 025deg_jra55_iaf_omip2_cycle6 | 1deg_jra55_iaf_omip2_cycle6 | OM2 | | Same IAF; eddy-permitting |
| 0.1° | 01deg_jra55v140_iaf_cycle4 | 025deg_jra55_iaf_omip2_cycle6 | OM2 | | IAF at eddying resolution |
| 1° / 0.25° | cj877 (0.25° ocean) | bz687 (1° ocean) | CM2 (native) | | Coupled; same atm. forcing |


## 2. Climate change (GHG-driven warming)

Surface warming and freshening reduce dense water formation. Available across all resolutions via CMIP6 scenarios and OMIP2 cycling.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 1° | ssp585 | piControl | ESM1-5 (40 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Strongest warming; largest projected AABW decline |
| 1° | ssp370 | piControl | ESM1-5 (40 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | High-emission scenario |
| 1° | ssp245 | piControl | ESM1-5 (40 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Intermediate scenario |
| 1° | ssp126 | piControl | ESM1-5 (40 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Low-emission; modest DWF change |
| 1° | ssp585 | piControl | CM2 (10 members) | [Mackallah et al. (2022)](https://doi.org/10.1071/ES21031) | Different model physics; same forcing |
| 1° | 1pctCO2 | piControl | ESM1-5 | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Idealized transient; clean DWF signal |
| 1° | abrupt-4xCO2 | piControl | ESM1-5 (2 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Equilibrium DWF response |
| 1° | ssp534-over | piControl | ESM1-5 (41 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Overshoot then decline; tests DWF recovery |
| 1° | PI_GWL_B2035..B2060 | esm-piControl | ESM1-5 (native) | [King et al. (2024)](https://doi.org/10.5194/esd-15-1353-2024) | Zero CO2 at 6 warming levels; DWF recovery gradient |


## 3. Physical perturbations

Targeted experiments isolating individual drivers of DWF change.

### 3.1 Tropical teleconnections

Tests how tropical climate modes affect Southern Ocean DWF via atmospheric bridges.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 1° | by473 (pacemaker hist) | bx944 (hist control) | CM2 (native) | [Mackallah et al. (2022)](https://doi.org/10.1071/ES21031) | Tropical Atlantic fixed SSTs from obs |
| 1° | by578 (pacemaker ssp245) | by647 (ssp245 control) | CM2 (native) | [Mackallah et al. (2022)](https://doi.org/10.1071/ES21031) | Future Atlantic teleconnection |
| 0.1° | 01deg_jra55_ryf_ENFull | 01deg_jra55_ryf_Control | OM2 | [Huguenin et al. (2024)](https://doi.org/10.1029/2023GL104518) | El Nino forcing |
| 0.1° | 01deg_jra55_ryf_LNFull | 01deg_jra55_ryf_Control | OM2 | [Huguenin et al. (2024)](https://doi.org/10.1029/2023GL104518) | La Nina forcing |

### 3.2 Wind stress

Antarctic wind perturbations affect DWF preconditioning via Ekman upwelling, isopycnal steepening, and cross-shelf transport.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 0.1° | easterlies_up10 | 01deg_jra55v13_ryf9091 | OM2 | [Morrison et al. (2023)](https://doi.org/10.1175/JCLI-D-22-0858.1) | +10% zonal + meridional wind |
| 0.1° | easterlies_down10 | 01deg_jra55v13_ryf9091 | OM2 | [Morrison et al. (2023)](https://doi.org/10.1175/JCLI-D-22-0858.1) | -10% zonal + meridional wind |
| 0.1° | easterlies_up10_zonal | 01deg_jra55v13_ryf9091 | OM2 | [Morrison et al. (2023)](https://doi.org/10.1175/JCLI-D-22-0858.1) | Zonal only; isolates ACC/upwelling effect |
| 0.1° | easterlies_up10_meridional | 01deg_jra55v13_ryf9091 | OM2 | [Morrison et al. (2023)](https://doi.org/10.1175/JCLI-D-22-0858.1) | Meridional only; isolates cross-shelf transport |

### 3.3 Freshwater (meltwater)

Direct perturbation of Weddell Sea meltwater forcing. Freshwater caps the surface, inhibiting convection and AABW formation.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 0.1° | weddell_up1 | 01deg_jra55v13_ryf9091 | OM2 | [Moorman et al. (2020)](https://doi.org/10.1175/JCLI-D-19-0846.1) | Increased meltwater; inhibits AABW |
| 0.1° | weddell_down2 | 01deg_jra55v13_ryf9091 | OM2 | [Moorman et al. (2020)](https://doi.org/10.1175/JCLI-D-19-0846.1) | Decreased meltwater; may enhance AABW |

### 3.4 Combined future (wind + thermal + meltwater)

Combines multiple forcing changes to project future AABW decline. Comparing the two experiments isolates the meltwater contribution.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 0.1° | qian_wthmp | 01deg_jra55v13_ryf9091 | OM2 | [Li et al. (2023)](https://doi.org/10.1038/s41586-023-05762-w) | Wind + thermal + meltwater; full projected AABW decline |
| 0.1° | qian_wthp | 01deg_jra55v13_ryf9091 | OM2 | [Li et al. (2023)](https://doi.org/10.1038/s41586-023-05762-w) | Wind + thermal only; isolates non-meltwater drivers |


## Maybe next time

### 4. Single-forcing attribution (DAMIP)

Decomposes the historical DWF signal into individual forcing agents. Ozone is particularly relevant: stratospheric ozone depletion strengthens Southern Hemisphere westerlies, affecting upwelling and DWF preconditioning.

| resolution | perturbation | reference | model | ref. | notes |
|------------|-------------|-----------|-------|------|-------|
| 1° | hist-GHG | piControl | ESM1-5 (7 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Greenhouse gas effect only |
| 1° | hist-aer | piControl | ESM1-5 (7 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Aerosol cooling/masking effect |
| 1° | hist-totalO3 | piControl | ESM1-5 (10 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Ozone depletion; drives Southern Ocean wind changes |
| 1° | hist-volc | piControl | ESM1-5 (10 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Episodic volcanic cooling |
| 1° | hist-nat | piControl | ESM1-5 (7 members) | [Ziehn et al. (2020)](https://doi.org/10.1071/ES19035) | Solar + volcanic; natural-only DWF variability |
