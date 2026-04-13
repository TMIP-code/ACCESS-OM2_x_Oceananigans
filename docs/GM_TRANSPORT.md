# GM Transport Implementation

There are four ways to include GM (Gent-McWilliams) effects in the tracer transport:

1. **No GM** -- resolved transport only.
2. **GM from parent model** -- GM streamfunction diagnostics (`tx_trans_gm`, `ty_trans_gm`) from MOM5 are converted to per-layer transport and added to the resolved transport during the `vel` preprocessing step.
3. **Diffusive Redi-GM** -- online Redi-GM parameterization using the diffusive formulation in Oceananigans (`GMREDI=true`).
4. **Advective Redi-GM** -- online Redi-GM using the advective formulation in Oceananigans.

## Option 2: GM from parent model

This section documents the preprocessing pipeline that combines resolved and GM transport from MOM5 output.

### MOM5 GM diagnostics

MOM5 outputs two GM-related diagnostics:

- `tx_trans_gm(xu_ocean, yt_ocean, st_ocean, time)` -- GM mass transport streamfunction in the x-direction (kg/s)
- `ty_trans_gm(xt_ocean, yu_ocean, st_ocean, time)` -- GM mass transport streamfunction in the y-direction (kg/s)

These share the same horizontal grid as the resolved mass transports (`tx_trans`, `ty_trans`) and have `Nz` vertical levels on the `st_ocean` axis. They represent the **overturning streamfunction** at w-levels (cell interfaces), not per-layer transport. The surface interface value (psi = 0) is implicit and not stored, so only `Nz` of the `Nz+1` interface values are present.

### Vertical convention

In MOM5 (k=1 surface, k=Nz bottom), `tx_trans_gm(k)` stores the GM streamfunction at interface k, defined as the cumulative GM transport from the surface down to that level:

```
psi(0) = 0           (surface, implicit -- not stored)
psi(k) = sum(T(j), j=1..k)   for k = 1..Nz
psi(Nz) ~ 0          (column-integrated GM transport is near zero by construction)
```

The per-layer GM transport for MOM5 cell k is:

```
T(k) = psi(k) - psi(k-1)
```

### Conversion to Oceananigans

The preprocessing in `src/prep_velocities.jl` performs the following steps:

1. **Load and place on C-grid** via `fill_Cgrid_transport_from_MOM_output!` (`src/shared_utils/data_loading.jl`):
   - Same horizontal shifts as resolved transport (di=1 for tx, dj=1 for ty)
   - Vertical flip: MOM k=1 (surface) maps to Oceananigans k=Nz (surface)

2. **Convert streamfunction to per-layer transport** via `streamfunction_to_perlayer!`:

   After the vertical flip, `psi_oce[1] = psi_MOM(Nz) ~ 0` (bottom) and `psi_oce[Nz] = psi_MOM(1)` (below surface). The kernel computes:

   ```
   T[k] = psi[k] - psi[k+1]    for k = 1..Nz-1
   T[Nz] = psi[Nz] - 0         (surface cell, psi[Nz+1] = 0 implied)
   ```

   This is correct because the vertical flip transforms the MOM5 per-layer formula `T_MOM(k) = psi(k) - psi(k-1)` into `T_oce[k_oce] = psi_oce[k_oce] - psi_oce[k_oce+1]`.

3. **Mask and fill halos** on the per-layer GM transport.

4. **Combine with resolved transport**:

   ```julia
   u_tt = (tx + tx_gm) / (rho0 * AxFCC)
   v_tt = (ty + ty_gm) / (rho0 * AyCFC)
   ```

5. **Derive vertical velocity from continuity** of the total (resolved + GM) horizontal transport:

   ```julia
   fill_continuity_tz_from_tx_ty!(tz_tt, grid, tx_total, ty_total)
   w_tt = tz_tt / (rho0 * AzCCF)
   ```

   This ensures mass conservation for the combined flow.

### Verification

The test `test/verify_gm_streamfunction.jl` validates the conversion by checking:

1. **Reconstruction**: reverse cumsum of per-layer transport recovers the original streamfunction (mathematical identity, verifies the diff kernel is self-consistent).
2. **MOC comparison**: GM MOC computed directly from the raw streamfunction matches the MOC derived from per-layer transport via zonal sum + cumsum.

Both tests are self-consistency checks (roundtrip identities). The physical sign correctness relies on the MOM5 convention described above, which is supported by the observation that `psi_oce[1] ~ 0` (GM streamfunction vanishes at the ocean floor).

### Key files

| File | Role |
|------|------|
| `src/periodicaverage.py` | Python preprocessing: extracts `tx_trans_gm` / `ty_trans_gm` from MOM5 output catalog |
| `src/prep_velocities.jl` (lines ~396-431) | Julia preprocessing: loads GM data, converts to per-layer, combines with resolved |
| `src/shared_utils/data_loading.jl` | `fill_Cgrid_transport_from_MOM_output!`, `streamfunction_to_perlayer!` |
| `test/verify_gm_streamfunction.jl` | Verification of streamfunction-to-perlayer conversion |
