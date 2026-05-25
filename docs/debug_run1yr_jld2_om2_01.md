# Debug: run1yr JLD2 `InvalidDataException` at OM2-01

## TL;DR

`run_1year.jl` at OM2-01 / 1×4 / `wparent` / `mkappaV` / `DTx2` dies during
output-writer snapshots with
`JLD2.InvalidDataException: Invalid Object header signature` (inside
`Oceananigans.OutputWriters.jld2output!` →
`JLD2.load_datatypes` → `JLD2.HeaderMessageIterator`).

The same simulation succeeds at OM2-1 / 1×2 with the same writer setup,
so the bug is **OM2-01-scale-specific** (very large per-rank JLD2 files
with `MmapIO` + `with_halos=true`), not a generic JLD2 / writer issue.

## Failing jobs

| Job | LOAD_BALANCE | Wall | Result | Where in sim |
|---|---|---|---|---|
| `169132253` | `no` (1x4) | 48 min | exit 1 | sim iter 29592 / 0.75 yr |
| `169132265` | `cell` (1x4_LB) | 21 min | exit 1 | sim iter 0 (first snapshot) |

Both with `LUMP_AND_SPRAY=no`, `MATRIX_PROCESSING=symdrop`, `TM_SOURCE=const`,
`INITIAL_AGE=0`. The rebuilt partitions had the correct
`Hx=Hy=7, Hz=2` halos (verified before submission), so the partition halo
mismatch is **not** the cause.

The successful counterpart at OM2-01 — `169128255` (`LOAD_BALANCE=surface`,
`1x4_LBS`, 52 min, exit 0) — wrote with the same code and same JLD2 0.6.4.
We don't yet have a clean reproducer that consistently fails (LBS worked at
the same scale).

### Caveat: the LB tag is missing from the OUTPUT path

`build_model_config()` in [src/shared_utils/config.jl](src/shared_utils/config.jl#L114-L130)
does NOT include the LB suffix (`_LB`, `_LBS`, `_LBmix`, `_LBminmax`). Only
the shell-side `MODEL_CONFIG` in `scripts/env_defaults.sh` does, and that
only feeds log filenames. So the Julia output writer for all LB variants
writes to the **same** directory:

```
outputs/{PM}/{EXP}/{TW}/standardrun/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/1x4/
```

Combined with `overwrite_existing = true`, this means:
- 169128255 (LBS, May 24 19:34) wrote its output to that dir.
- 169132253 (no, May 25 02:28) re-opened the *same files*,
  `overwrite_existing=true` truncated them, then it crashed mid-snapshot.
- 169132265 (cell, May 25 02:52) ran into the same dir again.
- Net effect: the LBS data is gone (silently clobbered), and the files I
  inspected after the fact are LB=no's truncated remnants
  (e.g. `age_1year_rank0.jld2` at 715 bytes), not LBS output.

This is a related bug worth fixing independently of the JLD2 issue — the
LB tag should be in the output path so concurrent (or sequential) LB
sweep submissions don't clobber each other. Possible fixes:

- Inline the LB suffix into the Julia `build_model_config()` (mirror the
  shell-side logic via `parse_load_balance_env()` already in
  `src/shared_utils/load_balance.jl`).
- Or push the LB into a sub-dir under `model_config`, similar to how
  `{px}x{py}` already sits between `model_config` and the rank files.

## Submission (failing job, e.g. 169132253)

```bash
PARENT_MODEL=ACCESS-OM2-01 \
W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=parent \
TIMESTEP_MULT=2 MONTHLY_KAPPAV=yes \
PARTITION=1x4 LOAD_BALANCE=no \
JOB_CHAIN=partition-run1yr \
bash scripts/driver.sh
```

(`partition` rebuilt the stale `1x4` halo before `run1yr` ran — both
steps' job IDs above.)

## Log locations

PBS scheduler stdout/stderr:
- `logs/PBS/169132253.gadi-pbs.OU`
- `logs/PBS/169132253.gadi-pbs.ER`
- `logs/PBS/169132265.gadi-pbs.OU`
- `logs/PBS/169132265.gadi-pbs.ER`

Julia logs (the real stacktraces are here):
- `logs/julia/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/standardrun/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2_1year_169132253.gadi-pbs.log`
- `logs/julia/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/standardrun/cgridtransports_wparent_centered2_AB2_mkappaV_LB_DTx2_1year_169132265.gadi-pbs.log`

Partial output files (showing the corruption shape):
- `outputs/ACCESS-OM2-01/01deg_jra55v140_iaf_cycle4/1968-1977/standardrun/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/1x4/`
  - `u_1year_rank0.jld2` (7.2 GB), `v_1year_rank0.jld2` (7.2 GB), `w_1year_rank0.jld2` (1.2 GB), `eta_1year_rank0.jld2` (618 MB) — written before the crash
  - `age_1year_rank0.jld2` (**715 bytes** — JLD2 header only, no snapshots) — suspect file

## What we know

- Stack trace ends in `Oceananigans.OutputWriters.jld2_writer.jl:385`
  (`jld2output!`) → `JLD2.prewrite` →
  `JLD2.load_datatypes(f::JLD2.JLDFile{JLD2.MmapIO})` →
  `JLD2.read_shared_datatype` →
  `JLD2.HeaderMessageIterator(...)` ← throws `InvalidDataException`.
- CUDA `take! at synchronization.jl:53` traces in the same log are the
  rank-death cascade, **not** the cause.
- Writer setup in [src/shared_utils/simulation.jl:106-115](src/shared_utils/simulation.jl#L106-L115):
  ```
  JLD2Writer(..., overwrite_existing = true, with_halos = true,
             including = [])
  ```
  → uses default `MmapIO` backend.
- JLD2 pinned at v0.6.4 (Manifest.toml).
- File sizes at OM2-01 are 1–7 GB per rank per field. At OM2-1 they're
  order of MB.

## Confirming reproducer (negative result)

OM2-1 / 1×2 / same flags ran cleanly in 1.77 min:

```bash
PARENT_MODEL=ACCESS-OM2-1 \
W_FORMULATION=wprescribed PRESCRIBED_W_SOURCE=parent \
TIMESTEP_MULT=2 MONTHLY_KAPPAV=yes \
PARTITION=1x2 LOAD_BALANCE=no \
JOB_CHAIN=run1yr \
bash scripts/driver.sh
```

Job `169159227`, exit 0, log at
`logs/julia/ACCESS-OM2-1/1deg_jra55_iaf_omip2_cycle6/1968-1977/standardrun/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2_1year_169159227.gadi-pbs.log`.

So the bug needs OM2-01 scale to trigger; a minimal repro likely needs
large file sizes (or wide grids) hitting MmapIO's mmap remap path.

## Debug ideas (in suggested order)

1. **Switch backend**: rerun the failing config with the JLD2Writer
   `:io` backend (non-mmap) — easiest way to confirm `MmapIO` is the
   culprit. Edit `src/shared_utils/simulation.jl:107-115`; add
   `backend = JLD2.IOStream` (or whatever the current JLD2 0.6.4
   non-mmap selector is).
2. **`with_halos=false`**: try without halos — cuts the per-snapshot
   byte count and may avoid whatever size threshold MmapIO trips on.
3. **`including = nothing`** vs `[]`: the comment in
   [simulation.jl:114](src/shared_utils/simulation.jl#L114) cites
   serializeproperty! deadlocks; worth checking if any related JLD2
   issue exists upstream.
4. **JLD2 bump**: check the JLD2 changelog for 0.6.x fixes touching
   `HeaderMessageIterator` / `load_datatypes` / `MmapIO`. If a newer
   patch fixes it, bump the pin.
5. **Bisect on LB**: LB=surface (LBS) worked at the same scale —
   diff the writer state between LBS and no/cell runs (rank layout,
   per-rank field sizes). The LBS partition gives un-balanced rank
   sizes; maybe LB=no's perfectly-uniform per-rank sizes hit a
   pathological mmap alignment.
6. **Minimal-mode repro**: write a stand-alone MWE that opens a
   `JLD2Writer` at OM2-01 scale and writes empty snapshots in a
   loop, no simulation — to isolate the JLD2 side from the sim side.

## Status

- Not blocking the NK_5x5 evaluation — NK uses `periodic_solver_common.jl`
  with no `JLD2Writer` attached (verified; see [docs/NK_OM2-01.md](NK_OM2-01.md)
  § Code path independence).
- Blocks `run1yr`-based LB sweeps and any diagnostic plotting that needs
  the per-snapshot output. Workaround for LB sweeps: use `run1yrfast`
  ([scripts/standard_runs/run_1year_benchmark.sh](../scripts/standard_runs/run_1year_benchmark.sh))
  which has no writers.
- Picked up only when convenient — low priority unless we need the
  `run1yr` outputs at OM2-01.
