# Debug: run1yr JLD2 `InvalidDataException` — concurrent writes to a shared dir

## TL;DR — root cause identified

Two `run_1year.jl` submissions at OM2-01 / 1×4 / `wparent` / `mkappaV` /
`DTx2` with different `LOAD_BALANCE` values **wrote to the same output
files at the same time**, because Julia's `build_model_config()` doesn't
include the LB tag in the path. Both used `overwrite_existing=true`, so
the later submission truncated the first one's files mid-write. JLD2
then (correctly) raised `InvalidDataException` because the file content
it was about to extend had been zeroed out from under it.

Failure trace:
`Oceananigans.OutputWriters.jld2output!` →
`JLD2.prewrite` → `JLD2.load_datatypes` → `JLD2.HeaderMessageIterator`
on a truncated file.

JLD2 is not buggy here. The bug is the path collision.

## Timing — the smoking gun

Both submissions wrote to
`outputs/.../standardrun/cgridtransports_wparent_centered2_AB2_mkappaV_DTx2/1x4/`
(no LB suffix — see "LB tag missing from output path" below):

```
169132253 (LB=no)   start 02:28:06, end 03:16:48  (48 min wall)
169132265 (LB=cell) start 02:52:16, end 03:13:27  (21 min wall)
                       ^ 24 min after LB=no started
```

Overlap = 02:52:16 → 03:13:27 = **21 minutes of concurrent writes** to
the same files. The chronology:

1. 02:28 — LB=no opens JLD2Writers with `overwrite_existing=true`,
   mmap-truncates the files, starts writing snapshots.
2. 02:28 → 02:52 — LB=no writes snapshots successfully for 24 min.
3. 02:52 — LB=cell opens the **same files** with
   `overwrite_existing=true`, truncating them while LB=no's MmapIO is
   still attached.
4. 02:52, **LB=cell iter 0** — tries its first `prewrite`, finds the
   file it just truncated and can't parse its own (non-existent)
   committed datatypes → InvalidDataException.
5. 03:00, **LB=no iter 29592** (0.75 yr, ~32 min wall) — tries its
   next scheduled snapshot. Its mmap view is now incoherent with the
   on-disk content (LB=cell zeroed the file). Same InvalidDataException.

The OM2-1 reproducer (`169159227`) ran as a single job and passed
cleanly in 1.77 min — no concurrent writer.

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

## Fix

Include the LB tag in Julia's `build_model_config()` so each LB variant
writes to its own dir and concurrent submissions can't clobber each
other. Mirror the shell-side logic in `scripts/env_defaults.sh` using
the existing `parse_load_balance_env()` in
[src/shared_utils/load_balance.jl:257](src/shared_utils/load_balance.jl#L257)
(returns the `_LB` / `_LBS` / `_LBmix` / `_LBminmax` tag). Touch:

- [src/shared_utils/config.jl:114-130](src/shared_utils/config.jl#L114-L130) — add the LB suffix to `mc`.

Once that's in, the OM2-01 LB sweep can be rerun without the
concurrent-write hazard. (Or use `run1yrfast` and avoid the writers
entirely — preferred for LB sweeps where we only need walltime.)

Open question (lower priority): why didn't `overwrite_existing=true`'s
truncate behaviour error out at LB=cell's open call when an mmap from
another process held the file? JLD2 + MmapIO seems to tolerate the
truncate silently, then explodes on the next read. Worth a tiny MWE if
we ever care about robustness against this case, but the path-collision
fix is what unblocks the immediate work.

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
