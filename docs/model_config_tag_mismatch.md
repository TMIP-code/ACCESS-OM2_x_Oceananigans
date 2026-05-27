# Issue: Shell- and Julia-side `MODEL_CONFIG` tags disagree (LBS / TB suffixes)

**Status:** Open. Discovered 2026-05-27 while resubmitting OMEGA=z1500 chains
after the per-model-defaults refactor (`6e85648`, `5f85014`).

## Symptom

OM2-025 forward TMbuilds (`169335067` for 1968-1977, `169335095` for 1999-2008)
wrote `M.jld2` to

```
outputs/.../TM/totaltransport_wparent_centered2_AB2_mkappaV_DTx3/const/M.jld2
```

(**no** `_LBS`), even though `scripts/env_defaults.sh` reports

```
MODEL_CONFIG=totaltransport_wparent_centered2_AB2_mkappaV_LBS_DTx3
```

(**with** `_LBS`).

## Root cause

Two independent tag builders that don't agree:

- **Shell** ([scripts/env_defaults.sh:118-160](../scripts/env_defaults.sh#L118-L160))
  adds `LB_TAG` (`_LBS` for `LOAD_BALANCE=surface|yes`, `_LB`/`_LBmix`/`_LBminmax`
  for the other values) — and auto-suppresses it when `RANKS=1` (serial).
  Also adds `_TB<K>` for `TBLOCKING=<K>`.

- **Julia** (`build_model_config` in
  [src/shared_utils/config.jl:131](../src/shared_utils/config.jl#L131)) does
  **not** know about either `LOAD_BALANCE` or `TBLOCKING`, so it never emits
  `_LBS`/`_LB…`/`_TB<K>`.

The Julia-side tag is what actually controls disk paths (where TMbuild writes
`M.jld2`, where NK reads it, where ventilation outputs land); the shell
`MODEL_CONFIG` is only **echoed** to the PBS log — never `export`ed or consumed
downstream. So functionally everything works **as long as nothing tries to
compare or rendezvous on the two tags**.

## Why it was latent before

Until the recent default flip (`LOAD_BALANCE: no → surface`, commit `6e85648`),
`LOAD_BALANCE=no` produced an empty `LB_TAG`, so the shell and Julia tags
trivially agreed. With the new default they diverge for OM2-025 (1x2) and
OM2-01 (1x4); OM2-1 (1x1 serial) still agrees because LBS is auto-suppressed
for serial runs in both worlds.

## Reproducer

```bash
for pm in ACCESS-OM2-1 ACCESS-OM2-025 ACCESS-OM2-01; do
  shell_mc=$(PARENT_MODEL=$pm bash -c 'source scripts/env_defaults.sh > /dev/null 2>&1; echo $MODEL_CONFIG')
  julia_mc=$(PARENT_MODEL=$pm bash -c '
    source scripts/env_defaults.sh > /dev/null 2>&1
    julia --project=. -e "
      include(\"src/shared_functions.jl\")
      (; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER) = parse_config_env()
      println(build_model_config(; VELOCITY_SOURCE, W_FORMULATION, ADVECTION_SCHEME, TIMESTEPPER))
    " 2>&1 | tail -1')
  printf "%-16s shell=%-58s julia=%s\n" "$pm" "$shell_mc" "$julia_mc"
done
```

Output as of 2026-05-27:

```
ACCESS-OM2-1     shell=totaltransport_wparent_centered2_AB2_mkappaV_DTx4    julia=totaltransport_wparent_centered2_AB2_mkappaV_DTx4
ACCESS-OM2-025   shell=totaltransport_wparent_centered2_AB2_mkappaV_LBS_DTx3   julia=totaltransport_wparent_centered2_AB2_mkappaV_DTx3
ACCESS-OM2-01    shell=cgridtransports_wparent_centered2_AB2_mkappaV_LBS_DTx2  julia=cgridtransports_wparent_centered2_AB2_mkappaV_DTx2
```

## Fix options

Pick one — they each have different blast-radius implications for existing
on-disk artefacts:

1. **Patch Julia (sync Julia → shell).** Extend `build_model_config` to read
   `PARTITION_X`/`Y` (compute RANKS) + `LOAD_BALANCE` + `TBLOCKING` and emit
   `_LBS`/`_LB…`/`_TB<K>` matching the shell. Implementation cost: ~25 lines.
   Real cost: any pre-existing OM2-025/OM2-01 output directory under the
   *no-`_LBS`* tag becomes orphaned — either migrate (`mv` or symlink the
   `no-_LBS` tree onto the `_LBS` tree) or accept that downstream tooling now
   looks elsewhere. I wrote and verified a patch during the original
   investigation — it produces matching tags across all three PMs; reverted
   in the working tree pending this decision.

2. **Patch shell (sync shell → Julia).** Drop `LB_TAG` and `_TB<K>` from the
   shell's `MODEL_CONFIG` so the echoed tag matches Julia's actual disk
   paths. Cost: lose human-readable LBS/TB tagging in PBS logs and submission
   manifests; LBS state has to be read from `LOAD_BALANCE` in the env dump
   instead.

3. **Make shell authoritative.** `export MODEL_CONFIG` from `env_defaults.sh`
   and have Julia *consume* that env var instead of recomputing. Removes the
   duplication entirely. Cost: tighter shell↔Julia coupling; need to remove
   or downgrade Julia's `build_model_config` to a sanity-check.

My recommendation: **option 1**, paired with a one-time migration script that
walks `outputs/ACCESS-OM2-{025,01}/.../TM/<...>_mkappaV_DTx<M>/` and renames
each into `<...>_mkappaV_LBS_DTx<M>/` (or symlinks). The patch I drafted is
the smaller code change; the migration is mechanical and reversible.

## In-flight runs at the time this issue was filed

- **OM2-1 fwd × 2 TWs:** chain completed (TMbuild, NK, run1yrNK, ventilation)
  under `totaltransport_wparent_centered2_AB2_mkappaV_DTx4/` — Julia and
  shell tags agree (serial 1x1), no action required.
- **OM2-025 fwd × 2 TWs:** TMbuild ✓ at the *no-`_LBS`* path; NK queued,
  run1yrNK + ventilation held. With Julia not emitting `_LBS`, the chain
  is internally consistent and should complete cleanly.
- **OM2-1 adj × 2 TWs and OM2-025 adj × 2 TWs:** all 4 TRAF chains died
  because their adjoint TMbuilds raced against (and ran before) the
  forward TMbuilds. **Resubmitted** today now that the forward `M.jld2`
  exists — these chains will also land at the *no-`_LBS`* path under the
  current Julia behaviour.

If/when option 1 is applied, plan to do the migration **before** any new
TMbuild lands, or you'll end up with `M.jld2` in both `…_DTx<M>` and
`…_LBS_DTx<M>` and have to reconcile by inspection.

## Related commits

- `6e85648` — `config: make PARENT_MODEL required; move VELOCITY_SOURCE per-model`
  (the refactor that exposed this).
- `5f85014` — `README: regen defaults tables` (the regenerated tables show
  `LOAD_BALANCE | surface` cross-model default).
