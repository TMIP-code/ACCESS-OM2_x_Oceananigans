# Issue: Shell- and Julia-side `MODEL_CONFIG` tags disagree (LBS / TB suffixes)

**Status:** Resolved (2026-05-27). Fix: option 3 — make shell authoritative,
Julia reads `ENV["MODEL_CONFIG"]`, `build_model_config` deleted.

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

Two independent tag builders that didn't agree:

- **Shell** ([scripts/env_defaults.sh:118-160](../scripts/env_defaults.sh#L118-L160))
  adds `LB_TAG` (`_LBS` for `LOAD_BALANCE=surface|yes`, `_LB`/`_LBmix`/`_LBminmax`
  for the other values) — and auto-suppresses it when `RANKS=1` (serial).
  Also adds `_TB<K>` for `TBLOCKING=<K>`.

- **Julia** (the deleted `build_model_config` in `src/shared_utils/config.jl`)
  did **not** know about either `LOAD_BALANCE` or `TBLOCKING`, so it never
  emitted `_LBS`/`_LB…`/`_TB<K>`.

## Fix applied

**Option 3: make shell authoritative.**

1. `scripts/env_defaults.sh` now `export`s `MODEL_CONFIG` (was only echoed).
2. Julia reads `require_env("MODEL_CONFIG")` — the `build_model_config`
   function and its duplicate tag-assembly logic were deleted entirely.
3. Existing on-disk artefacts (43 directories under OM2-025 and OM2-01) were
   renamed to their `_LBS` siblings via `scripts/maintenance/migrate_lbs_tag.sh`,
   with back-compat symlinks for pre-refactor queued jobs.

## On-disk migration

Run `scripts/maintenance/migrate_lbs_tag.sh --cleanup-symlinks` once all
pre-refactor PBS jobs have drained (`qstat` shows no Q/H/R from the old wave).

## Verification

Shell and Julia now return the same string (trivially — Julia reads the
shell's exported value):

```bash
for pm in ACCESS-OM2-1 ACCESS-OM2-025 ACCESS-OM2-01; do
  PARENT_MODEL=$pm bash -c '
    source scripts/env_defaults.sh > /dev/null 2>&1
    julia_mc=$(julia --project=. -e "
      include(\"src/shared_functions.jl\")
      println(require_env(\"MODEL_CONFIG\"))
    " 2>&1 | tail -1)
    printf "%-16s shell=%-58s julia=%-58s match=%s\n" \
      "$PARENT_MODEL" "$MODEL_CONFIG" "$julia_mc" \
      "$([ "$MODEL_CONFIG" = "$julia_mc" ] && echo YES || echo NO)"
  '
done
```

## Related commits

- `6e85648` — `config: make PARENT_MODEL required; move VELOCITY_SOURCE per-model`
  (the refactor that exposed this).
- `5f85014` — `README: regen defaults tables` (the regenerated tables show
  `LOAD_BALANCE | surface` cross-model default).
