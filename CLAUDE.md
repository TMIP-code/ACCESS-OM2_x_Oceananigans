Read the AGENTS.md file for project context.

## Bash: disable colored output

This shell's startup config forces ANSI color codes into command output (e.g.
`alias ls='command ls --color'` always colorizes, even when piped), which
clutters tool results with escape sequences like `[0m`, `[32m`, `[36m`. Always
suppress color when running Bash commands:

- `ls`/`l`/`la` → add `--color=never` (e.g. `ls --color=never`, `ls -la --color=never`).
- `grep`/`egrep`/`fgrep` → add `--color=never` (their aliases default to `--color=auto`).
- General fallback: prefix the command with a backslash to bypass the alias
  entirely (e.g. `\ls`, `\grep`), or pass `--color=never` to any tool that
  supports it.

## driver.sh: chain new jobs onto an earlier invocation's preprocessing

`scripts/driver.sh` has no raw `--afterok <jobid>` flag, but you can chain a
later invocation's steps onto jobs an earlier invocation already submitted by
**pre-setting the job-ID env vars** it reads (see driver.sh:253-257):
`PREP_JOB`, `GRID_JOB`, `VEL_JOB`, `CLO_JOB`, `DIAGW_JOB`, `PARTITION_JOB`.

When set, downstream steps compute their `afterok` deps from these instead of
resubmitting the preprocessing. Example — reuse an already-submitted grid/vel
rebuild for a second run without rebuilding (and without racing on the shared
per-experiment `grid.jld2`):

```bash
DIAGW_JOB=169655237.gadi-pbs CLO_JOB=169655236.gadi-pbs \
  GRID_HZ=4 PARENT_MODEL=ACCESS-OM2-1 ADVECTION_SCHEME=weno5 TIMESTEP_MULT=4 \
  JOB_CHAIN=run1yr-run1yrfast bash scripts/driver.sh
```

Standard runs use `VEL_DEP`, which the driver derives as `DIAGW_JOB:CLO_JOB`
(falling back to `VEL_JOB`/`GRID_JOB`, or `PARTITION_JOB` for multi-rank). So
pre-set whichever upstream jobs the step actually depends on.