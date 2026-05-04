Check the status of PBS Pro jobs on Gadi. Run the appropriate qstat command and present the results clearly.

## Behaviour

- If an argument is provided (`$ARGUMENTS`), treat it as a job ID and run `qstat -f $ARGUMENTS` for detailed info on that specific job.
- If no argument is provided, run `qstat -u bp3051 -wx` to list all current jobs (the `-wx` flag gives wide output with full job names).

## After showing status

- If any jobs show status `F` (finished) or `E` (exiting), offer to check the log file.
- For failed jobs, check the exit status and offer to tail the relevant log from `logs/`.
- For queued jobs (`Q`), mention estimated start time if visible in `qstat -f`.
- Keep the output concise — a summary table is better than raw qstat dump.

## Submissions index reconcile

If any jobs in the listing have transitioned to `F` since the user last reconciled, offer to run:

```bash
bash scripts/runs/reconcile_submissions.sh
```

This fills the PBS-side columns (`exit_code`, `queue`, walltime/mem
req+used, ncpus, ngpus) of `scripts/runs/submissions.tsv` for any rows
whose jobs have finished. PBS history retention on Gadi is ~7 days, so
prefer running reconcile while jobs are still visible in `qstat -x`.

Don't run reconcile silently — tell the user it'll run and what it does.
