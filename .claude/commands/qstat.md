Check the status of PBS Pro jobs on Gadi. Run the appropriate qstat command and present the results clearly.

## Behaviour

- If an argument is provided (`$ARGUMENTS`), treat it as a job ID and run `qstat -f $ARGUMENTS` for detailed info on that specific job.
- If no argument is provided, run `qstat -u bp3051` to list all current jobs.

## After showing status

- If any jobs show status `F` (finished) or `E` (exiting), offer to check the log file.
- For failed jobs, check the exit status and offer to tail the relevant log from `logs/`.
- For queued jobs (`Q`), mention estimated start time if visible in `qstat -f`.
- Keep the output concise — a summary table is better than raw qstat dump.
