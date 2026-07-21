# Proposal: `lazy-record` (record-then-exec)

> **This is a design proposal, not a slurmise feature.** Everything here runs
> against `./lazy_slurmise`, a small prototype that imports the installed
> slurmise package (config/job_spec parsing, sacct helpers, and the HDF5
> `JobDatabase` -- no slurmise source is modified) and adds only the lazy
> recording logic on top. It lives in the tutorial tree only so it can ride
> the existing tutorial CI; it will be either promoted into slurmise proper
> or removed.

Looking at slurmise tutorial, I don't like how we need to
specify the job parameters twice:
https://github.com/PrincetonUniversity/slurmise/blob/fix/docs/src/slurmise/tutorial/01_single_job/run_perfectScaler.sbatch

```bash
srun ../bin/perfectScaler --intensity 5000 --duration 10

slurmise --toml slurmise.toml record \
    "perfectScaler --intensity 5000 --duration 10"
```

Yes, we could use a variable to avoid the duplication like:
```bash
CMD="../bin/perfectScaler --intensity 5000 --duration 10"
srun $CMD
slurmise --toml slurmise.toml record $CMD
```

This prototype implements two alternatives, `record` and `lazy-record`, and
`run_perfectScaler.sbatch` runs both in the same job so they can be compared
directly (`run_perfectScaler_loop.sbatch` does the same 13 times for a fuller
side-by-side table -- run either, then `./lazy_slurmise display`).

## `record` (eager, mimics upstream)

Run *after* `srun cmd` completes, passing the command as a single quoted
string, exactly like upstream:

```bash
srun ../bin/perfectScaler --intensity 5000 --duration 10
./lazy_slurmise --toml slurmise.toml record "perfectScaler --intensity 5000 --duration 10"
```

Because it runs in the batch shell (no `SLURM_STEP_ID` there), it cannot know
which srun step just finished; pass `--step-id N` to attribute the record to a
specific step, otherwise sacct's last listed step is used.

It does a one-shot `sacct` read immediately and writes a complete row.
Trade-offs:
- **Read-your-writes**: the row is fully populated (runtime, max_rss, state)
  as soon as `record` returns, no later backfill step needed.
- **Racy**: there's no guarantee `sacct`/slurmdbd has committed the job's
  accounting data by the time `record` runs right after `srun`. If it hasn't,
  `record` fails loudly rather than silently retrying -- on a one-shot read
  that failure can mean permanently lost metrics for that job.
- **Double specification**: the command has to be written out twice (once for
  `srun`, once quoted for `record`), which is exactly the ergonomics we set
  out to avoid.

## `lazy-record` (record-then-exec)

Wraps the command directly, specified once:

```bash
srun ./lazy_slurmise --toml slurmise.toml lazy-record -- ../bin/perfectScaler --intensity 5000 --duration 10
```

It inserts a placeholder row (features known, runtime/memory absent) keyed by
`<jobid>.<stepid>`, then replaces itself with the wrapped command via
`os.execvp` -- there is no Python process left alive during the run, so exit
codes, signals, and memory accounting all belong to the real command with no
wrapper overhead or corruption. Notably, slurmise's HDF5 `JobDatabase` already
supports this natively: `record()` simply omits the memory/runtime datasets
when they are unknown, and `update_missing_data()` fills them from sacct
later. The prototype only adds the terminal-state guard and the
`method`/`state` bookkeeping (stored as HDF5 attributes on each job's group).

Metrics are filled in later, at read time (`display` or the standalone
`backfill` subcommand), which polls `sacct` and only writes
runtime/memory/state once the job has reached a **terminal** state
(COMPLETED/FAILED/TIMEOUT/OUT_OF_MEMORY/CANCELLED) -- so a still-running job
never gets partial data recorded against it. Trade-offs:
- **Single specification**, no wrapper process during the run.
- Metrics aren't available until someone calls `display`/`backfill` after
  the job finishes -- there's no "read-your-writes" immediately after
  `lazy-record` returns, by design.
- Likely fewer concurrent write situations because it just happens on backfill,
  not within each job

`no-update-display` prints the database as-is without backfilling, useful for
seeing which lazy rows are still pending while a job runs.
