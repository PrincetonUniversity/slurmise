---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Bash
  language: bash
  name: bash
---

```{code-cell}
:tags: [remove-cell]

# `./slrmise` is a `#!/usr/bin/env python3` script, so it runs under whichever
# python3 wins on PATH. The bash kernel that executes these cells when the docs are
# built sources the developer's ~/.bashrc, which may well put a different python3
# ahead of the one slurmise is installed into -- a conda base environment does
# exactly that. Put slurmise's own bin directory first so the interpreter that can
# `import click` is the one that runs. Nothing to do with the lesson: a reader who
# pip-installed slurmise into their active environment already has this.
PATH="$(dirname "$(command -v slurmise)"):$PATH"

rm -f slurmise.h5 fits.json *.pkl
rm -f out_slurm_logs/*.out
mkdir -p out_slurm_logs
unset SLRMISE_USED_MEM SLRMISE_USED_TIME SLRMISE_MOCK_JOB
```

# slurmise as a launcher

In {doc}`../01_single_job/lesson` you recorded a job by running it and then
telling slurmise about it afterwards:

```bash
srun ../bin/perfectScaler --intensity 2000 --duration 5

slurmise --toml slurmise.toml record \
    "perfectScaler --intensity 2000 --duration 5"
```

The command is written out twice — once for `srun` to execute, once as a quoted
string for `record` to parse against the `job_spec` — and the two have to agree.
Change one and forget the other and slurmise happily records a job that never ran.

This lesson tries the other arrangement: let slurmise **wrap** the command, the way
`time` does.

```bash
srun ./slrmise --toml slurmise.toml lazy-record -- \
    ../bin/perfectScaler --intensity 2000 --duration 5
```

Written once. And because slurmise is now running *inside* the `srun` step, it can
read `SLURM_STEP_ID` for itself instead of being told.

You'll run the *same three* `perfectScaler` jobs twice, once each way, so the two
sets of records can be put side by side. Three rather than one because the
difference between the approaches is mostly bookkeeping, and bookkeeping only shows
up in a loop.

To follow along, `cd` into `88_as_a_launcher/`.

## Files of interest

`./slrmise` is the prototype. It stands in for a hypothetical
`slurmise lazy-record`; the name is deliberately misspelled (no `u`), and it is not
on the `PATH`, so it is always run as `./slrmise`:

```{code-cell}
./slrmise --help
```

```{code-cell}
:tags: [remove-cell]
./slrmise --help | grep -q 'Commands:'
```

`./slrmise record` is upstream `slurmise record`, unchanged, plus two bookkeeping
attributes so `display` can put both kinds of row in one table. The eager half of
this lesson is not a new idea — it is lesson 01 in a loop.

`../bin/perfectScaler` is the same toy from the earlier lessons: `--intensity`
controls the memory it holds and `--duration` the seconds it holds it for.

```{code-cell}
cat slurmise.toml
```

```{code-cell}
:tags: [remove-cell]
cat slurmise.toml | grep -q job_spec
```

`srun_then_record.sbatch` and `run_as_launcher.sbatch` are the same three runs,
recorded the two different ways. Read them together; the difference between them is
the lesson:

```{code-cell}
cat srun_then_record.sbatch
```

```{code-cell}
:tags: [remove-cell]
cat srun_then_record.sbatch | grep -q -- --step-id
```

```{code-cell}
cat run_as_launcher.sbatch
```

```{code-cell}
:tags: [remove-cell]
cat run_as_launcher.sbatch | grep -q lazy-record
```

The eager loop carries a `step` counter, because `record` runs in the batch shell
after `srun` has returned and has no way to know which step it means. The launcher
loop carries nothing: it runs *inside* the step.

## The eager way: srun, then record

On a cluster:

```bash
sbatch --wait srun_then_record.sbatch
```

Mocked here:

```{code-cell}
bash mock_srun_then_record.sh
```

Three complete rows, no waiting:

```{code-cell}
./slrmise --toml slurmise.toml display --no-sync
```

```{code-cell}
:tags: [remove-cell]
./slrmise --toml slurmise.toml display --no-sync | grep -c eager | grep -q 3
```

Both metrics are already there: `record` ran after each step finished and asked
`sacct` on the spot. That is the appeal of doing it this way.

The state is another matter. On the cluster it reads `RUNNING`, because that is what
the *job* was doing when `record` asked — the batch job still had steps left to
launch. So those rows are not quite finished either. On the mock path there is no
job to be running, so it says `COMPLETED` from the start.

## The launcher way: slurmise wraps the command

Now the same three runs again, submitted the other way:

```bash
sbatch --wait run_as_launcher.sbatch
```

Mocked here:

```{code-cell}
bash mock_run_as_launcher.sh
```

Three more rows, and this time they are empty:

```{code-cell}
./slrmise --toml slurmise.toml display --no-sync
```

```{code-cell}
:tags: [remove-cell]
./slrmise --toml slurmise.toml display --no-sync | grep -c lazy | grep -q 3
```

`-` for both metrics, because each row was written *before* its command ran — at
that moment there was nothing to measure. What the row does have is its features and
a correct `<jobid>.<stepid>` key, and nobody had to count to get it.

## Sync fills them in

`display` syncs before printing, which is what turns all six rows into their final
form. On a cluster you may need to run it a few times until the lazy rows read
`COMPLETED` — each attempt is another look at `sacct`, so that doubles as the wait:

```{code-cell}
./slrmise --toml slurmise.toml display
```

```{code-cell}
:tags: [remove-cell]
./slrmise --toml slurmise.toml display | grep -c 'lazy .*COMPLETED' | grep -q 3
```

Six records of the same three jobs, and once settled you cannot tell from the
numbers which way each was taken. That is the point: the launcher is a different
route to the same database, not a different kind of data.

A row is **settled** once it has both metrics *and* a terminal state (`COMPLETED`,
`FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`). Anything else is looked up again
on the next read. That one rule covers two different kinds of unfinished business:
the lazy rows were missing their metrics, the eager rows were missing their verdict,
and both converge without either being special-cased.

Metrics are only filled when they are absent, never overwritten — a terminal step's
numbers do not change, so there is nothing to gain by asking twice.

## The trade-off

Neither approach is free.

**Eager `record`** gives you read-your-writes: the row is complete the moment
`record` returns, and nothing has to run later. In exchange, the command is written
twice, the loop has to track `--step-id` itself, and it races slurmdbd — if the
accounting data has not landed yet, `record` has nothing to read.

**Lazy `lazy-record`** writes the command once and gets the step id for free, so a
loop of three `srun` steps needs no counter, and neither would a loop of a hundred.
`os.execvp` means no python process is left alive during the run: exit codes,
signals, and memory accounting all belong to the real command, with no wrapper
overhead sitting in the way. In exchange the database is eventually consistent — the
metrics are not there until something calls `sync`.

The stub is not a new concept slurmise had to grow for this. `record()` already omits
the memory and runtime datasets when they are unknown, and `update_missing_data()`
already fills them from `sacct` later, splitting a `<jobid>.<stepid>` key and asking
about that step. `lazy-record` is those two pieces used in the order they were
always able to support.

```{note}
If you work through this lesson by hand on the mock path and then want a pass that
really submits, clear the mock environment first — the `export`s the mock scripts
set would otherwise keep `./slrmise` from ever touching SLURM:

    unset SLRMISE_USED_MEM SLRMISE_USED_TIME SLRMISE_MOCK_JOB
```
