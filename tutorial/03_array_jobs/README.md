# 03 — Array jobs

Use SLURM array jobs to record many runs in parallel, and see how slurmise
treats numeric vs. categorical parameters.

## Files

- `categoricalScaler` (shared [`../bin/`](../bin/)) — takes `--intensity` and
  `--duration` as category labels (`1|2|3`) instead of raw numbers. Internal
  lookup tables map each category to a memory and a sleep duration.
- `run_array.sbatch` — 16 array tasks: 8 for `perfectScaler`, 8 for
  `complexMemScaler`.
- `run_categorical_array.sbatch` — 9 array tasks, one per
  (intensity, duration) category combination.
- `slurmise.toml` — declares all three jobs.

## Why no `--step-id` here?

Each array task runs in its own SLURM job and gets its own `$SLURM_JOB_ID`,
so the records don't collide. Compare with tutorial 02, where all the runs
shared one job ID and needed `--step-id` to disambiguate.

## Numeric jobs

```bash
sbatch run_array.sbatch
slurmise --toml slurmise.toml print
```

You'll see 8 records each for `perfectScaler` and `complexMemScaler`. That's
below the 13-record threshold needed to train a model, so submit the same
sbatch a second time to get to 16 records each:

```bash
sbatch run_array.sbatch
slurmise --toml slurmise.toml update-all

slurmise --toml slurmise.toml predict \
    "perfectScaler --intensity 2750 --duration 10"
slurmise --toml slurmise.toml predict \
    "complexMemScaler --intensity 2750 --duration 10"
```

## Categorical job

```bash
sbatch run_categorical_array.sbatch
slurmise --toml slurmise.toml print
```

For categorical parameters, slurmise partitions the database by the
combination of category values: each `(intensity, duration)` pair is its own
sub-model. The 9 tasks here produce one record in each of the 9 buckets — so
*every* bucket is below the 13-record threshold and predict will fall back to
the toml defaults:

```bash
slurmise --toml slurmise.toml predict \
    "categoricalScaler --intensity 2 --duration 3"
```

To get a real prediction for one bucket, resubmit `run_categorical_array.sbatch`
many times to fill it. This is the natural trade-off with categorical
parameters: numeric parameters can share information across nearby values, but
categorical ones can't, so each combination needs its own data.
