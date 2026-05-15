# 02 — Jobs in a loop

Generate enough records to train a model for two different jobs, then use
`slurmise predict` to see how well each one is predicted.

## Files

- `perfectScaler` — deterministic memory.
- `complexMemScaler` — same interface, but adds +/-20% random noise to memory.
- `run_loop.sbatch` — loops over 13 intensities; runs both jobs at each one.
- `slurmise.toml` — declares both jobs.

## Run it

```bash
sbatch run_loop.sbatch
```

Total wall time: ~2.5 minutes (13 iterations x 2 jobs x ~10 s).

## Why `--step-id`?

This sbatch makes 26 `srun` calls inside one allocation, so all of them share
the same `$SLURM_JOB_ID`. Without `--step-id`, slurmise can't tell the
recordings apart and would clobber them. Each `record` call passes a
monotonically-increasing `--step-id` to keep them distinct.

## Inspect

```bash
slurmise --toml slurmise.toml print
```

You should see 13 records each for `perfectScaler` and `complexMemScaler`.

## Train and predict

Fit a model for every job in the database:

```bash
slurmise --toml slurmise.toml update-all
```

Now predict for an intensity that wasn't in the training set:

```bash
slurmise --toml slurmise.toml predict \
    "perfectScaler --intensity 2750 --duration 10"

slurmise --toml slurmise.toml predict \
    "complexMemScaler --intensity 2750 --duration 10"
```

Expect:
- `perfectScaler` memory prediction is essentially exact — the data is
  noiseless.
- `complexMemScaler` memory prediction is in the same ballpark but has more
  uncertainty because of the +/-20% noise that was injected at run time.

Runtime predictions for both jobs are uninteresting: every record had
`duration=10`, so there's nothing for slurmise to learn about how runtime
scales. The memory channel is the one worth predicting in this tutorial.
