# 02 — Jobs in a loop

Generate enough records to train a model for two different jobs, then use
`slurmise predict` to see how well each one is predicted.

## Files

- `run_perfectScaler_loop.sbatch` — loops over 13 (intensity, duration) pairs; runs the
  perfectScaler job at each one
- `run_complexMemScaler_loop.sbatch` — same but runs the complexMemScaler
- `slurmise.toml` — declares both jobs.

Take a looks at `../bin/complexMemScaler`, it's the same as `../bin/perfectScaler` except
it adds some randomness around how much memory is utilized.

## Submit both jobs

```bash
sbatch run_perfectScaler_loop.sbatch
sbatch run_complexMemScaler_loop.sbatch
```

You can submit both jobs and have them running at the same time.

Total wall time should be around 3 minutes for each job.

## Why `--step-id`?

If you look in the `.sbatch` files you'll see a `step` variable which gets
incremented after each `srun` command and gets passed to `slurmise record`.

The reason for this is that the sbatch makes 13 `srun` calls inside one
allocation, so all of them share the same `$SLURM_JOB_ID`. Without `--step-id`,
slurmise can't tell the recordings apart and would overwrite them.

## Inspect

```bash
slurmise --toml slurmise.toml print
```

You should see 13 records each for `perfectScaler` and `complexMemScaler`.

## Train

Fit a model for every job in the database:

```bash
slurmise --toml slurmise.toml update-all
```

We didn't do this step for the `01_single_job/` example
since it would have failed with a message saying something like
"not enough training data".

## Predict

Now predict for an (intensity, duration) pair that wasn't in the training set for the perfectScaler:

```bash
slurmise --toml slurmise.toml predict "perfectScaler --intensity 2750 --duration 7"
```

Your exact results may vary, but you'll likely get a pretty good estimate:

```
Predicted runtime: 30
Predicted memory: 2761.7625320476973
[33mWarnings:[0m
  Runtime prediction for job perfectScaler is not within 20% of actual value.
  Returing default runtime value.
```

NOTE!! The runtime prediction is currently around 30% mpe. Maybe because the runtimes are so short?

Let's see how well the prediction does on the complexMemScaler:

```bash
slurmise --toml slurmise.toml predict "complexMemScaler --intensity 2750 --duration 7"
```

```
Predicted runtime: 30
Predicted memory: 2761.7625320476973
[33mWarnings:[0m
  Runtime prediction for job complexMemScaler is not within 20% of actual value.
  Returing default runtime value.
```

NOTE!! Not sure why the exact same memory prediction is returned

Expect:
- `perfectScaler` memory prediction is essentially exact — the data is
  noiseless.
- `complexMemScaler` memory prediction is in the same ballpark but has more
  uncertainty because of the +/-20% noise that was injected at run time.
- Runtime predictions for both jobs should track the requested `--duration`
  closely, since the scripts simply `sleep` for that many seconds.
