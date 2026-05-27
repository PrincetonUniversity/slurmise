# 01 — Single job

Record a single `perfectScaler` job and see what `slurmise predict` does
*before* slurmise has enough data to train a model.

To follow along please `cd` into the folder that contains this README
`tutorial/01_single_job` since the example code uses some relative paths.

## Files

- `run_perfectScaler.sbatch` — runs `perfectScaler` (from the shared
  [`../bin/`](../bin/)) once at intensity 5000, duration 10.
- `slurmise.toml` — declares `perfectScaler` with two numeric parameters.

## Run it

```bash
sbatch run_perfectScaler.sbatch
```

(this should only run for 10 seconds)

## Inspect

After the job completes, look at what was recorded:

```bash
slurmise --toml slurmise.toml print
```

You should see one record for `perfectScaler` with intensity 5000, duration 10.

## Predict

Now ask slurmise what it would predict for a *different* intensity:

```bash
slurmise --toml slurmise.toml predict "perfectScaler --intensity 4000 --duration 10"
```

The prediction will not actually use the recorded value — slurmise needs
multiple records per job before it will fit a model. With one record it falls
back to the defaults:

```
Predicted runtime: 234
Predicted memory: 1000

Warnings:
  Not enough fitting data points in the fits. Returning default values.
```

Note that `default_time` is specified in the `slurmise.toml` to be 234, but `default_mem`
is not specified. Slurmise has it's own defaults of 60 minutes and 1 GB, but it is highly
recommended to specify defaults in the toml file.

```toml
[slurmise]
base_dir = "."

[slurmise.job.perfectScaler]
job_spec = "--intensity {intensity:numeric} --duration {duration:numeric}"
default_time = 234
```

## Next tutorial

This sets up the next tutorial: generate enough records to actually train a model.
