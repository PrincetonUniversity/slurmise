# 01 — Single job

Record a single `perfectScaler` job and see what `slurmise predict` does
*before* slurmise has enough data to train a model.

To follow along please `cd` into the folder that contains this README
`01_single_job/` since the example code uses some relative paths of the
binary files.

## Files

`run_perfectScaler.sbatch` — runs `perfectScaler` (from the shared
[`../bin/`](../bin/)) once at intensity 5000 and duration 10. It
includes an `srun` command to actually run the process with slurm and
then a `slurmise record` command to store the resulting time and memory
used by the prior `srun.

Feel free to take a look at `../bin/perfectScaler`. It's a simple python
script which simulates using a certain amount of time and memory which are
controlled by passing in the `--intensity` and `--duration` arguments.

Also take a look at `slurmise.toml` which is used to parse the `srun` invocation
and store the job info in the `slurmise.h5` hdf5 file which will get created.

```toml
[slurmise]
base_dir = "."

[slurmise.job.perfectScaler]
job_spec = "--intensity {intensity:numeric} --duration {duration:numeric}"
default_time = 234
```


## Run it

```bash
sbatch run_perfectScaler.sbatch
```

(this should only run for 10 seconds)

## Inspect

After the job completes, look at what was recorded in the
new `slurmise.h5` file:

```bash
slurmise --toml slurmise.toml print
```

You should see one record for `perfectScaler` with intensity 5000, duration 10 as
well as the resulting memory and runtime.

```
perfectScaler
|--- 8844261
|    |--- duration: () float64 10.0
|    |--- intensity: () float64 5000.0
|    |--- memory: () int64 5021
|    |--- runtime: () int64 12
attrs:
```

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

## Next tutorial

This sets up the next tutorial: generate enough records to actually train a model.
