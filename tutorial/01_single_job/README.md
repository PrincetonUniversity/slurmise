# slurmise record a job

## 01 — about this tutorial

Now we'll see how to record a single job with slurmise. This is the first
lesson, so it starts from nothing: one job, one recording, and a `predict` that
can't yet answer from your data.

`00_introduction/` covers how the lessons work and what the `#>` lines mean.

## 02 — important files

`../bin/perfectScaler` is the command we're recording. It's a small python
script that simulates using a certain amount of time and memory, controlled by
the `--duration` and `--intensity` command line arguments respectively.

This unintersting process is good for a tutorial since
we know ahead of time how much time and memory the job needs.

`slurmise.toml` is the config. `job_spec` tells slurmise how to parse the
command into features, and `default_time` / `default_mem` are the guesses
slurmise falls back on until a model has been trained:

```bash
$ cat slurmise.toml
#> expect /job_spec/
```

The `job_spec` has to agree with the command: it names `--intensity` and
`--duration`, so slurmise expects to find exactly those on the command line it
is asked to record. `base_dir = "."` is where the `slurmise.h5` database will be
created.

`run_perfectScaler.sbatch` runs `perfectScaler` once:

```bash
$ cat run_perfectScaler.sbatch
#> expect /slurmise --toml slurmise.toml record/
```

Notice that `slurmise record` is called **inside the job**, after the `srun`
finishes. That is the whole lesson: `record` asks slurm what the step it just
ran actually used, so it has to run on the compute node alongside the work.

One consequence worth keeping in mind as you go: the command is written out
twice — once for `srun` to execute, once as a string for `record` to parse
against the `job_spec`. They have to match.

The `#SBATCH --output=` line keeps the job's log in `out_slurm_logs/` rather
than dropping it in this directory. Nothing here prints to it, but your own jobs
will.

## 03 — run it

Submit it. `--wait` blocks until the job finishes, so there's no polling loop to
write — the command simply doesn't return for about ten seconds:

```bash
#> option cluster
$ sbatch --wait run_perfectScaler.sbatch
#> expect ok
#> option mock
$ bash mock_perfectScaler.sh
#> expect ok
```

Neither `perfectScaler` nor `slurmise record` prints anything, so the job's log
in `out_slurm_logs/` is empty and there is nothing to read there. The recording
went into the database instead, which is where we look next.

## 04 — inspect

The record outlives the job — it's in `slurmise.h5` now, so you can ask from
the login node:

```bash
$ slurmise --toml slurmise.toml print
#> expect /intensity/
```

You should see one record for `perfectScaler` with intensity 5000 and duration
10, alongside the memory and runtime it really used:

    perfectScaler
    |--- 8844261
    |    |--- duration: () float64 10.0
    |    |--- intensity: () float64 5000.0
    |    |--- memory: () int64 5021
    |    |--- runtime: () int64 12
    attrs:

The job id is slurm's, and the memory and runtime are measured, so your numbers
won't match those exactly. The `intensity` and `duration` values will: they were
parsed straight out of the command by the `job_spec`.

## 05 — predict

Now ask slurmise what it would predict for a *different* intensity:

```bash
$ slurmise --toml slurmise.toml predict \
    "perfectScaler --intensity 4000 --duration 10"
#> expect /Not enough fitting data points/
```

Unfortunately, you should see a warning about not enough data points.

This is because slurmise needs many records per job before it will fit a model,
and with one it falls back to the defaults. So `234` is `default_time` straight
out of `slurmise.toml` — no model was consulted at all. `default_mem` isn't
specified there, so the memory figure is slurmise's own built-in default of 1
GB.

The slurmise job-agnostic built-in defaults of 60 minutes and 1 GB are
arbitrary. It is good practice to set both defaults in your toml so the default
guess is at least in the right range for your job.

Now you're read for the next tutorial, `../02_jobs_in_loop/`: where we
actually generate enough records to train a model and get good predictions.
