# Enough runs in a loop to actually fit a model

## 01 — about this tutorial

In `01_single_job/` one recording wasn't enough to fit a model. Here we record
39 runs of the same job in a single allocation, train on them, and finally get a
prediction that comes from the data instead of from the toml's defaults.

There are two ways of working through this lesson:

1. Run `../tutorial.py 02_jobs_in_loop` for an interactive walkthrough
2. `cd` into this folder and type the `$` commands below yourself

The `#>` lines in the code blocks describe the expected output of the command
above them. They're shell comments — `tutorial.py` validates against them, and
you can ignore them if you're running manually.

**This lesson takes about 10 minutes**, nearly all of it the job sleeping.

**In a hurry, or no cluster to hand?** Where a block below submits a job it
offers two ways to do it — `cluster` really submits, `mock` runs a script that
states what the job would have used instead. Pick either; the rest of the lesson
holds the same way. Everything after the recording — the database, the fit, the
predictions — is the real thing regardless.

`tutorial.py` asks which you want. Working by hand, just type the one you
prefer. `../tutorial.py --option mock 02_jobs_in_loop` answers for you.

## 02 — the loop, and why `--step-id`

```bash
$ cat run_perfectScaler_loop.sbatch
#> expect /--step-id/
```

It walks 13 `(intensity, duration)` pairs and runs 3 replicates of each — 39
`srun` calls, each followed by a `slurmise record`, all inside a single
allocation.

That last part is why `--step-id` is there. All 39 `srun` calls share one
`$SLURM_JOB_ID`, because they are steps of the same job. Without something to
tell them apart, slurmise would file all 39 recordings under the same key and
each would overwrite the last. The `step` counter supplies that:

    slurmise --toml slurmise.toml record \
        --step-id "$step" \
        "perfectScaler --intensity $intensity --duration $duration"

`../04_array_jobs/` does the same work a different way, and needs no `--step-id`
at all — worth comparing once you get there.

The toml is `01_single_job/`'s with defaults filled in for both resources:

```bash
$ cat slurmise.toml
#> expect /default_mem/
```

## 03 — submit it

```bash
#> option cluster
$ sbatch --wait run_perfectScaler_loop.sbatch
#> expect ok
#> option mock
$ bash mock_perfectScaler_loop.sh
#> expect ok
```

`--wait` doesn't return until the job finishes, which here means all 39 steps —
the durations in the list add up to about 110 seconds per replicate.

## 04 — inspect

```bash
$ slurmise --toml slurmise.toml print
#> expect /perfectScaler/
```

39 records, each with the `intensity` and `duration` it was given plus the
memory and runtime it really used. Note the record ids: all 39 share one job id
and differ only in the step after the dot, which is `--step-id` doing its work.

## 05 — train

```bash
$ slurmise --toml slurmise.toml update-all
#> expect ok
```

`update-all` fits a model for every job in the database — here, just the one.
We skipped this in `01_single_job/` because it had nothing to fit: slurmise
holds back 20% of the runs to test the model against and wants at least 10 runs
left to train on, so it takes about 13 completed runs before a prediction stops
being the toml's defaults. One run was never going to do it; 39 comfortably
does.

The fitted model lands in this directory as `fits.json` and a couple of `.pkl`
files:

```bash
$ grep -o '"job_name": "[^"]*"' fits.json
#> expect /perfectScaler/
```

## 06 — predict

Ask for an `(intensity, duration)` pair that was never run — 2750 sits between
the 2500 and 3000 in the list, and no run used a duration of 7:

```bash
$ slurmise --toml slurmise.toml predict \
    "perfectScaler --intensity 2750 --duration 7"
#> expect /Predicted memory: [1-4]\d\d\d/
```

Memory comes back in the low thousands and runtime around 8 seconds — both from
the model now, not from `default_mem = 5000` and `default_time = 30` in the
toml. Compare that with `01_single_job/`, where the same command returned the
toml's numbers unchanged whatever you asked for.

`perfectScaler` is the easy case: it uses exactly the memory you ask it for, so
there is a clean line to fit. `../03_noisy_job/` runs the same loop with a job
that isn't so obliging.

## Starting over

The lesson only holds from a clean slate: with records left over from a previous
pass, the fit would be using more data than the 39 runs you just made. So throw
away the database and the fitted models before starting. `tutorial.py` runs
exactly this block for you, every time, before the lesson starts:

```bash
#> reset
$ rm -f slurmise.h5 fits.json *.pkl
$ rm -f out_slurm_logs/*.out
$ mkdir -p out_slurm_logs
```
