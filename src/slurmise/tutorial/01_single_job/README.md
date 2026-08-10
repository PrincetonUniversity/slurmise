# Record one job, and predict before there's a model

## 01 — about this tutorial

Record a single job with slurmise, then ask it to predict — and see why one
recording isn't enough.

There are two ways of working through this lesson:

1. Run `../tutorial.py 01_single_job` for an interactive walkthrough
2. `cd` into this folder and type the `$` commands below yourself

Either way the commands run from inside `01_single_job/`: the example uses
relative paths to the shared `../bin/` scripts.

The `#>` lines in the code blocks describe the expected output of the command
above them. For example this block runs `sbatch` and checks that the string
"Submitted batch job" appears in the output:

    $ sbatch --wait run_perfectScaler.sbatch
    #> expect /Submitted batch job/

These `#>` lines are for `tutorial.py` to validate against — they're shell
comments, so you can ignore them if you're running manually.

The `sbatch` below really goes to the queue. The job runs for about 10 seconds,
plus however long it waits to start.

**In a hurry, or no cluster to hand?** Where a block below submits a job it
offers two ways to do it — `cluster` really submits, `mock` runs a script that
states what the job would have used instead. Pick either; the rest of the lesson
holds the same way. Everything after the recording — the database, the fit, the
predictions — is the real thing regardless.

`tutorial.py` asks which you want. Working by hand, just type the one you
prefer. `../tutorial.py --option mock 01_single_job` answers for you.

## 02 — the files

`../bin/perfectScaler` is the command we're recording. It's a small python
script that simulates using a certain amount of time and memory, controlled by
the `--duration` and `--intensity` arguments respectively. It's a helpful toy
because we know exactly what each invocation will cost — which is why it's
named `perfectScaler`.

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

The prediction does not use the value we just recorded:

    Predicted runtime: 234
    Predicted memory: 1000

    Warnings:
      Not enough fitting data points in the fits. Returning default values.

slurmise needs many records per job before it will fit a model, and with one it
falls back to the defaults. So `234` is `default_time` straight out of
`slurmise.toml` — no model was consulted at all. `default_mem` isn't specified
there, so the memory figure is slurmise's own built-in default of 1 GB.

Those built-ins (60 minutes and 1 GB) are a backstop, not a recommendation. Set
both defaults in your toml so the cold-start guess is at least in the right
range for your job.

That sets up the next tutorial, `../02_jobs_in_loop/`: generate enough records
to actually train a model, so `predict` starts answering from your data instead
of your defaults.

# TODO: This should automatically happen, the user shouldn't be responsible!
## Starting over

Section 05 only holds from a clean slate — with records left over from a previous
pass, `predict` may have enough data to fit a model and the "not enough fitting
data points" warning won't appear. So throw away the database before starting.
`tutorial.py` runs exactly this block for you, every time, before the lesson
starts:

```bash
#> reset
$ rm -f slurmise.h5 fits.json *.pkl
$ rm -f out_slurm_logs/*.out
$ mkdir -p out_slurm_logs
```

The logs go too, so a second pass doesn't leave you sifting through the last
one's.
