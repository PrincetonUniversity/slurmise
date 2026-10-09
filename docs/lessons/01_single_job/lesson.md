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

# Start from nothing. The last section of this lesson only holds with no fitted
# model -- with one left over from a previous pass, `predict` would answer from it
# and the warning the lesson is built around would never appear.
rm -f slurmise.h5
rm -rf PolynomialFit
rm -f out_slurm_logs/*.out
mkdir -p out_slurm_logs
```

# Recording a single job

This is the first lesson, so it starts from nothing: one job, one recording, and
a `predict` that can't yet answer from your data.

To follow along, `cd` into `01_single_job/` in the tutorial tarball.

## The files

`../bin/perfectScaler` is the command we're recording. It's a small Python script
that simulates using a certain amount of time and memory, controlled by the
`--duration` and `--intensity` arguments respectively. This uninteresting process
is good for a tutorial, since we know ahead of time how much time and memory the
job needs.

`slurmise.toml` is the config. `job_spec` tells slurmise how to parse the command
into features, and the `default` under `[slurmise.job.perfectScaler.runtime]` is
the guess slurmise falls back on until a model has been trained:

```{code-cell}
---
mystnb:
  text_lexer: toml
---
cat slurmise.toml
```

```{code-cell}
:tags: [remove-cell]
cat slurmise.toml | grep -q job_spec
```

The `job_spec` has to agree with the command: it names `--intensity` and
`--duration`, so slurmise expects to find exactly those on the command line it is
asked to record. `base_dir = "."` is where the `slurmise.h5` database will be
created.

`run_perfectScaler.sbatch` runs `perfectScaler` once:

```{code-cell}
---
mystnb:
  text_lexer: bash
---
cat run_perfectScaler.sbatch
```

```{code-cell}
:tags: [remove-cell]
cat run_perfectScaler.sbatch | grep -q 'slurmise --toml slurmise.toml record'
```

Notice that `slurmise record` is called **inside the job**, after the `srun`
finishes. That is the whole lesson: `record` asks SLURM what the step it just ran
actually used, so it has to run on the compute node alongside the work.

One consequence worth keeping in mind as you go: the command is written out twice
— once for `srun` to execute, once as a string for `record` to parse against the
`job_spec`. They have to match. Change one and forget the other and slurmise
happily records a job that never ran.

The `#SBATCH --output=` line keeps the job's log in `out_slurm_logs/` rather than
dropping it in this directory. Nothing here prints to it, but your own jobs will.

That directory has to exist before you submit. SLURM opens the output file itself,
before your script runs, and it will not create a missing directory — the job dies
without writing anything, and the reason only shows up in the `slurmd` log. A
`mkdir` inside the `.sbatch` is too late to help, so it goes in the shell first.

## Run it

On a cluster, submit it. `--wait` blocks until the job finishes, so there's no
polling loop to write — the command simply doesn't return for about ten seconds:

```bash
mkdir -p out_slurm_logs
sbatch --wait run_perfectScaler.sbatch
```

This page has no scheduler, so it takes the mock path instead. Rather than
submitting anything, the mock tells slurmise directly what the job would have
used, with `raw-record`:

```{code-cell}
slurmise --toml slurmise.toml raw-record \
    --job-name perfectScaler \
    --slurm-id 12345 \
    --numerics '"intensity":5000,"duration":10' \
    --used-minutes 12 \
    --used-mbs 5020
```

Neither `perfectScaler` nor `slurmise record` prints anything, so on the cluster
path the job's log in `out_slurm_logs/` is empty and there is nothing to read
there. The recording went into the database instead, which is where we look next.

## Inspect

The record outlives the job — it's in `slurmise.h5` now, so you can ask from the
login node:

```{code-cell}
slurmise --toml slurmise.toml print
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print | grep -q intensity
```

One record for `perfectScaler` with intensity 5000 and duration 10, alongside the
memory and runtime it used. On a cluster the job id is SLURM's and the memory and
runtime are measured, so your numbers won't match those exactly. The `intensity`
and `duration` values will: they were parsed straight out of the command by the
`job_spec`.

## Predict

Now ask slurmise what it would predict for a *different* intensity:

```{code-cell}
slurmise --toml slurmise.toml predict "perfectScaler --intensity 4000 --duration 10"
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml predict "perfectScaler --intensity 4000 --duration 10" 2>&1 \
    | grep -q 'No model has been fit'
```

Unfortunately, a warning that no model has been fit.

slurmise only predicts from a model once one has been fit, and fitting needs many
records per job — with one there is nothing to fit, so it falls back to the
defaults. So `234` is the runtime `default` straight out of `slurmise.toml` — no
model was consulted at all. No memory default is set there, so the memory figure
is slurmise's own built-in default of 1 GB.

Those job-agnostic built-in defaults of 60 minutes and 1 GB are arbitrary. It is
good practice to set both defaults in your toml so the fallback guess is at least
in the right range for your job.

Next: {doc}`../02_jobs_in_loop/lesson`, where we generate enough records to train
a model and get good predictions.
