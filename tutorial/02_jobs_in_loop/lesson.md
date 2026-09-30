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

# The lesson only holds from a clean slate: with records left over from a previous
# pass, the fit would be using more data than the 39 runs made below.
rm -f slurmise.h5 fits.json *.pkl
rm -rf PolynomialFit
rm -f out_slurm_logs/*.out
mkdir -p out_slurm_logs
```

# Enough runs in a loop to fit a model

In {doc}`../01_single_job/lesson` one recording wasn't enough to fit a model.
Here we record 39 runs of the same job in a single allocation, train on them, and
finally get a prediction that comes from the data instead of from the toml's
defaults.

To follow along, `cd` into `02_jobs_in_loop/`.

## The loop, and why `--step-id`

```{code-cell}
cat run_perfectScaler_loop.sbatch
```

```{code-cell}
:tags: [remove-cell]
cat run_perfectScaler_loop.sbatch | grep -q -- --step-id
```

It walks 13 `(intensity, duration)` pairs and runs 3 replicates of each — 39
`srun` calls, each followed by a `slurmise record`, all inside a single
allocation.

That last part is why `--step-id` is there. All 39 `srun` calls share one
`$SLURM_JOB_ID`, because they are steps of the same job. Without something to tell
them apart, slurmise would file all 39 recordings under the same key and each
would overwrite the last. The `step` counter supplies that:

```bash
slurmise --toml slurmise.toml record \
    --step-id "$step" \
    "perfectScaler --intensity $intensity --duration $duration"
```

{doc}`../04_array_jobs/lesson` does the same work a different way, and needs no
`--step-id` at all — worth comparing once you get there.

The toml is lesson 01's with defaults filled in for both resources:

```{code-cell}
cat slurmise.toml
```

```{code-cell}
:tags: [remove-cell]
cat slurmise.toml | grep -q default_mem
```

## Submit it

On a cluster:

```bash
sbatch --wait run_perfectScaler_loop.sbatch
```

`--wait` doesn't return until the job finishes, which here means all 39 steps —
the durations in the list add up to about 110 seconds per replicate, so about six
minutes in total. Go grab a coffee.

This page has no scheduler, so it runs the mock instead. Same 39 recordings,
asserted rather than measured, and instant:

```{code-cell}
bash mock_perfectScaler_loop.sh
```

## Inspect

```{code-cell}
slurmise --toml slurmise.toml print | head -30
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print | grep -q perfectScaler
```

39 records, each with the `intensity` and `duration` it was given plus the memory
and runtime it used. Note the record ids: all 39 share one job id and differ only
in the step after the dot, which is `--step-id` doing its work.

## Train

```{code-cell}
slurmise --toml slurmise.toml update-all
```

`update-all` fits a model for every job in the database — here, just the one. We
skipped this in lesson 01 because it had nothing to fit: slurmise holds back 20%
of the runs to test the model against and wants at least 10 runs left to train on,
so it takes about 13 completed runs before a prediction stops being the toml's
defaults.

The fitted model lands in a sub-directory as `fits.json` and a couple of `.pkl`
files. A polynomial model is the current default:

```{code-cell}
grep -o '"job_name": "[^"]*"' PolynomialFit/*/fits.json
```

```{code-cell}
:tags: [remove-cell]
grep -o '"job_name": "[^"]*"' PolynomialFit/*/fits.json | grep -q perfectScaler
```

## Predict

Ask for an `(intensity, duration)` pair that was never run — 2750 sits between the
2500 and 3000 in the list, and no run used a duration of 7:

```{code-cell}
slurmise --toml slurmise.toml predict "perfectScaler --intensity 2750 --duration 7"
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml predict "perfectScaler --intensity 2750 --duration 7" 2>&1 \
    | grep -qE 'Predicted memory: [1-4][0-9][0-9][0-9]'
```

Memory comes back in the low thousands — from the model now, not from
`default_mem = 5000` and `default_time = 30` in the toml. Compare that with lesson
01, where the same command returned the toml's numbers unchanged whatever you
asked for.

`perfectScaler` is the easy case: it uses exactly the memory you ask it for, so
there is a clean line to fit. {doc}`../03_noisy_job/lesson` runs the same loop
with a job that isn't so obliging.
