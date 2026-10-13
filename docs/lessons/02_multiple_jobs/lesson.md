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
# pass, the fit would be using more data than the 13 runs made below.
rm -f slurmise.h5 fits.json *.pkl
rm -rf PolynomialFit
rm -f out_slurm_logs/*.out
mkdir -p out_slurm_logs
```

# Recording multiple jobs

In {doc}`../01_single_job/lesson` one recording wasn't enough to fit a model.
Here we record 13 runs of the same job in a single allocation, train on them, and
finally get a prediction that comes from the data instead of from the toml's
defaults.

To follow along, `cd` into `02_multiple_jobs/`.

## Many runs in one job, and why `--step-id`

`run_perfectScaler_loop.sbatch` runs `perfectScaler` 13 times, each with a
different `--intensity` and `--duration`, and records each run straight after it
finishes. Here are the first three:

```{literalinclude} run_perfectScaler_loop.sbatch
:language: bash
:caption:
:lines: 1-20
```

```{code-cell}
:tags: [remove-cell]
cat run_perfectScaler_loop.sbatch | grep -q -- --step-id
```

The other ten follow the same pattern, through `--intensity 6500`.

All 13 `srun` calls share one `$SLURM_JOB_ID`, because they are steps of the same
job. Without something to tell them apart, slurmise would file all 13 recordings
under the same key and each would overwrite the last. `--step-id` gives each
record its own number.

The toml is lesson 01's with defaults filled in for both resources:

```{literalinclude} slurmise.toml
:language: toml
:caption:
```

```{code-cell}
:tags: [remove-cell]
cat slurmise.toml | grep -qF '[slurmise.job.perfectScaler.memory]'
```

## Submit it

::::{tab-set}
:::{tab-item} On a cluster
:sync: cluster
```bash
mkdir -p out_slurm_logs
sbatch --wait run_perfectScaler_loop.sbatch
```

`--wait` doesn't return until the job finishes, which here means all 13 steps —
about two minutes.
:::

:::{tab-item} Mocked
:sync: mock
```bash
bash mock_perfectScaler_loop.sh
```

Nothing is submitted. The mock makes the same 13 recordings with `slurmise
raw-record`, each with its own `--step-id`. The first three:

```{literalinclude} mock_perfectScaler_loop.sh
:language: bash
:caption:
:lines: 1-14
```
:::
::::

```{code-cell}
:tags: [remove-cell]
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

13 records, each with the `intensity` and `duration` it was given plus the memory
and runtime it used. Note the record ids: all 13 share one job id and differ only
in the step after the dot, which is `--step-id` doing its work.

The same records are easier to take in as a picture. `plot_records.py` reads them
back out of `slurmise.h5` with slurmise's Python API and plots memory against
`--intensity`. It needs `matplotlib` installed alongside slurmise.

It leaves runtime out on purpose. These jobs are unrealistically short — seconds,
not minutes — and slurmise records runtime in whole minutes, so on a cluster every
one of them is stored as `0` (see {doc}`../../tutorial`). The lessons are only
trying to fit memory:

```{literalinclude} plot_records.py
:language: python
:caption:
```

```{code-cell}
python plot_records.py
```

```{code-cell}
:tags: [remove-input]
display < records.png && rm records.png
```

Memory lies on a straight line through intensity. That clean line is what the
model is about to learn.

## Train

```{code-cell}
slurmise update-all
```

`update-all` fits a model for every job in the database — here, just the one. We
skipped this in lesson 01 because it had nothing to fit: slurmise holds back 20%
of the runs to test the model against and wants at least 10 runs left to train on,
so it takes 13 completed runs before a prediction stops being the toml's
defaults — exactly what we have.

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
slurmise predict "perfectScaler --intensity 2750 --duration 7"
```

```{code-cell}
:tags: [remove-cell]
slurmise predict "perfectScaler --intensity 2750 --duration 7" 2>&1 \
    | grep -qE 'Predicted memory: [1-4][0-9][0-9][0-9]'
```

Memory comes back in the low thousands — from the model now, not from
the toml's defaults of 5000 MB and 30 minutes. Compare that with lesson
01, where the same command returned the toml's numbers unchanged whatever you
asked for.

## Using bash loops

Writing every run out by hand keeps the script easy to read, but it doesn't scale.
For your own jobs, a loop inside the `.sbatch` does the same thing, with the loop
counter as the `--step-id`:

```bash
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(    2    4    8    8    4   10   12   14    2   13   19    5    9)

for step in "${!intensities[@]}"; do
    intensity=${intensities[$step]}
    duration=${durations[$step]}

    srun ../bin/perfectScaler --intensity "$intensity" --duration "$duration"
    slurmise record --step-id "$step" \
        "perfectScaler --intensity $intensity --duration $duration"
done
```

## Using the Python API

If your workflow already lives in Python, you can loop there instead and record
through slurmise's Python API rather than the `slurmise` command. `Slurmise()`
finds `slurmise.toml` the same way the command does, and `record` reads the job id
from `$SLURM_JOB_ID` just like `slurmise record`.

`run_perfectScaler.py` does the same 13 runs this way. `sbatch` will submit any
script with a shebang, and reads `#SBATCH` lines from it as long as they come
before the first line of code, so the Python file is its own job script:

```{literalinclude} run_perfectScaler.py
:language: python
:caption:
```

```{code-cell}
:tags: [remove-cell]
head -1 run_perfectScaler.py | grep -q '^#!' && grep -q '^#SBATCH --job-name' run_perfectScaler.py
```

```bash
mkdir -p out_slurm_logs
sbatch --wait run_perfectScaler.py
```

The `python3` the shebang finds has to be one with slurmise installed. `sbatch`
passes your environment along to the job, so activating that environment before
you submit is enough.

{doc}`../04_array_jobs/lesson` does the same work a different way, and needs no
`--step-id` at all — worth comparing once you get there.

`perfectScaler` is the easy case: it uses exactly the memory you ask it for, so
there is a clean line to fit. {doc}`../03_noisy_job/lesson` runs the same loop
with a job that isn't so obliging.
