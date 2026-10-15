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

In {doc}`../01_single_job/lesson` we saw that one recording wasn't enough to fit a model.
Here we record 13 runs of the same command in a single job, train on them, and
get a prediction of how much memory would be required for a future run of this command.

We're still using the `perfectScaler` command from the last lesson.

To follow along, `cd` into `02_multiple_jobs/`.

## Many runs in one job, and why `--step-id`

The `run_perfectScaler_loop.sbatch` file runs `perfectScaler` 13 times, each with a
different `--intensity`, and records each run straight after it
finishes. Here is the head of this repetitive file:

```{literalinclude} run_perfectScaler_loop.sbatch
:language: bash
:caption:
:lines: 1-20
```

```{code-cell}
:tags: [remove-cell]
cat run_perfectScaler_loop.sbatch | grep -q -- --step-id
```

#RB NOTE
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
a minute or so. There's also the wait time of the job queueing and starting.
:::

:::{tab-item} Mocked
:sync: mock
```bash
bash mock_perfectScaler_loop.sh
```

Nothing is submitted. The mock makes the same 13 recordings with `slurmise
raw-record`, each with its own `--step-id`. Again, this is a mock for the sake
of the tutorial. In practice you would use `srun` with `slurmise record`.

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

After your job, or mock, runs of the `perfectScaler` command finish, each
one should be recorded in the `slurmise.h5` database:

```{code-cell}
slurmise print slurmise.h5 | head -30
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print | grep -q perfectScaler
```

13 records, each with the `intensity` parameter it was given as well as the
memory and runtime it used. Note the record ids: all 13 share one job id and
differ only in the step after the dot, which is `--step-id` doing its work.

Here are the same records as a table, ordered by intensity.

```{code-cell}
:tags: [remove-input]
{ echo "record intensity memory"
  slurmise --toml slurmise.toml print \
    | awk '/^\|--- / {id = $2} /intensity:/ {i = $NF} /memory:/ {print id, i, $NF}' \
    | sort -n -k2
} | column -t
```

Again, the table leaves runtime out on purpose since this command is unrealistically short
and we're just focusing on fitting memory in this tutorial.

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print \
    | awk '/^\|--- / {id = $2} /intensity:/ {i = $NF} /memory:/ {print id, i, $NF}' \
    | wc -l | grep -q 13
```

## Train

```{code-cell}
slurmise update-all
```

Now that we have enough measurments, we can train a model to understand the
resource requirements of the `perfectScaler` command.

`update-all` fits a model for every command in the database — here, just `perfectScaler`.

How this is working under the hood is that slurmise splits the recordings into
test and train datasets and by default fits a 2nd order polynomial model on the training set.
It then uses the held-out testing data to estimate the model accuracy.

The fitted model lands in a sub-directory as `fits.json` and a couple of `.pkl`
files. Since a polynomial model is the current default then `PolynomialFit/`
folder is created.

```{code-cell}
:tags: [remove-cell]
grep -o '"job_name": "[^"]*"' PolynomialFit/*/fits.json
```

```{code-cell}
:tags: [remove-cell]
grep -o '"job_name": "[^"]*"' PolynomialFit/*/fits.json | grep -q perfectScaler
```

## Predict

Ask for an intensity that was never run — 2750 sits between the 2500 and 3000 in
the list:

```{code-cell}
slurmise predict "perfectScaler --intensity 2750"
```

```{code-cell}
:tags: [remove-cell]
slurmise predict "perfectScaler --intensity 2750" 2>&1 \
    | grep -qE 'Predicted memory: [1-4][0-9][0-9][0-9]'
```

Memory comes back at about 2750 MB — from the model now, not from the toml's
default of 5000. Compare that with lesson 01, where the same command returned the
toml's numbers unchanged whatever you asked for.

Runtime is still the toml's default of 30 minutes, with a warning. Every record
says `0` minutes, so the runtime model predicts `0`, and slurmise falls back to the
default rather than suggest a job that takes no time at all.

## Using bash loops

Writing every run out by hand keeps the script easy to read, but it doesn't scale.
For your own jobs, a loop inside the `.sbatch` does the same thing, with the loop
counter as the `--step-id`:

```bash
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)

for step in "${!intensities[@]}"; do
    intensity=${intensities[$step]}

    srun ../bin/perfectScaler --intensity "$intensity"
    slurmise record --step-id "$step" \
        "perfectScaler --intensity $intensity"
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
