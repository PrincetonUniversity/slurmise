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
# Delete artifacts that get produced in this lesson
rm -f slurmise.h5
rm -rf PolynomialFit
rm -f out_slurm_logs/*.out
mkdir -p out_slurm_logs
```

# Recording a single job

To follow along, `cd` into `01_single_job/` in the tutorial tarball.

## The files

### A command with linear memory scaling

`../bin/perfectScaler` is the command we're recording. It's a small Python script
that simulates using a configurable amount of memory, controlled by its `--intensity`
argument. This toy command is good for a tutorial since we know ahead
of time how much memory the job will need!

```{literalinclude} ../bin/perfectScaler
:language: python
:caption: ../bin/perfectScaler
```

The `bytearray` allocates `--intensity`
megabytes, and `time.sleep` holds on to them for a second so SLURM's accounting
has a chance to see the amount of memory used.

### A TOML config to describe the command to slurmise

`slurmise.toml` tells slurmise which commands it should know about and how to read
them:

```{literalinclude} slurmise.toml
:language: toml
:caption:
```

```{code-cell}
:tags: [remove-cell]
cat slurmise.toml | grep -q job_spec
```

Taking it a table at a time:

- **`[slurmise]`**: Settings for slurmise as a whole. `base_dir` is the only
  required one: it's the directory slurmise keeps its files in. The database of
  recorded jobs, `slurmise.h5`, goes there, and so do the models it fits later.
  `"."` means this lesson's directory.

- **`[slurmise.job.perfectScaler]`**: Declares a job named `perfectScaler`. Every
  table under `[slurmise.job.…]` is one job, and a config can declare as many as you
  like. The name is how slurmise recognises a command: when you record
  `"perfectScaler --intensity 5000"`, it matches the start of that string to the job
  name.

- **`job_spec = "--intensity {intensity}"`**: A template for the arguments that
  follow the job name. Literal text has to match, and each `{…}` is a placeholder
  whose value slurmise captures. So `--intensity 5000` gives `intensity = 5000`. The
  template has to agree with the command you record.

- **`[slurmise.job.perfectScaler.variables]`**: Indicates what kind of value each
  parameter holds and is mandatory. `numeric` means a number
  the model can fit along. The
  other common type, `category`, is for labels like a mode or algorithm name;
  {doc}`../04_array_jobs/lesson` covers it.

- **`[slurmise.job.perfectScaler.memory]`**: `default` is the memory, in MB, that
  slurmise predicts for this job until it has enough records to fit a model. A
  `[slurmise.job.perfectScaler.runtime]` table would do the same for runtime.

### The sbatch script to record

`run_perfectScaler.sbatch` runs `perfectScaler` once:

```{literalinclude} run_perfectScaler.sbatch
:language: bash
:caption:
```

```{code-cell}
:tags: [remove-cell]
cat run_perfectScaler.sbatch | grep -q 'slurmise record'
```

Notice that the perfectScaler command is basically written out twice
— once for `srun` to execute, and once as a string for `record` to parse against the
`job_spec` in the `slurmise.toml`.

`srun`, provided by the cluster, is what makes the recording possible. Inside a
job, `srun` launches the command as a *job step*: a separate piece of the job
that SLURM's accounting tracks on its own, with its own peak memory and elapsed
time. When `perfectScaler` finishes, its step is complete and SLURM has its
final usage on record. `slurmise record` runs straight after, still inside the
job, and asks SLURM (through `sacct`) for the usage of the last step.
`record` reads what SLURM measured and stores it alongside the
`intensity` it parsed from the command.

Think of it as using the standard `srun` to record the resource usage into
SLURM's standard database, and then using `slurmise record` to get that
information AND annotate it with some details about the command.

## Run it

::::{tab-set}
:::{tab-item} On a cluster
:sync: cluster
```bash
mkdir -p out_slurm_logs
sbatch --wait run_perfectScaler.sbatch
```

`--wait` blocks until the job finishes, so there's no polling loop to write — the
command simply doesn't return until the job has run.
:::

:::{tab-item} Mocked
:sync: mock
```bash
bash mock_perfectScaler.sh
```

Nothing is submitted. The mock calls `slurmise raw-record`, which tells slurmise
directly what the job would have used:

```{literalinclude} mock_perfectScaler.sh
:language: bash
:caption:
```
:::
::::

```{code-cell}
:tags: [remove-cell]
bash mock_perfectScaler.sh
```

## Inspect

We should now have a this job storead in the `slurmise.h5` database.
You can print the contents of the slurmise database with:

```{code-cell}
slurmise --toml slurmise.toml print
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print | grep -q intensity
```

One record for `perfectScaler` with intensity 5000, alongside the memory and
runtime it used. Again, note the runtime is `0` minutes because the job is unrealistically
short. If you ran the mock, your memory usage will be exactly 5000 Mb.
If you ran `sbatch`, however, the job actually ran on SLURM and
the memory is measured so your numbers won't match those exactly.
Either way, the `intensity` value will always be 5000 since this is
a parameter parsed out of `slurmise record` command.

## Predict

Now ask slurmise what it would predict for a *different* intensity:

```{code-cell}
slurmise predict "perfectScaler --intensity 4000"
```

```{code-cell}
:tags: [remove-cell]
slurmise predict "perfectScaler --intensity 4000" 2>&1 \
    | grep -q 'No model has been fit'
```

Unfortunately, a warning that no model has been fit.

slurmise only predicts from a model once it has been fit, and fitting needs multiple
runs per job (at least 13 jobs currently), so slurmise falls back to the
defaults. The `2000` Mb is the memory `default` from `slurmise.toml`.
We haven't set a default runtime in the `slurmise.toml`, so the
predicted runtime is slurmise's own built-in default of 60 minutes.

Those job-agnostic built-in defaults are arbitrary. It is
good practice to set both defaults in your toml so the fallback guess is at least
in the right ballpark.

## Pointing at the config

Apart from `print`, none of the commands above said which TOML config to use.
This is because slurmise looks for `slurmise.toml` in the current directory, and
failing that for `~/.slurmise/slurmise.toml`. `--toml` (or `-t`) names one
explicitly, which you'll want when the config lives somewhere else. Here it
gives the same answer:

```{code-cell}
slurmise --toml slurmise.toml predict "perfectScaler --intensity 4000"
```

```{code-cell}
:tags: [remove-cell]
diff <(slurmise predict "perfectScaler --intensity 4000" 2>&1) \
     <(slurmise --toml slurmise.toml predict "perfectScaler --intensity 4000" 2>&1)
```

Next: {doc}`../02_multiple_jobs/lesson`, where we generate enough records to train
a model and get good predictions.
