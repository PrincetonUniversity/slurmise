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
rm -f slurmise.h5
rm -rf PolynomialFit
rm -rf out_slurm_logs
```

# Arrays for parallel recording, and a categorical feature

{doc}`../02_jobs_in_loop/lesson` collected its runs one after another inside a
single allocation. Here we collect them in parallel with a SLURM array, and then
look at what changes when a job's feature isn't a number.

**This lesson always uses the mock scripts**, unlike the others, which offer a
choice. Its three arrays come to 112 tasks, and each `sbatch --wait` waits for
every task in its array — on a small or busy cluster that is a long wait for a
lesson whose subject is what the *records* look like afterwards, not what SLURM
does to produce them. The `.sbatch` files are here all the same, and each section
shows the submission it stands in for.

To follow along, `cd` into `04_array_jobs/`.

## Arrays, and why no `--step-id`

```{code-cell}
---
mystnb:
  text_lexer: bash
---
cat run_perfectScaler.sbatch
```

```{code-cell}
:tags: [remove-cell]
cat run_perfectScaler.sbatch | grep -q 'SBATCH --array'
```

`--array=0-25` asks for 26 tasks. Each one picks an intensity out of the list with
its own `$SLURM_ARRAY_TASK_ID`, runs `perfectScaler` once, and records it. Only 8
intensities are on offer, so the modulo means each gets run three or four times.

Notice what's *not* there: no `--step-id`. In lesson 02 all 39 runs were steps of
one job and shared a `$SLURM_JOB_ID`, so slurmise needed the step counter to keep
the records apart. An array task is a job in its own right, with its own id, so
there is nothing to disambiguate.

That's the trade. The loop needs one allocation and runs serially; the array needs
26 and runs as wide as the queue lets it.

## The numeric arrays

To run these for real, submit the `.sbatch` files the mocks stand in for —
`--wait` on an array waits for the whole array, not just the first task:

```bash
mkdir -p out_slurm_logs
sbatch --wait run_perfectScaler.sbatch
sbatch --wait run_complexMemScaler.sbatch
```

Mocked:

```{code-cell}
bash mock_perfectScaler.sh
bash mock_complexMemScaler.sh
```

```{code-cell}
slurmise --toml slurmise.toml print | head -20
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print | grep -q complexMemScaler
```

26 records for each job — comfortably past the ~13 completed runs slurmise wants
before it will predict from a model rather than from the toml's defaults.

## Predict

slurmise keeps one fitted model per `base_dir`, so fit the job you're about to ask
about:

```{code-cell}
slurmise --toml slurmise.toml update-model --job-name perfectScaler
grep -o '"job_name": "[^"]*"' PolynomialFit/*/fits.json
```

```{code-cell}
slurmise --toml slurmise.toml predict "perfectScaler --intensity 2750 --duration 10"
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml predict "perfectScaler --intensity 2750 --duration 10" 2>&1 \
    | grep -q 'Predicted memory'
```

2750 was never run — it sits between the 2500 and 3000 that were — and the memory
prediction should land near it anyway. Every task here ran with the same
`--duration 10`, so there's no variation for a runtime model to learn from; memory
is the interesting one.

```{code-cell}
slurmise --toml slurmise.toml update-model --job-name complexMemScaler
slurmise --toml slurmise.toml predict "complexMemScaler --intensity 2750 --duration 10"
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml predict "complexMemScaler --intensity 2750 --duration 10" 2>&1 \
    | grep -q 'Predicted memory'
```

Higher, when you get a model answer at all: `../bin/complexMemScaler` adds a flat
1000 MB on top of the intensity you ask for, and jitters it by ±20% on the way.
That noise sometimes pushes the fit's error past 20%, and slurmise then returns
`default_mem` with a warning instead — {doc}`../03_noisy_job/lesson` covers why.

## A feature that isn't a number

`../bin/categoricalScaler` takes a `--scaling` of `linear`, `quadratic`, or
`cubic`, and raises `--intensity` to that power to decide how much memory to
allocate. So the same `--intensity 20` means 20 MB, 400 MB, or 8000 MB depending
on a *word*, not a number.

The toml declares that word as a category:

```{code-cell}
---
mystnb:
  text_lexer: toml
---
cat slurmise.toml
```

```{code-cell}
:tags: [remove-cell]
cat slurmise.toml | grep -qE 'scaling.*category'
```

`{scaling:category}` is the only new thing here — `{intensity:numeric}` and
`{duration:numeric}` are the same as ever. That one word changes how slurmise
stores the job's history, as the next command shows. The array runs 60 tasks, 20 at
each of the three `--scaling` values:

```bash
sbatch --wait run_categoryScaler.sbatch
```

```{code-cell}
bash mock_categoryScaler.sh
```

```{code-cell}
slurmise --toml slurmise.toml print | grep -A3 'scaling='
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml print | grep -q 'scaling=cubic'
```

Look at how `print` lays those out. The numeric jobs listed their records straight
under the job name; this one groups them by category first:

```text
categoricalScaler
|--- scaling=linear
|    |--- ...
|--- scaling=quadratic
|--- scaling=cubic
```

**A category splits the job's history into separate piles**, and each pile is
fitted separately. That's the substantive difference from a numeric feature. A
model can interpolate `--intensity 2750` from runs at 2500 and 3000; there is no
such thing as halfway between `linear` and `cubic`, so slurmise doesn't try — it
keeps a model per category instead.

The practical consequence is that the ~13-run threshold applies to *each* pile. The
60 tasks above are 20 per `--scaling` value, which clears it three times over — but
a fourth category added tomorrow would start again from nothing, however much
history the other three have.

So fit and ask one category at a time:

```{code-cell}
slurmise --toml slurmise.toml update-model --job-name categoricalScaler
slurmise --toml slurmise.toml predict "categoricalScaler --intensity 20 --duration 10 --scaling linear"
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml predict "categoricalScaler --intensity 20 --duration 10 --scaling linear" 2>&1 \
    | grep -qE 'Predicted memory: [1-9][0-9]$'
```

About 20 MB — `--scaling linear` means intensity to the first power. Now the same
question at the other end of the scale:

```{code-cell}
slurmise --toml slurmise.toml predict "categoricalScaler --intensity 20 --duration 10 --scaling cubic"
```

```{code-cell}
:tags: [remove-cell]
slurmise --toml slurmise.toml predict "categoricalScaler --intensity 20 --duration 10 --scaling cubic" 2>&1 \
    | grep -qE 'Predicted memory: [0-9][0-9][0-9][0-9]'
```

About 8000 MB — 20³. Same job, same `--intensity`, a four-hundred-fold difference
in what it needs, and slurmise has it because it never mixed the two histories
together.
