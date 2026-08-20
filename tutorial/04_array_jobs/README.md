# Arrays for parallel recording, and a categorical feature

## 01 — about this tutorial

`02_jobs_in_loop/` collected its runs one after another inside a single
allocation. Here we collect them in parallel with a SLURM array, and then look
at what changes when a job's feature isn't a number.

**This lesson always uses the mock scripts**, unlike the others, which offer you
the choice. Its three arrays come to 112 tasks, and each `sbatch --wait` waits
for every task in its array — on a small or busy cluster that is a long wait for
a lesson whose subject is what the *records* look like afterwards, not what
SLURM does to produce them.

The `.sbatch` files are here all the same, and each block below shows the
submission it stands in for, so you can run the real thing whenever you want.
Those commands are examples: nothing in this lesson will submit for you.

`00_introduction/` covers how the lessons work and what the `#>` lines mean.

## 02 — arrays, and why no `--step-id`

```bash
$ cat run_perfectScaler.sbatch
#> expect /SBATCH --array/
```

`--array=0-25` asks for 26 tasks. Each one picks an intensity out of the list
with its own `$SLURM_ARRAY_TASK_ID`, runs `perfectScaler` once, and records it.
Only 8 intensities are on offer, so the modulo means each gets run three or four
times.

Notice what's *not* there: no `--step-id`. In `02_jobs_in_loop/` all 39 runs were
steps of one job and shared a `$SLURM_JOB_ID`, so slurmise needed the step
counter to keep the records apart. An array task is a job in its own right, with
its own id, so there is nothing to disambiguate.

That's the trade. The loop needs one allocation and runs serially; the array
needs 26 and runs as wide as the queue lets it.

## 03 — the numeric arrays

`--wait` on an array waits for the whole array, not just the first task:

```bash
$ bash mock_perfectScaler.sh
#> expect ok
$ bash mock_complexMemScaler.sh
#> expect ok
```

To run them for real instead, submit the `.sbatch` files the mocks stand in for.
Nothing here will do it for you — copy these if you want them:

    sbatch --wait run_perfectScaler.sbatch
    sbatch --wait run_complexMemScaler.sbatch

```bash
$ slurmise --toml slurmise.toml print
#> expect /complexMemScaler/
```

26 records for each job — comfortably past the ~13 completed runs slurmise wants
before it will predict from a model rather than from the toml's defaults.

## 04 — predict

As in `02_jobs_in_loop/`, slurmise keeps one fitted model per `base_dir`, so
clear it and fit the job you're about to ask about:

```bash
$ rm -f fits.json *.pkl
$ slurmise --toml slurmise.toml update-model \
    "perfectScaler --intensity 2750 --duration 10"
#> expect ok
$ grep -o '"job_name": "[^"]*"' fits.json
#> expect /perfectScaler/
$ slurmise --toml slurmise.toml predict \
    "perfectScaler --intensity 2750 --duration 10"
#> expect /Predicted memory/
```

2750 was never run — it sits between the 2500 and 3000 that were — and the
memory prediction should land near it anyway. Every task here ran with the same
`--duration 10`, so there's no variation for a runtime model to learn from;
memory is the interesting one.

```bash
$ rm -f fits.json *.pkl
$ slurmise --toml slurmise.toml update-model \
    "complexMemScaler --intensity 2750 --duration 10"
#> expect ok
$ grep -o '"job_name": "[^"]*"' fits.json
#> expect /complexMemScaler/
$ slurmise --toml slurmise.toml predict \
    "complexMemScaler --intensity 2750 --duration 10"
#> expect /Predicted memory/
```

Higher, when you get a model answer at all: `../bin/complexMemScaler` adds a
flat 1000 MB on top of the intensity you ask for, and jitters it by ±20% on the
way. That noise sometimes pushes the fit's error past 20%, and slurmise then
returns `default_mem` with a warning instead — `03_noisy_job/` section 05
covers why. The `grep` above is worth keeping in the habit either way: it's how
you confirm the model answering your question belongs to the job you asked
about.

## 05 — a feature that isn't a number

`../bin/categoricalScaler` takes a `--scaling` of `linear`, `quadratic`, or
`cubic`, and raises `--intensity` to that power to decide how much memory to
allocate. So the same `--intensity 20` means 20 MB, 400 MB, or 8000 MB depending
on a *word*, not a number.

The toml declares that word as a category:

```bash
$ cat slurmise.toml
#> expect /scaling:category/
```

`{scaling:category}` is the only new thing here — `{intensity:numeric}` and
`{duration:numeric}` are the same as ever. That one word changes how slurmise
stores the job's history, as the next command shows. The array runs 60 tasks,
20 at each of the three `--scaling` values:

```bash
$ cat run_categoryScaler.sbatch
#> expect /scaling_categories/
```

```bash
$ bash mock_categoryScaler.sh
#> expect ok
```

Again, the real submission if you want it:

    sbatch --wait run_categoryScaler.sbatch

```bash
$ slurmise --toml slurmise.toml print
#> expect /scaling=cubic/
```

Look at how `print` lays those out. The numeric jobs listed their records
straight under the job name; this one groups them by category first:

    categoricalScaler
    |--- scaling=linear
    |    |--- ...
    |--- scaling=quadratic
    |--- scaling=cubic

**A category splits the job's history into separate piles**, and each pile is
fitted separately. That's the substantive difference from a numeric feature. A
model can interpolate `--intensity 2750` from runs at 2500 and 3000; there is no
such thing as halfway between `linear` and `cubic`, so slurmise doesn't try —
it keeps a model per category instead.

The practical consequence is the ~13-run threshold applies to *each* pile. The
60 tasks above are 20 per `--scaling` value, which clears it three times over —
but a fourth category added tomorrow would start again from nothing, however
much history the other three have.

So fit and ask one category at a time, the same discipline as the two jobs
earlier:

```bash
$ rm -f fits.json *.pkl
$ slurmise --toml slurmise.toml update-model \
    "categoricalScaler --intensity 20 --duration 10 --scaling linear"
#> expect ok
$ slurmise --toml slurmise.toml predict \
    "categoricalScaler --intensity 20 --duration 10 --scaling linear"
#> expect /Predicted memory: [1-9]\d?\.\d/
```

About 20 MB — `--scaling linear` means intensity to the first power. Now the
same question with the other end of the scale:

```bash
$ rm -f fits.json *.pkl
$ slurmise --toml slurmise.toml update-model \
    "categoricalScaler --intensity 20 --duration 10 --scaling cubic"
#> expect ok
$ slurmise --toml slurmise.toml predict \
    "categoricalScaler --intensity 20 --duration 10 --scaling cubic"
#> expect /Predicted memory: \d\d\d\d/
```

About 8000 MB — 20³. Same job, same `--intensity`, a four-hundred-fold
difference in what it needs, and slurmise has it because it never mixed the two
histories together.

Try it without the `rm` and you'll get the previous category's answer to the new
category's question: one fitted model lives in this directory at a time, and it
does not check whether it was fitted for what you're asking about.

## Starting over

The lesson only holds from a clean slate — otherwise the fits pick up runs from
a previous pass as well as this one. `tutorial.py` runs exactly this block
for you, every time, before the lesson starts:

```bash
#> reset
$ rm -f slurmise.h5 fits.json *.pkl
$ rm -f out_slurm_logs/*.out
$ mkdir -p out_slurm_logs
```
