# A job whose memory doesn't sit still

## 01 — about this tutorial

`02_jobs_in_loop/` fitted a model to `perfectScaler`, which uses exactly the
memory you ask it for. Real jobs aren't that obliging. This lesson runs the same
39-run loop against `complexMemScaler`, whose memory moves from run to run, and
looks at what slurmise does when the fit isn't good enough to trust.

**This lesson takes about 10 minutes**, nearly all of it the job sleeping.

`00_introduction/` covers how the lessons work and what the `#>` lines mean.

## 02 — the noisy job

`../bin/complexMemScaler` takes the same `--intensity` and `--duration` as
`perfectScaler`, and the sbatch walks the same 13 pairs × 3 replicates. What
differs is inside the script: it allocates the intensity you asked for, jitters
it by ±20%, **and adds a flat 1000 MB on top**.

```bash
$ cat run_complexMemScaler_loop.sbatch
#> expect /complexMemScaler/
```

So two things are true at once, and it's worth keeping them apart. The flat
1000 MB is a real, learnable offset — this job simply needs more memory than
`perfectScaler` at the same arguments, and a model can pick that up. The ±20%
jitter is noise, and no model can predict it; the best it can do is fit through
the middle and be wrong by up to a fifth either way.

The toml is `02_jobs_in_loop/`'s with the job name changed:

```bash
$ cat slurmise.toml
#> expect /complexMemScaler/
```

## 03 — submit it

```bash
#> option cluster
$ sbatch --wait run_complexMemScaler_loop.sbatch
#> expect ok
#> option mock
$ bash mock_complexMemScaler_loop.sh
#> expect ok
```

## 04 — inspect

```bash
$ slurmise --toml slurmise.toml print
#> expect /complexMemScaler/
```

39 records again, sharing one job id and separated by `--step-id`. Look down the
`memory` values for the replicates of a single intensity — in
`02_jobs_in_loop/` those were near-identical; here they scatter.

## 05 — train, and predict

```bash
$ slurmise --toml slurmise.toml update-all
#> expect ok
$ slurmise --toml slurmise.toml predict \
    "complexMemScaler --intensity 2750 --duration 7"
#> expect /Predicted memory/
```

You'll get one of two things here, and which one depends on how the noise fell
in your 39 runs. Either a model-based number — noticeably larger than
`02_jobs_in_loop/`'s answer to the same question, which is the flat 1000 MB
showing up — or this:

    Predicted memory: 5000

    Warnings:
      Memory prediction for job complexMemScaler is not within 20% of actual value.
      Returing default memory value.

5000 is `default_mem` from the toml. slurmise scored the fitted model against
the runs it held back, found it missed by more than 20%, and declined to use it.
Across several passes while writing this lesson the error came out at 15.6%,
17.7% and 22.4% — the same command, genuinely falling on both sides of that
line, because the ±20% jitter lands differently every time.

**That is the guard rail worth taking away.** slurmise would rather hand back a
default it has warned you about than a model estimate it can't stand behind.
`perfectScaler` never trips it; a job with real-world variance sometimes will,
and when it does the honest answer is the one you got.

If your own job behaves like this, more runs are what help — not a better
model. The jitter doesn't shrink, but a larger sample makes the fit's estimate
of the middle steadier, and the held-out error less of a lottery.

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
