# Setting up the tutorial

These lessons walk through slurmise on a few toy jobs, from recording a single run
to predicting resources across an array. You'll need slurmise installed first —
see {doc}`install`.

## Getting the tutorial files

The tutorial provides self-contained lessons with important files such as a
`slurmise.toml` config and a couple of toy programs that use a known amount of
time and memory. Those files ship as a tarball attached to each release:

```bash
wget -qO- https://github.com/PrincetonUniversity/slurmise/releases/latest/download/slurmise-tutorial.tar.gz | tar -xz
cd slurmise-tutorial
```

## A note on the tutorial's example commands

The example commands we've provided, such as `perfectScaler`, are toy examples
that only run for about a second so you don't
have to twiddle your thumbs waiting for results as you work through the tutorial.
This is not representative of normal SLURM jobs that often
run for hours or even days. In fact, if you're frequently submitting SLURM jobs that finish
in a few seconds, your HPC admin might have a few choice words for you.

These short runtimes cause a tutorial-specific problem that all slurmise
predicted runtimes are going to be `0` minutes and you'll get warnings like:

```text
Warnings:
  Predicted runtime for job perfectScaler is zero or negative (0).
```

**Therefore, the lessons in the tutorial focus exclusively on using slurmise
to predict memory requirements of the commands.**

In real slurmise usage you'll be able to predict both time and memory requirements.

## Lessons

Each lesson in this book corresponds to a directory in the tutorial tarball.
We recommend taking the lessons in order as each builds on the one before.

To follow along, open a shell, `cd` into the lesson's directory, and type/copy the
commands as you need them.

Your numbers might not match the page exactly as measured memory varies from run to run.

## Running the lessons without a cluster

You can follow along with this tutorial on your local laptop.
Every lesson that submits a job offers two ways to do so:

- **cluster** — a real `sbatch`, which is what you would do in earnest.
- **mock** — a stand-in that calls `slurmise raw-record` which asserts what the job
  *would* have used instead of actually submitting.

Both approaches result in the slurmise database recording similar information.

## Running the lessons with a cluster

Even if you choose to follow the tutorial on a SLURM-enabled HPC, you may still
want to opt for the **mock** approach to skip waiting for the queue to schedule
the jobs.

If you do want to submit jobs, your SLURM setup may require extra flags such as
`--account` when submitting the included `.sbatch` files. `sbatch` honors
`SBATCH_ACCOUNT`, `SBATCH_PARTITION` and `SBATCH_QOS` from your environment,
which is the easiest way to supply them without editing every file:

```bash
export SBATCH_ACCOUNT=myaccount
```

Now start with {doc}`lessons/01_single_job/lesson` when you're ready.
