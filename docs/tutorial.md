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

That gives you one directory per lesson.

## A note on the tutorial's numbers

The lessons use short sleep durations so every job finishes quickly. slurmise
records runtime in whole minutes, so when you run them on a cluster, a job that
sleeps for seconds is stored as `0` and every fitted runtime comes out flat.
slurmise notices that the fit is degenerate rather than trusting it, and says so:

```text
Warnings:
  Predicted runtime for job perfectScaler is zero or negative (0).
```

The prediction then falls back to the runtime `default` from the toml. That is
worth seeing once — it is the same guard that protects you from a bad fit on real
jobs — but it does mean these lessons teach predicting **memory**.

## Lessons

Each lesson in this book corresponds to a directory in the tutorial tarball. Take
the lessons in order; each builds on the one before.

To follow along, open a shell, `cd` into the lesson's directory, and type/copy the
commands as you need them.

Your numbers might not match the page exactly as measured memory and runtime vary
from run to run.

## Running the lessons without a cluster

Every lesson that submits a job offers two ways to do it, as a pair of tabs.
Picking one switches every pair on the page:

- **cluster** — a real `sbatch`, which is what you would do in earnest.
- **mock** — a stand-in that calls `slurmise raw-record` to assert what the job
  *would* have used instead of actually submitting.

Your SLURM setup may require extra flags such as `--account` when submitting the
included `.sbatch` files. `sbatch` honors `SBATCH_ACCOUNT`, `SBATCH_PARTITION`
and `SBATCH_QOS` from your environment, which is the easiest way to supply them
without editing every file:

```bash
export SBATCH_ACCOUNT=myaccount
```

Start with {doc}`lessons/01_single_job/lesson` when you're ready.
