# Quickstart

Run `./tutorial.py` to view and select a lesson

# Slurmise Tutorial

This tutorial walks through using slurmise to track SLURM job resource usage
and predict requirements for future jobs. Each subfolder is a self-contained
lesson with its own database, so they can be taken in any order or one at a
time.

You likely already have `slurmise` installed and ran `slurmise-generate-tutorial`
to access this file.

## Prerequisites

- Access to a SLURM-enabled cluster (with `sbatch` and `srun`)
- slurmise installed (`pip install slurmise`)

## Usage

```bash
./tutorial.py                    # pick a lesson from the menu and walk it
./tutorial.py 02_jobs_in_loop    # skip the menu
./tutorial.py --yes              # every lesson, in order, without pausing
./tutorial.py --option mock      # take the no-cluster path at every choice
```

`tutorial.py` shows each command before running it, waits for you to press
Enter, and checks the command did what the lesson says it should. Every lesson
starts by clearing its own database, so nothing carries over from a previous
pass.

You don't have to use it. Each lesson's `README.md` is the tutorial — the
same prose and the same commands — so you can equally `cd` into a lesson folder
and type the `$` lines yourself. The `#>` lines are what `tutorial.py` checks
against; they're shell comments, so they're inert if you copy a block.

### Trying it without a cluster

Wherever a lesson submits a job, its block offers two ways to do it:

```bash
#> option cluster
$ sbatch --wait run_perfectScaler.sbatch
#> expect ok
#> option mock
$ bash mock_perfectScaler.sh
#> expect ok
```

`cluster` really submits. `mock` runs a script shipped beside the `.sbatch`
that states what the job would have used — via `slurmise raw-record --memory
--runtime`, which takes those numbers at face value instead of asking `sacct` —
so nothing is submitted and nothing sleeps. Either leaves the lesson in the same
state, so the rest of it reads the same way, and the whole tutorial takes a
couple of minutes instead of the better part of an hour.

Working by hand, type whichever you want. `tutorial.py` asks; `--option mock`
or `--option cluster` answers for you, and `--yes` alone takes the first.

The mock scripts are kept separate from the `.sbatch` files on purpose. The
`.sbatch` is the thing you'll copy and adapt for your own job, and it should
read as exactly that — not as a script with a second mode folded into it.

Everything after the recording is the real thing either way: a real database,
real fits, real predictions. What `mock` can't show you is the part that needs
SLURM — `srun`, and `slurmise record` reading back from `sacct` what a job
actually used.

`tutorial.py` runs under [uv](https://docs.astral.sh/uv/), which supplies the
one library it needs. The commands it runs use whatever `slurmise` is on your
PATH.

Your SLURM setup may require additional flags such as `--account` when you're
submitting the included sbatch files. `sbatch` honours `SBATCH_ACCOUNT`,
`SBATCH_PARTITION` and `SBATCH_QOS` from your environment, which is the easiest
way to supply them without editing every `.sbatch`.
