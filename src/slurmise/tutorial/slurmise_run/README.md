# Proposal: slurmise does sbatch for you

This was motivated by looking at the slurmise tutorial and not liking how
you have to specify the job parameters twice:

```bash
srun ../bin/perfectScaler --intensity 5000 --duration 10

slurmise --toml slurmise.toml record \
    "perfectScaler --intensity 5000 --duration 10"
```

Talking with Troy, we ended up thinking about this single-command
form for slurmise where it acts as a wrapper like `time`:

```bash
./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler --intensity 5000 --duration 10
```

That one `slrmise run` does everything in order:

1. **syncs** the slurmise h5 database, filling time/mem from `sacct` for prior
   jobs that were previously submitted but not finished running
2. **fits** a model on any newly-completed runs;
3. **predicts** time/mem for this command;
4. **submits** it via `sbatch --mem=<pred> --time=<pred> --wrap "<command>"`;
5. **records** for the new job id (features + predicted resources) — its
   actual time/mem are unknown until it runs, so they start as `None` and get
   filled by a later sync.

`./slrmise` is a prototype standing in for a real `slurmise`: a small Python
script that imports and uses the `slurmise` code unmodified but exposes this new
CLI.


## Tutorial for this new approach

[`tutorial.md`](tutorial.md) is a follow-along tutorial showing
how the `./slrmise` prototype works. You can read it and type the
commands yourself, or use `tutorial.py` to drive those same commands for you:

```bash
./tutorial.py
```

By default this submits nothing: each lesson tells `slrmise` what its job would
have used, and the result is written straight to the slurmise database. That
avoids queue times entirely, so the whole tour takes under a minute and needs no
cluster — but it doesn't exercise `sbatch` or `sacct`. Use `./tutorial.py
--slurm` to submit real jobs instead.

## The subcommands

```bash
# predict only — print the raw estimate, submit nothing
./slrmise --toml slurmise.toml predict -- \
    ../bin/perfectScaler --intensity 5000 --duration 10

# run — sync, fit, predict, submit, record a stub (optional --margin, --dry-run)
./slrmise --toml slurmise.toml run --margin 2.0 -- \
    ../bin/perfectScaler --intensity 5000 --duration 10

# display — sync pending rows, then print predicted vs actual side by side
./slrmise --toml slurmise.toml display          # --no-sync to see stubs as-is
```

## Scope & limitations

`slrmise` predicts two scalars (memory, time), which fully describe
single-node, single-task (possibly multi-threaded) commands — the common case.

Recording `srun` steps in a job and MPI jobs are not currently handled by `./slrmise`
and may not be easy to support.
