# Proposal: slurmise owns the run

This was motivated by looking at slurmise tutorial and not liking how
we need specify the job parameters twice:

```bash
srun ../bin/perfectScaler --intensity 5000 --duration 10

slurmise --toml slurmise.toml record \
    "perfectScaler --intensity 5000 --duration 10"
```

Pulling on that thread led to having slurmise wrap the command,

```bash
./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler \
        --intensity 5000 \
        --duration 10
```

That one slrmise command does the following:
1. syncs the slurmise h5 database to collect time/mem results from prior jobs
2. predicts time/mem for the current wrapped command
3. submits the command with subprocess via `sbatch --mem=<pred> --time=<pred>`
4. Records a stub for the new job now that it has the `SLURM_ID`
   --> At this point we know the job name, slurmid, run params, but NOT the used time/mem
       since the job will still be running, or even just queued
   --> So we add an entry into the slurmise h5 database with None's for time/mem
   --> We can always fill these in later since we already know the slurmid

To test whether this is possible, I've made the `./slrmise` prototype as a
stand-in for the real `slurmise`. It's a small python script that imports and
uses the `slurmise` code without modifications, but has a new CLI.

**Scope.** slrmise predicts two scalars (memory, time), which fully describe
single-node, single-task (possibly multi-threaded) commands -- the common
case. MPI/multi-node geometry (`--ntasks`, `--nodes`, scaling curves) is not
currently supported in this prototype, and might be hard to support.


## `./slrmise run` -- the main entrypoint

```bash
./slrmise --toml slurmise.toml run -- <command...>
# with a custom safety margin:
./slrmise --toml slurmise.toml run --margin 2.0 -- ../bin/perfectScaler --intensity 5000 --duration 10
```

`run` does everything in order: sync prior runs from sacct, refit the model on
the completed runs, predict this command's resources,
sbatch it, and record a stub.


## `./slrmise predict` -- just print the predicted resources

```bash
./slrmise --toml slurmise.toml predict -- ../bin/perfectScaler --intensity 5000 --duration 10
```

Reports the raw model estimate (memory MB, time minutes) and submits nothing.

## `./slrmise display` -- print the slurmise h5 database

Syncs pending rows, then prints the database with predicted and actual side by
side, so every submitted job scores the model that submitted it. `--no-sync`
prints as-is, useful for seeing stubs while jobs are still queued/running
(their metrics show as `-` until the job is terminal).

## Limitations

The following are not supported:

- **Step-level recording for bundled jobs.** When queue economics push you to
  bundle many commands into one job, job-level accounting can't attribute
  usage per command -- but SLURM steps can. A `record` primitive run under
  `srun` inside a hand-written sbatch script could stub a row keyed
  `<jobid>.<stepid>` (from `SLURM_STEP_ID`) and `exec` the command, feeding
  per-command samples into the same models. Left out here to keep the
  prototype centered on `run`; easy to add back if the maintainers want it.
- **MPI/multi-node geometry.** Predicting a resource *shape* (`--ntasks`,
  `--nodes`, scaling curves) rather than two scalars.
