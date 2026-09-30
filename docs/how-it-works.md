# How slurmise works

slurmise predicts the time and memory a SLURM job will need by learning from
jobs you have already run. The workflow is a loop:

0. **Run.** Use a standard `srun` to launch your command, for example
   `srun myCommand --param1 12`. The tutorial shows how to launch `srun`s from
   inside `.sbatch` files.
1. **Record.** After a job finishes, `slurmise record` reads the job's resource
   usage from SLURM's accounting and stores it in a local HDF5 database, tagged
   with the parameters that job ran with — the input size, a mode flag, whatever
   you think matters.
2. **Update.** `slurmise update-all` (or `update-model` for a single job) fits a
   model to the recorded data for each job.
3. **Predict.** `slurmise predict` uses the fitted model to estimate runtime and
   memory for a new set of parameters, so you can request sensible resources
   before submitting.

A few things that shape what you will see in the tutorial:

- **Configuration lives in a toml file.** Every command takes `--toml`; there is
  no auto-discovery. The toml declares each job's `job_spec` and any default
  time and memory values.
- **Parameters can be numeric or categorical.** Numeric parameters let the model
  interpolate across nearby values. Categorical parameters partition the
  database into a separate model per category combination, so each combination
  needs its own data.
- **A model needs enough data before it will fit.** Until a job has accumulated
  enough records, `predict` falls back to the defaults from your toml rather
  than guessing from too few points.

## A note on the tutorial's numbers

The lessons use short sleep durations so every job finishes quickly. slurmise
records runtime in whole minutes, so a job that sleeps for seconds is stored as
`0` and every fitted runtime comes out flat. slurmise notices that the fit is
degenerate rather than trusting it, and says so:

```text
Warnings:
  Predicted runtime for job perfectScaler is zero or negative: 0.0
  Returning default runtime value.
```

The prediction then falls back to `default_time` from the toml. That is worth
seeing once — it is the same guard that protects you from a bad fit on real
jobs — but it does mean these lessons teach predicting **memory**.

## Running the lessons without a cluster

Every lesson that submits a job offers two ways to do it:

- **cluster** — a real `sbatch`, which is what you would do in earnest.
- **mock** — a stand-in that calls `slurmise raw-record` to assert what the job
  *would* have used, so nothing is submitted and nothing is queued.

The lesson pages in this book show the mock path, because that is the path that
runs when the documentation is built — there is no scheduler in CI. Each lesson
also shows the `sbatch` command it stands in for, so you can run the real thing.
Mocking is a convenience for learning; in actual use you always want the
`sbatch` path.

Your SLURM setup may require extra flags such as `--account` when submitting the
included `.sbatch` files. `sbatch` honours `SBATCH_ACCOUNT`, `SBATCH_PARTITION`
and `SBATCH_QOS` from your environment, which is the easiest way to supply them
without editing every file:

```bash
export SBATCH_ACCOUNT=myaccount
```
