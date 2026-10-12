# How slurmise works

## Why slurmise

Knowing how much time and memory to assign a job is very difficult. Even if
you're just running the same command a thousand times, the runtime might vary
depending on what parameters are used.

For example, you might be a biologist who has a script that predicts the protein
structure of a gene and you want to run that script on all ~20,000 genes. You
could submit an ARRAY job, but how much time and memory should you provide? Some
genes are small, others are huge.

Or maybe you're an astrophysicist interested in estimating the number of black
holes across different regions of space.

slurmise aims to help you request an efficient amount of time and memory when
you submit jobs to the SLURM scheduler.

slurmise might be a good fit if you need to run the same code thousands of times
with different parameters or inputs. It won't be helpful if you just need to run
a script twice, because it won't see enough examples to make a helpful
prediction.

Why should you care about being efficient with SLURM resources? The less
resources you ask for, the shorter your queue times.

## The record / update / predict loop

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

- **Configuration lives in a toml file.** Pass it with `--toml` (or `-t`).
  Without it, slurmise looks for `slurmise.toml` in the current directory, then
  for `~/.slurmise/slurmise.toml`. The toml declares each job's `job_spec` and
  any default time and memory values.
- **Parameters can be numeric or categorical.** Numeric parameters let the model
  interpolate across nearby values. Categorical parameters partition the
  database into a separate model per category combination, so each combination
  needs its own data.
- **Until a model is fit, `predict` returns your defaults.** `predict` only
  answers from a model once `update-all` or `update-model` has fit one, and
  fitting needs enough records for that job. Until then it returns the defaults
  from your toml rather than guessing from too few points.
