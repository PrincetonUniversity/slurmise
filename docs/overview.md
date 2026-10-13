# Overview

## Is slurmise a good fit?

slurmise might be a good fit if you need to run the same code thousands of times
with different parameters or inputs. It won't be helpful if you just need to run
a script twice, because it won't see enough examples to make a helpful
prediction.

For example, you might be a biologist who has a script that predicts the protein
structure of a gene and you want to run that script on all ~20,000 genes. You
could submit an ARRAY job, but how much time and memory should you provide? Some
genes are small, others are huge.

Or maybe you're an astrophysicist interested in estimating the number of black
holes across different regions of space.

## Workflow when using slurmise

First specify your command of interest in a `slurmise.toml` configuration file.
This file includes the command name, how it is run, and parameters and their types.

slurmise predicts the time and memory a SLURM job will need by learning from
jobs you have already run. The workflow is a loop:

1. **Run.** launch your command, for example `srun myCommand --paramA 12`
2. **Record.** After a job finishes, `slurmise record` reads the job's resource
   usage from SLURM's accounting and stores it in a local HDF5 database, tagged
   with the parameters that job ran with — the input size, a mode flag, whatever
   you think matters.
3. **Update.** `slurmise update-model` fits a
   model to the recorded data for each job.
4. **Predict.** `slurmise predict` uses the fitted model to estimate runtime and
   memory for a new set of parameters, so you can request sensible resources
   before submitting.
