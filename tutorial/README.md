# Slurmise Tutorial

This tutorial walks through using slurmise to track SLURM job resource usage.
Each subfolder is a self-contained example that can be run independently.

## Prerequisites

- Access to a SLURM-enabled cluster
- slurmise installed (`pip install slurmise`)

## Examples

- [01_single_job](01_single_job/) - Record a single job's resource usage
- [02_jobs_in_loop](02_jobs_in_loop/) - Record multiple jobs run sequentially in a loop
- [03_array_jobs](03_array_jobs/) - Record jobs run in parallel using a SLURM array

## The test job

Each example uses `job1`, a simple bash script that allocates a specified amount of
memory and sleeps for a specified duration. This lets you verify that slurmise
correctly tracks resource usage with known values.

## Usage

`cd` into any example directory and submit with:

```bash
sbatch <script>.sbatch
```

Note that everyone's SLURM setup is different so you might need to add or change
some of the slurm params. Personally I need to add `--account` to the `sbatch` submission.

After the job completes, check the recorded data with:

```bash
slurmise --toml slurmise.toml print
```
