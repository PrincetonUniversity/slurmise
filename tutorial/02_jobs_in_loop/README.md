# 02 - Jobs in a Loop

Runs multiple parameter combinations sequentially within a single SLURM job.
Each iteration creates a separate SLURM step via `srun`, and `--step-id` is
used to tell slurmise which step to record.

## Run

```bash
sbatch run_many_job1s.sbatch
```

## What happens

1. A loop iterates over combinations of intensity and duration
2. Each iteration runs `srun job1` (creating a new SLURM step) then calls
   `slurmise record` with `--step-id` to record that specific step's stats

## Check results

```bash
slurmise --toml slurmise.toml print
```
