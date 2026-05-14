# 03 - Array Jobs

Runs multiple parameter combinations in parallel using a SLURM job array.
Each array task gets its own SLURM job ID, so no `--step-id` is needed.

## Run

```bash
sbatch run_job1_array.sbatch
```

## What happens

1. SLURM launches one task per array index (0-3)
2. Each task looks up its intensity and duration from arrays using
   `$SLURM_ARRAY_TASK_ID`
3. Each task runs `srun job1` then calls `slurmise record` with its own
   `$SLURM_JOB_ID`

## Check results

```bash
slurmise --toml slurmise.toml print
```
