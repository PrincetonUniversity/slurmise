# 01 - Single Job

Records a single job that allocates 5000 MB of memory and sleeps for 120 seconds.

## Run

```bash
sbatch run_job1.sbatch
```

## What happens

1. `srun` runs `job1` with the specified intensity and duration
2. After the job completes, `slurmise record` queries SLURM for the job's
   runtime and peak memory, then stores them alongside the parsed parameters

## Check results

```bash
slurmise --toml slurmise.toml print
```
