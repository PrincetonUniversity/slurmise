#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-3

intensities=(8000 8000 10000 10000)
durations=(100 100 160 160)

intensity=${intensities[$SLURM_ARRAY_TASK_ID]}
duration=${durations[$SLURM_ARRAY_TASK_ID]}

srun job1 --intensity "$intensity" --duration "$duration"
slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" "job1 --intensity $intensity --duration $duration"
