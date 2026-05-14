#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00

srun job1 --intensity 5000 --duration 120
slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" "job1 --intensity 5000 --duration 120"
