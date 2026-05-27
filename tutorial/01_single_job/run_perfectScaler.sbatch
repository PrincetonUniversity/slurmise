#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=perfectScaler

srun ../bin/perfectScaler --intensity 5000 --duration 10

slurmise --toml slurmise.toml record \
    --slurm-id "$SLURM_JOB_ID" \
    "perfectScaler --intensity 5000 --duration 10"
