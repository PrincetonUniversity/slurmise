#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=perfectScaler
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

srun ../bin/perfectScaler --intensity 5000 --duration 10

slurmise --toml slurmise.toml record \
    "perfectScaler --intensity 5000 --duration 10"

# Show what was recorded (visible in this job's slurm .out)
slurmise --toml slurmise.toml print
