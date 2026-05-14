#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00

step=0
for intensity in 2000 4000; do
    for duration in 30 60; do
        srun job1 --intensity "$intensity" --duration "$duration"
        slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" --step-id "$step" "job1 --intensity $intensity --duration $duration"
        step=$((step + 1))
    done
done
