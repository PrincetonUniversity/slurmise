#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --array=0-59
#SBATCH --job-name=categoricalScaler
#SBATCH --output=slurm_outs/slurm-%x-%A_%a.out

scaling_categories=(linear quadratic cubic)

idx=$((SLURM_ARRAY_TASK_ID % 3))
scaling=${scaling_categories[$idx]}

srun ../bin/categoricalScaler --intensity 20 --duration 10 --scaling $scaling

slurmise --toml slurmise.toml record \
    --slurm-id "$SLURM_JOB_ID" \
    "categoricalScaler --intensity 20 --duration 10 --scaling $scaling"
