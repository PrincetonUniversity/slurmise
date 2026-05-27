#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --array=0-8
#SBATCH --job-name=categorical_array

int_cats=(1 1 1 2 2 2 3 3 3)
dur_cats=(1 2 3 1 2 3 1 2 3)

int_cat=${int_cats[$SLURM_ARRAY_TASK_ID]}
dur_cat=${dur_cats[$SLURM_ARRAY_TASK_ID]}

srun ../bin/categoricalScaler --intensity "$int_cat" --duration "$dur_cat"
slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" \
    "categoricalScaler --intensity $int_cat --duration $dur_cat"
