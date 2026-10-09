#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --array=0-25
#SBATCH --job-name=complexMemScaler
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

intensities=(500 1000 1500 2000 2500 3000 3500 4000)
duration=10

idx=$((SLURM_ARRAY_TASK_ID % 8))
intensity=${intensities[$idx]}

srun ../bin/complexMemScaler --intensity "$intensity" --duration "$duration"

slurmise --toml slurmise.toml record \
    "complexMemScaler --intensity $intensity --duration $duration"
