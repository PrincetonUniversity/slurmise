#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --array=0-15
#SBATCH --job-name=scaler_array

intensities=(500 1000 1500 2000 2500 3000 3500 4000)
duration=10

idx=$((SLURM_ARRAY_TASK_ID % 8))
intensity=${intensities[$idx]}

if (( SLURM_ARRAY_TASK_ID < 8 )); then
    job=perfectScaler
else
    job=complexMemScaler
fi

srun ../bin/$job --intensity "$intensity" --duration "$duration"
slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" \
    "$job --intensity $intensity --duration $duration"
