#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00

intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
duration=10

step=0
for intensity in "${intensities[@]}"; do
    srun ./perfectScaler --intensity "$intensity" --duration "$duration"
    slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" \
        --step-id "$step" \
        "perfectScaler --intensity $intensity --duration $duration"
    step=$((step + 1))

    srun ./complexMemScaler --intensity "$intensity" --duration "$duration"
    slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" \
        --step-id "$step" \
        "complexMemScaler --intensity $intensity --duration $duration"
    step=$((step + 1))
done
