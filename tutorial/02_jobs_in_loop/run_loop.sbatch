#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=scaler_loop

intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(2 4 8 8 4 10 12 14 2 13 19 5 9)

step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

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
