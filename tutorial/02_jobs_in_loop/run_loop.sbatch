#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=scaler_loop

# Specify 13 runs on the perfect and complex scaler.
#
#       Runs:  1   2     3    4    5    6    7    8    9   10   11   12   13
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(    2    4    8    8    4   10   12   14    2   13   19    5    9)

# Launch perfectScaler srun jobs in a loop. First job will get intensity=500, duration=2
step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    # srun the command
    srun ../bin/perfectScaler --intensity "$intensity" --duration "$duration"

    # record the results keeping track of the --step-id
    slurmise --toml slurmise.toml record \
        --slurm-id "$SLURM_JOB_ID" \
        --step-id "$step" \
        "perfectScaler --intensity $intensity --duration $duration"

    # increment the step
    step=$((step + 1))

done

# Launch the same jobs but use the complexMemScaler command instead
step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    srun ../bin/complexMemScaler --intensity "$intensity" --duration "$duration"
    slurmise --toml slurmise.toml record --slurm-id "$SLURM_JOB_ID" \
        --step-id "$step" \
        "complexMemScaler --intensity $intensity --duration $duration"
    step=$((step + 1))
done
