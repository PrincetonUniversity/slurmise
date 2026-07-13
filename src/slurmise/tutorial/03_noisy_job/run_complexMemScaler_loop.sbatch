#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=complexMemScaler_loop
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Specify 13 run parameters on the complex scaler.
#
#       Runs:  1   2     3    4    5    6    7    8    9   10   11   12   13
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(    2    4    8    8    4   10   12   14    2   13   19    5    9)

replicates=3

# Launch complexMemScaler srun jobs in a loop. First job will get intensity=500, duration=2
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    # Run multiple replicates of the same same --intensity and --duration job
    for (( j=0; j<replicates; j++ )); do
        # srun the command
        srun ../bin/complexMemScaler --intensity "$intensity" --duration "$duration"

        # record the results
        slurmise --toml slurmise.toml record \
            "complexMemScaler --intensity $intensity --duration $duration"
    done

done
