#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=perfectScaler_loop
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Start from a clean database so this example's output stands alone
# (lazy_recording/ holds two sbatch files that share ./slurmise.h5).
rm -f slurmise.h5

# Specify 13 runs on the perfect scaler.
#
#       Runs:  1   2     3    4    5    6    7    8    9   10   11   12   13
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(    2    4    8    8    4   10   12   14    2   13   19    5    9)

# Launch perfectScaler srun jobs in a loop. First job will get intensity=500, duration=2
# Each iteration runs two srun steps (eager then lazy), so the step counter
# advances by 2 per iteration.
step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    # srun the command and then record (eager)
    # --step-id is required to target the right step's metrics AND
    # keeps the 13 database keys (<jobid>.<step>) unique
    srun ../bin/perfectScaler --intensity "$intensity" --duration "$duration"

    ./slrmise --toml slurmise.toml --step-id "$step" record \
        "perfectScaler --intensity $intensity --duration $duration"

    step=$((step + 1))

    # srun the wrapped command, which records just a stub (lazy)
    # No need to provide --step-id since slrmise has access to SLURM_STEP_ID here
    srun ./slrmise --toml slurmise.toml lazy-record -- \
        ../bin/perfectScaler --intensity "$intensity" --duration "$duration"

    step=$((step + 1))
done
