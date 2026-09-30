#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=srun_then_record
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Approach 1 -- eager: run the command, then tell slurmise about it.
#
# The command is written twice: once for srun to execute, once as a quoted
# string for record to parse against the job_spec. They have to match.

#          Runs:  1    2    3
intensities=(2000 3000 4000)
durations=(     5    5    5)

# record runs in the batch shell, which has no SLURM_STEP_ID, so it has to be
# told which step it is talking about -- hence this counter. Get it wrong and
# the three rows collide on one key, or point at the wrong step's metrics.
step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    srun ../bin/perfectScaler --intensity "$intensity" --duration "$duration"

    ./slrmise --toml slurmise.toml record --step-id "$step" \
        "perfectScaler --intensity $intensity --duration $duration"

    step=$((step + 1))
done
