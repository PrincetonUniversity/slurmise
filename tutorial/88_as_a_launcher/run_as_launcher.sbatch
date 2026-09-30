#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=run_as_launcher
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Approach 2 -- lazy: slrmise wraps the command, the way `time` does.
#
# The same three runs as srun_then_record.sbatch, so the two sets of records can
# be compared directly.

#          Runs:  1    2    3
intensities=(2000 3000 4000)
durations=(     5    5    5)

# No step counter. slrmise runs inside the step, so it reads SLURM_STEP_ID for
# itself -- and the command is written once, after the `--`, with nothing to
# keep in sync.
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    srun ./slrmise --toml slurmise.toml lazy-record -- \
        ../bin/perfectScaler --intensity "$intensity" --duration "$duration"
done
