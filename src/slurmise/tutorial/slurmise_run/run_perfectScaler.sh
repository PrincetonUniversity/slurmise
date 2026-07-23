#!/bin/bash
# The front-door workflow: `slrmise run` predicts each command's resources,
# submits it with sbatch, and records a stub for the new job.
#
# Run this script on the login node with `bash run_perfectScaler.sh`,
# it submits jobs and is not itself a job.
#
# Each `run` does:
#   1. sync the slurmise h5 database to fill in time/mem from stubs of prior runs by running sacct
#   2. predict time/mem for this command. Use toml defaults if there's not enough data for `fit`
#   3. subprocess run of `sbatch --mem=<pred> --time=<pred> --wrap "<command>"`
#   4. write a stub row for the new job id (features + predicted resources)
#
set -euo pipefail

# Keep the submitted jobs' logs out of this directory
mkdir -p out_slurm_logs
export SBATCH_OUTPUT="out_slurm_logs/slurm-%j.out"

#       Runs:  1   2     3    4    5    6
intensities=(500 1000 1500 2000 2500 3000)
durations=(    2    4    8    8    4   10)

# This loop is submitting SLURM jobs
for i in "${!intensities[@]}"; do
    ./slrmise --toml slurmise.toml run -- \
        ../bin/perfectScaler --intensity "${intensities[$i]}" --duration "${durations[$i]}"
done

echo ""
echo "Submitted ${#intensities[@]} jobs. The database now holds their stubs:"
./slrmise display --no-sync
