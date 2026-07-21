#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=perfectScaler
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

###############################################
#                                             #
# Approach 1: (current-style) Run then record #
#                                             #
###############################################
srun ../bin/perfectScaler --intensity 4444 --duration 20

./slrmise --toml slurmise.toml record \
    "perfectScaler --intensity 4444 --duration 20"

./slrmise display

#######################################################
#                                                     #
# Approach 2: Wrap the command like when using `time` #
#                                                     #
#######################################################
# slrmise adds a "stub" to the database.
# It doesn't have time/mem info yet because it is running inside the job.
# Then it hands off the process to the wrapped command with os.execvp()
srun ./slrmise --toml slurmise.toml lazy-record -- \
    ../bin/perfectScaler --intensity 5555 --duration 20

# `no-update-display` prints contents of database, NOTE time/mem not filled in!
./slrmise no-update-display

# `display` first backfills then displays. The lazy step above has finished,
# so even though this job is still running, the step's final time/mem/state
# are available and get filled in here.
./slrmise display
