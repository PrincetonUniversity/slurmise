#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=perfectScaler
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Start from a clean database so this example's output stands alone
# (lazy_recording/ holds two sbatch files that share ./slurmise.h5).
rm -f slurmise.h5

###############################################
#                                             #
# Approach 1: (current-style) Run then record #
#                                             #
###############################################
# Run the command
srun ../bin/perfectScaler --intensity 4444 --duration 20

./slrmise --toml slurmise.toml record \
    "perfectScaler --intensity 4444 --duration 20"

echo "==================================================="
echo "Contents of slurmise.h5 after standard eager-record"
echo "==================================================="
./slrmise display
echo ""

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

echo "========================================================"
echo "Contents of slurmise.h5 after lazy-record w/out backfill"
echo "========================================================"
./slrmise no-update-display
echo ""

echo "======================================================="
echo "Contents of slurmise.h5 after lazy-record WITH backfill"
echo "======================================================="
./slrmise display
echo ""
