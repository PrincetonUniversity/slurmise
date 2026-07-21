#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=perfectScaler
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Approach 1: eager (upstream-style) -- command specified twice, exactly as
# tutorial 01 does it (no --step-id: with a single srun, sacct's last listed
# step is the right one, and the row is keyed by the bare job id).
# `|| true` so the known eager sacct race can't fail the job; a missing
# eager row in the database is itself the interesting result.
srun ../bin/perfectScaler --intensity 5000 --duration 10
./slrmise --toml slurmise.toml record \
    "perfectScaler --intensity 5000 --duration 10" || true

# Approach 2: lazy (record-then-exec) -- command specified once
srun ./slrmise --toml slurmise.toml lazy-record -- \
    ../bin/perfectScaler --intensity 5000 --duration 10
