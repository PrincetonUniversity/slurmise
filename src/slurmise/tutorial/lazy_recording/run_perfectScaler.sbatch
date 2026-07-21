#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=perfectScaler
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# Approach 1: eager (upstream-style) -- command specified twice.
# `|| true` so the known eager sacct race can't fail the job; a missing
# eager row in the database is itself the interesting result.
srun ../bin/perfectScaler --intensity 5000 --duration 10
./lazy_slurmise --toml slurmise.toml --step-id 0 record \
    "perfectScaler --intensity 5000 --duration 10" || true

# Approach 2: lazy (record-then-exec) -- command specified once
srun ./lazy_slurmise --toml slurmise.toml lazy-record -- \
    ../bin/perfectScaler --intensity 5000 --duration 10
