#!/bin/bash
set -euo pipefail

# Keep the submitted job's log out of this directory
mkdir -p out_slurm_logs
export SBATCH_OUTPUT="out_slurm_logs/slurm-%j.out"

./slrmise --toml slurmise.toml run -- \
    ../bin/perfectScaler \
        --intensity 6500 \
        --duration 9
