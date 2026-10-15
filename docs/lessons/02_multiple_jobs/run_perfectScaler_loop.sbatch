#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=perfectScaler_loop
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# 13 runs of perfectScaler, one after another in this one job. Each srun is a step
# of the job, so each record gets its own --step-id.

srun ../bin/perfectScaler --intensity 500
slurmise record --step-id 0 "perfectScaler --intensity 500"

srun ../bin/perfectScaler --intensity 1000
slurmise record --step-id 1 "perfectScaler --intensity 1000"

srun ../bin/perfectScaler --intensity 1500
slurmise record --step-id 2 "perfectScaler --intensity 1500"

srun ../bin/perfectScaler --intensity 2000
slurmise record --step-id 3 "perfectScaler --intensity 2000"

srun ../bin/perfectScaler --intensity 2500
slurmise record --step-id 4 "perfectScaler --intensity 2500"

srun ../bin/perfectScaler --intensity 3000
slurmise record --step-id 5 "perfectScaler --intensity 3000"

srun ../bin/perfectScaler --intensity 3500
slurmise record --step-id 6 "perfectScaler --intensity 3500"

srun ../bin/perfectScaler --intensity 4000
slurmise record --step-id 7 "perfectScaler --intensity 4000"

srun ../bin/perfectScaler --intensity 4500
slurmise record --step-id 8 "perfectScaler --intensity 4500"

srun ../bin/perfectScaler --intensity 5000
slurmise record --step-id 9 "perfectScaler --intensity 5000"

srun ../bin/perfectScaler --intensity 5500
slurmise record --step-id 10 "perfectScaler --intensity 5500"

srun ../bin/perfectScaler --intensity 6000
slurmise record --step-id 11 "perfectScaler --intensity 6000"

srun ../bin/perfectScaler --intensity 6500
slurmise record --step-id 12 "perfectScaler --intensity 6500"
