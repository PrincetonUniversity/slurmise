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

srun ../bin/perfectScaler --intensity 500 --duration 2
slurmise record --step-id 0 "perfectScaler --intensity 500 --duration 2"

srun ../bin/perfectScaler --intensity 1000 --duration 4
slurmise record --step-id 1 "perfectScaler --intensity 1000 --duration 4"

srun ../bin/perfectScaler --intensity 1500 --duration 8
slurmise record --step-id 2 "perfectScaler --intensity 1500 --duration 8"

srun ../bin/perfectScaler --intensity 2000 --duration 8
slurmise record --step-id 3 "perfectScaler --intensity 2000 --duration 8"

srun ../bin/perfectScaler --intensity 2500 --duration 4
slurmise record --step-id 4 "perfectScaler --intensity 2500 --duration 4"

srun ../bin/perfectScaler --intensity 3000 --duration 10
slurmise record --step-id 5 "perfectScaler --intensity 3000 --duration 10"

srun ../bin/perfectScaler --intensity 3500 --duration 12
slurmise record --step-id 6 "perfectScaler --intensity 3500 --duration 12"

srun ../bin/perfectScaler --intensity 4000 --duration 14
slurmise record --step-id 7 "perfectScaler --intensity 4000 --duration 14"

srun ../bin/perfectScaler --intensity 4500 --duration 2
slurmise record --step-id 8 "perfectScaler --intensity 4500 --duration 2"

srun ../bin/perfectScaler --intensity 5000 --duration 13
slurmise record --step-id 9 "perfectScaler --intensity 5000 --duration 13"

srun ../bin/perfectScaler --intensity 5500 --duration 19
slurmise record --step-id 10 "perfectScaler --intensity 5500 --duration 19"

srun ../bin/perfectScaler --intensity 6000 --duration 5
slurmise record --step-id 11 "perfectScaler --intensity 6000 --duration 5"

srun ../bin/perfectScaler --intensity 6500 --duration 9
slurmise record --step-id 12 "perfectScaler --intensity 6500 --duration 9"
