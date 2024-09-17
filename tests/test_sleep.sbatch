#!/bin/bash
#SBATCH --job-name=sleep                         # create a short name for your job
#SBATCH --nodes=1                                # node count
#SBATCH --ntasks=1                               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                        # cpu-cores per task (equal to $SLURM_CPUS_PER_TASK)
#SBATCH --mem=1G                                 # total cpu requested memory
#SBATCH --time=00:01:00                          # total run time limit (HH:MM:SS)

#set bash strict mode (http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

sleep 2 #this is the fake command we're running

slurmise record sleep 2 -o 2 -i 3 -m fast
