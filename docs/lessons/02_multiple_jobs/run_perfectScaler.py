#!/usr/bin/env python3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=perfectScaler_python
#SBATCH --output=out_slurm_logs/slurm-%x-%j.out

# The same 13 runs as run_perfectScaler_loop.sbatch, looped in Python and recorded
# through slurmise's Python API. Submit it directly: sbatch --wait run_perfectScaler.py

import subprocess

from slurmise.api import Slurmise

slurmise = Slurmise()

intensities = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]

for step, intensity in enumerate(intensities):
    subprocess.run(["srun", "../bin/perfectScaler", "--intensity", str(intensity)], check=True)
    slurmise.record(f"perfectScaler --intensity {intensity}", step_id=str(step))
