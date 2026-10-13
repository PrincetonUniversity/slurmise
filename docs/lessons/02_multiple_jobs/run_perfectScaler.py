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

runs = [
    (500, 2), (1000, 4), (1500, 8), (2000, 8), (2500, 4), (3000, 10), (3500, 12),
    (4000, 14), (4500, 2), (5000, 13), (5500, 19), (6000, 5), (6500, 9),
]  # fmt: skip

for step, (intensity, duration) in enumerate(runs):
    args = f"--intensity {intensity} --duration {duration}"
    subprocess.run(["srun", "../bin/perfectScaler", *args.split()], check=True)
    slurmise.record(f"perfectScaler {args}", step_id=str(step))
