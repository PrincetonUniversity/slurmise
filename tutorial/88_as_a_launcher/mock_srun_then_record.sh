#!/bin/bash
# Stands in for srun_then_record.sbatch when there is no scheduler.
#
# Instead of submitting, we tell ./slrmise what each run would have used, so no
# sbatch, srun or sacct is ever involved. SLRMISE_MOCK_JOB is the job id the
# scheduler would have handed out, so all three steps land under one job.
#
# We're lying to slurmise here so we don't have to wait for the queue!
export SLRMISE_MOCK_JOB=90000001
export SLRMISE_USED_TIME=7

#          Runs:  1    2    3
intensities=(2000 3000 4000)
durations=(     5    5    5)

step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    SLRMISE_USED_MEM=$((intensity + 15)) \
        ./slrmise --toml slurmise.toml record --step-id "$step" \
            "perfectScaler --intensity $intensity --duration $duration"

    step=$((step + 1))
done
