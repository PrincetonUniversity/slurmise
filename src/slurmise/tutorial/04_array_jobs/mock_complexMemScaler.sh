#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran the 26 tasks of an array, as slurmids 32000-32025, and know how much time
# and memory each one took, so no job submission occurs.
#
# We're lying to slurmise here for the sake of the tutorial!
#
# Each task gets its own slurmid because each array task is a job in its own
# right -- which is why, unlike 02_jobs_in_loop/, there is no --step-id here.

intensities=(500 1000 1500 2000 2500 3000 3500 4000)
duration=10

for task in $(seq 0 25); do
    intensity=${intensities[$((task % 8))]}

    # Keep this in step with ../bin/complexMemScaler: it jitters --intensity by
    # +/-20% and adds a flat 1000 MB.
    noise=$(( (RANDOM % 41) - 20 ))
    memory=$(( intensity + intensity * noise / 100 + 1000 ))

    slurmise --toml slurmise.toml raw-record \
        --job-name complexMemScaler \
        --slurm-id "$((32000 + task))" \
        --numerics "\"intensity\":$intensity,\"duration\":$duration" \
        --runtime "$((duration + 2))" \
        --memory "$memory"
done
