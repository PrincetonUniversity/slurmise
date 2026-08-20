#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran the 26 tasks of an array, as slurmids 31000-31025, and know how much time
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

    # ../bin/perfectScaler allocates --intensity MB and sleeps --duration
    # seconds, so a real run measures a little over both.
    slurmise --toml slurmise.toml raw-record \
        --job-name perfectScaler \
        --slurm-id "$((31000 + task))" \
        --numerics "\"intensity\":$intensity,\"duration\":$duration" \
        --runtime "$((duration + 2))" \
        --memory "$((intensity + 20))"
done
