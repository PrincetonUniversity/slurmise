#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran 39 steps of a job with slurmid 23456, and know how much time and memory
# each one took, so no job submission occurs.
#
# We're lying to slurmise here for the sake of the tutorial!
#
# The steps share one slurmid, exactly as they would inside a real allocation,
# so --step-id is what keeps their records apart.

#       Runs:  1   2     3    4    5    6    7    8    9   10   11   12   13
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(    2    4    8    8    4   10   12   14    2   13   19    5    9)

replicates=3

step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    for (( j=0; j<replicates; j++ )); do
        # Keep this in step with ../bin/complexMemScaler: it jitters --intensity
        # by +/-20% and adds a flat 1000 MB. The noise is the whole point of
        # this job, so the numbers we make up here need it too -- without it the
        # fit would be suspiciously good and the lesson would teach the wrong
        # thing.
        noise=$(( (RANDOM % 41) - 20 ))
        memory=$(( intensity + intensity * noise / 100 + 1000 ))

        slurmise --toml slurmise.toml raw-record \
            --job-name complexMemScaler \
            --slurm-id 23456 --step-id "$step" \
            --numerics "\"intensity\":$intensity,\"duration\":$duration" \
            --runtime "$((duration + 2))" \
            --memory "$memory"

        step=$((step + 1))
    done
done
