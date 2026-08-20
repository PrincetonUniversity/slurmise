#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran 39 steps of a job with slurmid 12345, and know how much time and memory
# each one took, so no job submission occurs.
#
#       Runs:  1   2     3    4    5    6    7    8    9   10   11   12   13
intensities=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500)
durations=(    2    4    8    8    4   10   12   14    2   13   19    5    9)

replicates=3

step=0
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    for (( j=0; j<replicates; j++ )); do

        slurmise --toml slurmise.toml raw-record \
            --job-name perfectScaler \
            --slurm-id 12345 --step-id "$step" \
            --numerics "\"intensity\":$intensity,\"duration\":$duration" \
            --runtime "$((duration + 2))" \
            --memory "$((intensity + 20))"

        step=$((step + 1))
    done
done
