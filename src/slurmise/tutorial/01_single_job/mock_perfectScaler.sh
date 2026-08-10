#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that
# we already ran a job with slurmid 12345 and know how much time and memory
# the process took, so no job submission occurs.
#
# We're lying to slurmise here for the sake of the tutorial!

slurmise --toml slurmise.toml raw-record \
    --job-name perfectScaler \
    --slurm-id 12345 \
    --numerics '"intensity":5000,"duration":10' \
    --runtime 12 \
    --memory 5020
