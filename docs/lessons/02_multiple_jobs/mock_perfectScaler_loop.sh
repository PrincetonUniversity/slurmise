#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran 13 steps of a job with slurmid 12345, and know how much time and memory
# each one took, so no job submission occurs. perfectScaler uses exactly
# --intensity MB, and each step finishes within a second, which slurmise
# records as 0 minutes.

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 0 \
    --numerics '"intensity":500' --used-mbs 500 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 1 \
    --numerics '"intensity":1000' --used-mbs 1000 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 2 \
    --numerics '"intensity":1500' --used-mbs 1500 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 3 \
    --numerics '"intensity":2000' --used-mbs 2000 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 4 \
    --numerics '"intensity":2500' --used-mbs 2500 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 5 \
    --numerics '"intensity":3000' --used-mbs 3000 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 6 \
    --numerics '"intensity":3500' --used-mbs 3500 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 7 \
    --numerics '"intensity":4000' --used-mbs 4000 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 8 \
    --numerics '"intensity":4500' --used-mbs 4500 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 9 \
    --numerics '"intensity":5000' --used-mbs 5000 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 10 \
    --numerics '"intensity":5500' --used-mbs 5500 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 11 \
    --numerics '"intensity":6000' --used-mbs 6000 --used-minutes 0

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 12 \
    --numerics '"intensity":6500' --used-mbs 6500 --used-minutes 0
