#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran 13 steps of a job with slurmid 12345, and know how much time and memory
# each one took, so no job submission occurs. perfectScaler uses exactly
# --intensity MB, and we pretend each step ran for --duration minutes.

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 0 \
    --numerics '"intensity":500,"duration":2' --used-mbs 500 --used-minutes 2

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 1 \
    --numerics '"intensity":1000,"duration":4' --used-mbs 1000 --used-minutes 4

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 2 \
    --numerics '"intensity":1500,"duration":8' --used-mbs 1500 --used-minutes 8

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 3 \
    --numerics '"intensity":2000,"duration":8' --used-mbs 2000 --used-minutes 8

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 4 \
    --numerics '"intensity":2500,"duration":4' --used-mbs 2500 --used-minutes 4

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 5 \
    --numerics '"intensity":3000,"duration":10' --used-mbs 3000 --used-minutes 10

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 6 \
    --numerics '"intensity":3500,"duration":12' --used-mbs 3500 --used-minutes 12

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 7 \
    --numerics '"intensity":4000,"duration":14' --used-mbs 4000 --used-minutes 14

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 8 \
    --numerics '"intensity":4500,"duration":2' --used-mbs 4500 --used-minutes 2

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 9 \
    --numerics '"intensity":5000,"duration":13' --used-mbs 5000 --used-minutes 13

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 10 \
    --numerics '"intensity":5500,"duration":19' --used-mbs 5500 --used-minutes 19

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 11 \
    --numerics '"intensity":6000,"duration":5' --used-mbs 6000 --used-minutes 5

slurmise raw-record --job-name perfectScaler --slurm-id 12345 --step-id 12 \
    --numerics '"intensity":6500,"duration":9' --used-mbs 6500 --used-minutes 9
