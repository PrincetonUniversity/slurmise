#!/bin/bash
# Stands in for run_as_launcher.sbatch when there is no scheduler.
#
# The rows are still recorded as stubs and still need a `sync` before they have
# any metrics, so the lesson's two-stage story holds without a cluster.
export SLRMISE_MOCK_JOB=90000002
export SLRMISE_USED_TIME=7

#          Runs:  1    2    3
intensities=(2000 3000 4000)
durations=(     5    5    5)

# No step counter here either: with no SLURM_STEP_ID to read, ./slrmise takes
# the next free step under SLRMISE_MOCK_JOB.
for i in "${!intensities[@]}"; do
    intensity=${intensities[$i]}
    duration=${durations[$i]}

    SLRMISE_USED_MEM=$((intensity + 15)) \
        ./slrmise --toml slurmise.toml lazy-record -- \
            ../bin/perfectScaler --intensity "$intensity" --duration "$duration"
done
