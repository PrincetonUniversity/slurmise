#!/bin/bash
# Instead of `record` we use `raw-record` and tell `slurmise` that we already
# ran the 60 tasks of an array, as slurmids 33000-33059, and know how much time
# and memory each one took, so no job submission occurs.
#
# We're lying to slurmise here for the sake of the tutorial!
#
# Each task gets its own slurmid because each array task is a job in its own
# right -- which is why, unlike 02_jobs_in_loop/, there is no --step-id here.

scaling_categories=(linear quadratic cubic)

# Keep this in step with ../bin/categoricalScaler: it raises --intensity to the
# power its --scaling names -- 20, 400, 8000 MB -- and jitters the result by
# +/-20%. That spread between categories is what the lesson is about.
memories=(20 400 8000)

for task in $(seq 0 59); do
    idx=$((task % 3))
    scaling=${scaling_categories[$idx]}
    memory=${memories[$idx]}
    memory=$(( memory + memory * ((RANDOM % 41) - 20) / 100 ))

    # --scaling is a category, so it goes in --categories rather than
    # --numerics. That is the one thing this job does differently.
    slurmise --toml slurmise.toml raw-record \
        --job-name categoricalScaler \
        --slurm-id "$((33000 + task))" \
        --numerics '"intensity":20,"duration":10' \
        --categories "\"scaling\":\"$scaling\"" \
        --runtime "$(( 10 + (RANDOM % 5) - 2 ))" \
        --memory "$memory"
done
