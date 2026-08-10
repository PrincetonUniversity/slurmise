#!/usr/bin/env bash
#
# Unlike the unit tests (which stub `sacct`), this
# End-to-end tutorial integration test against a slurm scheduler.
#
# It is meant to run in the GitHub Actions workflow or on a cluster,
# NOT as part of the regular unit tests
#
# Relies on uv
#
# You might need to export your SBATCH_ACCOUNT
set -euo pipefail

# It runs `slurmise-generate-tutorial` and then walks the generated tutorial with
# its own `tutorial.py`, which submits the shipped `.sbatch` files and checks
# each lesson's `#>` expectations -- so a broken lesson fails here, not just a
# broken job.
#
# Environment knobs:
#   LESSONS         space-separated lesson directories to walk (default: all of
#                   them, in order). e.g. LESSONS="01_single_job 02_jobs_in_loop"
#   RUN_ACCT_PROBE  set to 1 to run the accounting probe first (CI uses this on a
#                   freshly stood-up cluster; skip it on a real cluster).
#   SBATCH_ACCOUNT  honored natively by sbatch -- export it on clusters that
#                   require an account (likewise SBATCH_PARTITION / SBATCH_QOS).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Use the project's uv environment. Export its bin dir onto PATH so the bare
# `slurmise` / `slurmise-generate-tutorial` calls inside the shipped .sbatch
# files resolve once sbatch propagates the environment (default --export=ALL).
VENV_BIN="$(cd "$ROOT" && uv run python -c 'import os, sys; print(os.path.dirname(sys.executable))')"
export PATH="$VENV_BIN:$PATH"

# Overridable so CI can point it at a known path to upload as an artifact.
# The default temp dir is created next to the repo (a shared filesystem) rather
# than in /tmp: /tmp is node-local on many clusters, so a /tmp workdir is
# invisible to the compute node and `srun ../bin/...` fails to find the binary.
WORK="${WORK:-$(mktemp -d "$ROOT/tutorial-run.XXXXXX")}"
mkdir -p "$WORK"
TUTORIAL="$WORK/tutorial"
echo "Work dir:   $WORK"
echo "venv bin:   $VENV_BIN"
echo "lessons:    ${LESSONS:-all}"

# Generate the tutorial files
slurmise-generate-tutorial --dest "$TUTORIAL"

# -----------------------------------------------------------------------------
# Optional accounting probe: confirm sacct reports nonzero MaxRSS for a short
# job. On a freshly stood-up cluster this is the single most likely thing to be
# misconfigured (JobAcctGatherType=none), so fail loudly with diagnostics.
# -----------------------------------------------------------------------------
if [ "${RUN_ACCT_PROBE:-0}" = 1 ]; then
  echo "== accounting probe =="
  probe_bin="$TUTORIAL/bin/perfectScaler"
  probe_id="$(sbatch --parsable --wait \
    --job-name=probe --mem=2G --time=00:03:00 --acctg-freq=task=1 \
    --wrap "srun --acctg-freq=task=1 '$probe_bin' --intensity 400 --duration 15")"
  echo "probe job id: $probe_id"

  max_rss="$(sacct -j "$probe_id" --json | python3 -c '
import json, sys
data = json.load(sys.stdin)
best = 0
for job in data.get("jobs", []):
    for step in job.get("steps", []):
        for item in step.get("tres", {}).get("requested", {}).get("max", []):
            if item.get("type") == "mem":
                best = max(best, item.get("count", 0))
print(best)
')"
  echo "probe MaxRSS (bytes): $max_rss"
  if [ "$max_rss" -le 0 ]; then
    echo "ERROR: sacct reported MaxRSS=0 for the probe job." >&2
    echo "Accounting is not capturing memory. Check that slurmdbd is running and" >&2
    echo "that JobAcctGatherType is jobacct_gather/linux (or /cgroup), not /none." >&2
    echo "--- scontrol show config (JobAcctGather*) ---" >&2
    scontrol show config | grep -i jobacctgather >&2 || true
    echo "--- sacct table ---" >&2
    sacct -j "$probe_id" --format=JobID,JobName,State,MaxRSS,Elapsed >&2 || true
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# Walk the tutorial. The markdown supplies the ordering and the assertions, so
# there is no example-to-sbatch mapping here any more.
#
# `--yes` runs unattended and answers every `#> option` block with its first
# alternative, which the lessons write as the real one -- so this submits for
# real. `--option mock` would be the no-cluster path, which is not what this
# workflow is for.
# -----------------------------------------------------------------------------
( cd "$TUTORIAL" && ./tutorial.py --yes ${LESSONS:-} )
