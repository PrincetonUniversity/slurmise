import os
import json
import subprocess

def parse_slurm_job_metadata() -> dict:
    """Return a dictionary of metadata for the current SLURM job."""
    sacct_json = get_slurm_job_sacct()
    sstat_out = get_slurm_job_sstat()

    try:
        elapsed_seconds = int(sacct_json["jobs"][0]["time"]["elapsed"])
    except Exception as e:
        raise ValueError(f"Could not parse json from sacct cmd:\n\n {elapsed_out}") from e

    metadata = {
        "elapsed_seconds": elapsed_seconds,
    }
    return metadata

def get_slurm_job_sacct() -> dict:
    """Return the JSON output of the sacct command for the current SLURM job."""
    if 'SLURM_JOBID' not in os.environ:
        raise ValueError("Not running in a SLURM job")

    try:
        json_encoded_str = subprocess.check_output(['sacct', '-j', os.environ['SLURM_JOBID'], '--json'])
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error running sacct cmd: {e}") from e

    return json.loads(json_encoded_str.decode())


def get_slurm_job_sstat() -> dict:
    try:
        rss_out = subprocess.check_output([
            "sstat", "-j", os.environ['SLURM_JOBID'], "--format", "maxrss",
            "--noheader", "--parsable",
            ])
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error running sstat cmd: {e}") from e
    
    #print(rss_out) #TODO RSS is empty, need to figure out how to get it
    return {}

