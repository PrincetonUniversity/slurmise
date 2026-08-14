from __future__ import annotations

import json
import os
import subprocess
from math import ceil

JOB_ID_ENV_VARS = ("SLURM_JOB_ID", "SLURM_JOBID")  # modern name first; SLURM sets both


def get_current_job_id() -> str | None:
    """Return the job ID of the current SLURM job, or None when not inside a SLURM job."""
    for env_var in JOB_ID_ENV_VARS:
        if env_var in os.environ:
            return os.environ[env_var]
    return None


def resolve_job_id(slurm_id: str | int | None = None, step_id: str | None = None) -> str:
    """
    Resolve a job ID, falling back to the current SLURM job's environment.
    Parameters:
        slurm_id (str | int | None): The SLURM job ID. If None, the ID is read from the
            SLURM_JOB_ID environment variable.
        step_id (str | None): The SLURM step ID. If provided, it is appended to the job ID
            as "<slurm_id>.<step_id>".
    Returns:
        str: The resolved job ID, with the step ID appended if provided.
    """
    if slurm_id is None:
        slurm_id = get_current_job_id()
    if slurm_id is None:
        msg = (
            "slurm_id was not provided and the SLURM_JOB_ID environment variable "
            "is not set; not running inside a SLURM job."
        )
        raise ValueError(msg)
    slurm_id = str(slurm_id)
    if step_id is not None and "." not in slurm_id:
        slurm_id = f"{slurm_id}.{step_id}"
    return slurm_id


def split_job_id(slurm_id: str) -> tuple[str, str | None]:
    """Split a combined "<slurm_id>.<step_id>" string into its parts."""
    if "." in slurm_id:
        job_id, step_id = slurm_id.split(".", 1)
        return job_id, step_id
    return slurm_id, None


def parse_slurm_job_metadata(slurm_id: str | None = None, step_id: str | None = None) -> dict:
    """
    Return a dictionary of metadata for the current SLURM job.
    Parameters:
        slurm_id (str | None): The SLURM job ID. If None, the function will attempt to retrieve
            the job ID from the SLURM_JOB_ID environment variable.
        step_id (str | None): The SLURM step ID. If None, the function defaults to the last step
            of the job. If provided, it specifies which step's metadata to return.
    Returns:
        dict: A dictionary containing metadata for the specified SLURM job and step.
    """

    slurm_id = resolve_job_id(slurm_id)
    sacct_json = get_slurm_job_sacct(slurm_id)

    try:
        job_id = sacct_json["jobs"][0]["job_id"]
        job_name = sacct_json["jobs"][0]["name"]
        state = sacct_json["jobs"][0]["state"]["current"][0]
        partition = sacct_json["jobs"][0]["partition"]
        cpus = sacct_json["jobs"][0]["required"]["CPUs"]
        memory_per_cpu = sacct_json["jobs"][0]["required"]["memory_per_cpu"]
        memory_per_node = sacct_json["jobs"][0]["required"]["memory_per_node"]
        max_rss = 0
        steps = {}
        jobstep_ids = []
        for step in sacct_json["jobs"][0]["steps"]:
            steps[step["step"]["id"]] = step
            jobstep_ids.append(step["step"]["id"])

        if step_id is None:
            step_key = jobstep_ids[-1]
            step_id = step_key.split(".")[-1]
        else:
            step_key = f"{job_id}.{step_id}"

        # In addition, the max requested memory is updated as slurm steps are completed.
        elapsed_seconds = int(steps[step_key]["time"]["elapsed"])
        task_count = steps[step_key]["tasks"]["count"]
        for item in steps[step_key]["tres"]["requested"]["max"]:
            if item["type"] == "mem":
                max_rss = max(max_rss, ceil(item["count"] / (2**20)) * task_count)  # convert to MB
    except Exception as e:
        msg = f"Could not parse json from sacct cmd:\n\n {sacct_json}"
        raise ValueError(msg) from e

    return {
        "slurm_id": job_id,
        "step_id": step_id,
        "job_name": job_name,
        "state": state,
        "partition": partition,
        "elapsed_seconds": elapsed_seconds,
        "CPUs": cpus,
        "memory_per_cpu": memory_per_cpu,
        "memory_per_node": memory_per_node,
        "max_rss": max_rss,
    }


def get_slurm_job_sacct(slurm_id: str) -> dict:
    """Return the JSON output of the sacct command for the given SLURM job."""
    try:
        json_encoded_str = subprocess.check_output(["sacct", "-j", slurm_id, "--json"])
    except subprocess.CalledProcessError as e:
        msg = f"Error running sacct cmd: {e}"
        raise ValueError(msg) from e

    return json.loads(json_encoded_str.decode())
