import json
import os
import subprocess


def parse_slurm_job_metadata(slurm_id: str | None = None, step_name: str | None = None) -> dict:
    """
    Return a dictionary of metadata for the current SLURM job.
    Parameters:
        slurm_id (str | None): The SLURM job ID. If None, the function will attempt to retrieve
            the job ID from the SLURM_JOBID environment variable.
        step_id (str | None): The SLURM step ID. If None, the function defaults to the last step
            of the job. If provided, it specifies which step's metadata to return.
    Returns:
        dict: A dictionary containing metadata for the specified SLURM job and step.
    """
    
    sacct_json = get_slurm_job_sacct(slurm_id)
    
    try:
        job_id = sacct_json["jobs"][0]["job_id"]
        job_name = sacct_json["jobs"][0]["name"]
        state = sacct_json["jobs"][0]["state"]["current"][0]
        partition = sacct_json["jobs"][0]["partition"]
        CPUs = sacct_json["jobs"][0]["required"]["CPUs"]
        memory_per_cpu = sacct_json["jobs"][0]["required"]["memory_per_cpu"]
        memory_per_node = sacct_json["jobs"][0]["required"]["memory_per_node"]
        max_rss = 0
        steps = {}
        jobstep_ids = []
        for step in sacct_json["jobs"][0]["steps"]:
            steps[step["step"]["id"]] = step
            jobstep_ids.append(step["step"]["id"])

        if step_name is None:
            step_id = jobstep_ids[-1]
            step_name = step_id.split(".")[-1]
        else:
            step_id = f"{job_id}.{step_name}"

        # In addition, the max requested memory is updated as slurm steps are completed.
        elapsed_seconds = int(steps[step_id]["time"]["elapsed"])
        for item in steps[step_id]["tres"]["requested"]["max"]:
            if item["type"] == "mem":
                max_rss = max(max_rss, item["count"] // (2**20))  # convert to MB
    except Exception as e:
        raise ValueError(
            f"Could not parse json from sacct cmd:\n\n {sacct_json}"
        ) from e

    metadata = {
        "slurm_id": job_id,
        "step_id": step_name,
        "job_name": job_name,
        "state": state,
        "partition": partition,
        "elapsed_seconds": elapsed_seconds,
        "CPUs": CPUs,
        "memory_per_cpu": memory_per_cpu,
        "memory_per_node": memory_per_node,
        "max_rss": max_rss,
    }
    return metadata


def get_slurm_job_sacct(slurm_id: str | None = None) -> dict:
    """Return the JSON output of the sacct command for the current SLURM job."""
    if slurm_id is None:
        if "SLURM_JOBID" not in os.environ:
            raise ValueError("Not running in a SLURM job")
        slurm_id = os.environ["SLURM_JOBID"]

    try:
        json_encoded_str = subprocess.check_output(["sacct", "-j", slurm_id, "--json"])
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error running sacct cmd: {e}") from e

    return json.loads(json_encoded_str.decode())
