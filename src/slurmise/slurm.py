import json
import os
import subprocess


def parse_slurm_job_metadata(slurm_id: str | None = None, step_id: str | None = None) -> dict:
    """Return a dictionary of metadata for the current SLURM job."""
    sacct_json = get_slurm_job_sacct(slurm_id)
    # sstat_out = get_slurm_job_sstat(slurm_id)

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
        step_ids = []
        for step in sacct_json["jobs"][0]["steps"]:
            steps[step["id"]] = step
            step_ids.append(step["id"])
        # step_ids = {step["id"]=step for step in sacct_json["jobs"][0]["steps"]}
        step_id = step_ids[-1] if step_id is None else ".".join([job_id, step_id])
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
        "step_id": step_id,
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


def get_slurm_job_sstat(slurm_id: str | None = None) -> dict:
    if slurm_id is None:
        if "SLURM_JOBID" not in os.environ:
            raise ValueError("Not running in a SLURM job")
        slurm_id = os.environ["SLURM_JOBID"]

    try:
        rss_out = subprocess.check_output(  # noqa: F841
            [
                "sstat",
                "-j",
                slurm_id,
                "--format",
                "maxrss",
                "--noheader",
                "--parsable",
            ]
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error running sstat cmd: {e}") from e

    # print(rss_out) # TODO RSS is empty, need to figure out how to get it
    return {}
