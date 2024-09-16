import click
import os
import re
import subprocess
import json


@click.group()
def main():
    pass

@main.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
def record(ctx):
    """Command to record a job.
    For example: `slurmise record -o 2 -i 3 -m fast`
    """
    print("SLURMISE RECORD ARGS", ctx.args)

    #TODO get job info from slurm for currently running job
    if 'SLURM_JOBID' not in os.environ:
        raise click.ClickException("Not running in a SLURM job")

    try:
        elapsed_out = subprocess.check_output(['sacct', '-j', os.environ['SLURM_JOBID'], '--json'])
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Error running sacct cmd: {e}") from e


    try:
        elapsed_out = json.loads(elapsed_out.decode())
        elapsed_seconds = int(elapsed_out["jobs"][0]["time"]["elapsed"])
    except Exception as e:
        raise click.ClickException(f"Could not parse json from sacct cmd:\n\n {elapsed_out}") from e

    print("Elapsed time in seconds: ", elapsed_seconds)

    try:
        rss_out = subprocess.check_output(["sstat", "-j", os.environ['SLURM_JOBID'], "--format", "maxrss"])
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Error running sstat cmd: {e}") from e
    
    #print(rss_out) #TODO RSS is empty, need to figure out how to get it

    #time_regex = re.compile(r"(?P<days>\d*)-?(?P<hours>\d+):(?P<minutes>\d\d):(?P<seconds>\d\d)")
    #time_match = time_regex.search(elapsed_out)
    #if time_match:
    #    print(time_match.groupdict())
    #else:
    #    raise click.ClickException(f"Could not parse time:\n\n {elapsed_out}")

    #TODO parse args
