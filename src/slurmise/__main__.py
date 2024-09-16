import click
import os
import re
import subprocess
import json

from slurmise import utils
from slurmise import slurm


@click.group()
def main():
    pass

@main.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('-v', '--verbose', is_flag=True, help="Print verbose output")
@click.pass_context
def record(ctx, verbose):
    """Command to record a job.
    For example: `slurmise record -o 2 -i 3 -m fast`
    """
    parsed_args = utils.parse_slurmise_record_args(ctx.args)
    print(json.dumps(parsed_args, indent=4))

    metadata_json = slurm.parse_slurm_job_metadata()
    print("METADATA JSON", metadata_json)