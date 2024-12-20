import json

import click

from slurmise import job_data, job_database, slurm
from slurmise.config import SlurmiseConfiguration


@click.group()
@click.option(
    "--database",
    "-d",
    type=click.Path(exists=False),
    default=".slurmise.h5",
    help="Path to the hdf5 database file",
)
# TODO: Make this optional with a default config.
@click.option(
    "--toml",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Path to the hdf5 database file",
)
@click.pass_context
def main(ctx, database, toml):
    ctx.ensure_object(dict)
    ctx.obj["database"] = database
    ctx.obj["config"] = SlurmiseConfiguration(toml_file=toml)


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.option("--slurm-id", type=str, help="SLURM id of job")
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output")
@click.pass_context
# TODO: create a small toml config file and test it until the job data is created and
# populated with recorded information
def record(ctx, cmd, job_name, slurm_id, verbose):
    """Command to record a job.
    For example: `slurmise record "-o 2 -i 3 -m fast"`
    """
    metadata_json = slurm.parse_slurm_job_metadata(slurm_id=slurm_id)  # Pull just the slurm ID.

    parsed_jd = ctx.obj["config"].parse_job_cmd(
        cmd=cmd, job_name=job_name, slurm_id=metadata_json["slurm_id"]
    )

    if verbose:
        print(json.dumps(parsed_jd, indent=4))

    parsed_jd.memory = metadata_json["max_rss"]
    parsed_jd.runtime = metadata_json["elapsed_seconds"]

    if verbose:
        print("METADATA JSON", metadata_json)
    with job_database.JobDatabase.get_database(ctx.obj["database"]) as db:
        db.record(parsed_jd)


@main.command()
@click.option("--job-name", type=str, required=True, help="Name of the job")
@click.option("--slurm-id", type=str, required=True, help="SLURM id of job")
@click.option(
    "--numerical",
    type=str,
    help="Numerical run parameters in JSON format without outer {}, such as 'n:3,q:17.4'",
)
@click.option(
    "--categorical",
    type=str,
    help="Categorical run parameters in JSON format without outer {}",
)
@click.option("--cmd", type=str, help="Actual command run")
@click.pass_context
def raw_record(ctx, job_name, slurm_id, numerical, categorical, cmd):
    """Record a job"""
    # NOTE hack to get JSON parsing working with click who is too eager to try and process the args
    categorical = json.loads("{" + categorical + "}") if categorical else {}
    numerical = json.loads("{" + numerical + "}") if numerical else {}

    jd = job_data.JobData(
        job_name=job_name,
        slurm_id=slurm_id,
        numerical=numerical,
        categorical=categorical,
        cmd=cmd,
    )

    with job_database.JobDatabase.get_database(ctx.obj["database"]) as db:
        db.record(jd)


@main.command()
@click.pass_context
def print(ctx):
    with job_database.JobDatabase.get_database(ctx.obj["database"]) as db:
        db.print()
