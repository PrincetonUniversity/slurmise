import json

import click

from slurmise import job_data, job_database, parse_args, slurm


@click.group()
@click.option(
    "--database",
    "-d",
    type=click.Path(exists=False),
    default=".slurmise.h5",
    help="Path to the hdf5 database file",
)
@click.pass_context
def main(ctx, database):
    ctx.ensure_object(dict)
    ctx.obj["database"] = database


@main.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output")
@click.pass_context
def record(ctx, verbose):
    """Command to record a job.
    For example: `slurmise record -o 2 -i 3 -m fast`
    """
    parsed_args = parse_args.parse_slurmise_record_args(ctx.args)
    if verbose:
        print(json.dumps(parsed_args, indent=4))

    metadata_json = slurm.parse_slurm_job_metadata()
    if verbose:
        print("METADATA JSON", metadata_json)


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
