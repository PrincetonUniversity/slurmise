import json
import sys

import click

import numpy as np
from slurmise import job_data, job_database, slurm
from slurmise.config import SlurmiseConfiguration
from slurmise.fit.poly_fit import PolynomialFit


@click.group()
@click.option(
    "--toml",
    "-t",
    type=click.Path(exists=True),
    required=False,
    help="Path to the hdf5 database file",
)
@click.pass_context
def main(ctx, toml):
    if toml is None:
        click.echo("Slurmise requires a toml file", err=True)
        click.echo("See readme for more information", err=True)
        sys.exit(1)
    ctx.ensure_object(dict)
    ctx.obj["config"] = SlurmiseConfiguration(toml_file=toml)
    ctx.obj["database"] = ctx.obj["config"].db_filename


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.option("--slurm-id", type=str, help="SLURM id of job")
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output")
@click.pass_context
def record(ctx, cmd, job_name, slurm_id, verbose):
    """Command to record a job.
    For example: `slurmise record "-o 2 -i 3 -m fast"`
    """
    metadata_json = slurm.parse_slurm_job_metadata(
        slurm_id=slurm_id
    )  # Pull just the slurm ID.

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


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.pass_context
def predict(ctx, cmd, job_name):
    query_jd = ctx.obj["config"].parse_job_cmd(cmd=cmd, job_name=job_name)
    query_jd = ctx.obj["config"].add_defaults(query_jd)
    query_model = PolynomialFit.load(
        query=query_jd, path=ctx.obj["config"].slurmise_base_dir
    )
    query_jd, query_warns = query_model.predict(query_jd)
    click.echo(f"Predicted runtime: {query_jd.runtime}")
    click.echo(f"Predicted memory: {query_jd.memory}")
    if query_warns:
        click.echo(click.style("Warnings:", fg="yellow"), err=True, color="red")
        for warn in query_warns:
            click.echo(f"  {warn}", err=True)


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.pass_context
def update_model(ctx, cmd, job_name):

    query_jd = ctx.obj["config"].parse_job_cmd(cmd=cmd, job_name=job_name)
    with job_database.JobDatabase.get_database(ctx.obj["database"]) as db:
        jobs = db.query(query_jd)

    try:
        query_model = PolynomialFit.load(
            query=query_jd, path=ctx.obj["config"].slurmise_base_dir
        )
    except FileNotFoundError:
        query_model = PolynomialFit(
            query=query_jd, degree=2, path=ctx.obj["config"].slurmise_base_dir
        )

    random_state = np.random.RandomState(42)
    query_model.fit(jobs, random_state=random_state)

    query_model.save()


@main.command()
@click.pass_context
def update_all(ctx):
    # Query the DB for all unique jobs and update the models
    with job_database.JobDatabase.get_database(ctx.obj["database"]) as db:
        for query_jd, jobs in db.iterate_database():
            try:
                query_model = PolynomialFit.load(query=query_jd, path=ctx.obj["config"].slurmise_base_dir)
            except FileNotFoundError:
                query_model = PolynomialFit(query=query_jd, degree=2, path=ctx.obj["config"].slurmise_base_dir)

            random_state = np.random.RandomState(42)
            query_model.fit(jobs, random_state=random_state)

            query_model.save()
