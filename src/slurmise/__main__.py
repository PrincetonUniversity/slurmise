import json
import sys

import click

from slurmise import job_data
from slurmise.api import Slurmise


@click.group()
@click.option(
    "--toml",
    "-t",
    type=click.Path(exists=True),
    required=False,
    help="Path to the slurmise configuration file",
)
@click.pass_context
def main(ctx, toml):
    if toml is None:
        click.echo("Slurmise requires a toml file", err=True)
        click.echo("See readme for more information", err=True)
        sys.exit(1)
    ctx.ensure_object(dict)
    ctx.obj["slurmise"] = Slurmise(toml)


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.option("--slurm-id", type=str, help="SLURM id of job")
@click.option("--step-id", type=str, help="SLURM step id")
@click.pass_context
def record(ctx, cmd, job_name, slurm_id, step_id):
    """Command to record a job.
    For example: `slurmise record "-o 2 -i 3 -m fast"`
    """
    ctx.obj["slurmise"].record(cmd, job_name, slurm_id, step_id)


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.pass_context
def parse(ctx, cmd, job_name):
    """Command to record a job.
    For example: `slurmise record "-o 2 -i 3 -m fast"`
    """
    parsed_output = ctx.obj["slurmise"].dry_parse(cmd, job_name)
    click.echo(parsed_output)


@main.command()
@click.option("--job-name", type=str, required=True, help="Name of the job")
@click.option("--slurm-id", type=str, required=True, help="SLURM id of job")
@click.option("--step-id", type=str, required=False, default=None, help="SLURM step id")
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
def raw_record(ctx, job_name, slurm_id, step_id, numerical, categorical, cmd):
    """Record a job"""
    # NOTE hack to get JSON parsing working with click who is too eager to try and process the args
    categorical = json.loads("{" + categorical + "}") if categorical else {}
    numerical = json.loads("{" + numerical + "}") if numerical else {}
    slurm_id = f"{slurm_id}.{step_id}" if step_id is not None else slurm_id

    jd = job_data.JobData(
        job_name=job_name,
        slurm_id=slurm_id,
        numerical=numerical,
        categorical=categorical,
        cmd=cmd,
    )

    ctx.obj["slurmise"].raw_record(jd)


@main.command()
@click.pass_context
def print(ctx):
    ctx.obj["slurmise"].print()


@main.command()
@click.argument("cmd", nargs=1)
@click.option("--job-name", type=str, help="Name of the job")
@click.pass_context
def predict(ctx, cmd, job_name):
    query_jd, query_warns = ctx.obj["slurmise"].predict(cmd, job_name)
    click.echo(f"Predicted runtime: {query_jd.runtime}")
    click.echo(f"Predicted memory: {query_jd.memory}")
    if query_warns:
        click.echo(click.style("Warnings:", fg="yellow"), err=True, color="red")
        for warn in query_warns:
            click.echo(f"  {warn}", err=True)


@main.command()
@click.option("--job-name", type=str, required=True, help="Name of the job")
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
def raw_predict(ctx, job_name, numerical, categorical, cmd):
    """predict a job"""
    # NOTE hack to get JSON parsing working with click who is too eager to try and process the args
    categorical = json.loads("{" + categorical + "}") if categorical else {}
    numerical = json.loads("{" + numerical + "}") if numerical else {}

    jd = job_data.JobData(
        job_name=job_name,
        numerical=numerical,
        categorical=categorical,
        cmd=cmd,
    )

    query_jd, query_warns = ctx.obj["slurmise"].raw_predict(jd)
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
    ctx.obj["slurmise"].update_model(cmd, job_name)


@main.command()
@click.pass_context
def update_all(ctx):
    ctx.obj["slurmise"].update_all_models()
