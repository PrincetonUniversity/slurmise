import json
import shutil
from pathlib import Path
from packaging import version

import snakemake
from snakemake.logging import logger
from snakemake.workflow import Workflow

from slurmise.api import Slurmise
from slurmise.extras import snake_parsers
from slurmise.job_data import JobData
from slurmise.job_parse.file_parsers import FileMD5


def patch_snakemake_workflow(
    slurmise: Slurmise,
    workflow: Workflow,
    adapter: snake_parsers.SnakemakeAdapter,
    rules: list | None = None,
):
    extras = slurmise.configuration.extras.get('snakemake', {})
    benchmark_dir = Path(extras.get('benchmark_dir', 'slurmise/benchmarks'))
    record_benchmarks = extras.get('record_benchmarks', True)

    original_onstart = workflow._onstart

    def onstart_slurmise_update(log):
        original_onstart(log)
        logger.info("SLURMISE: Updating all models")
        slurmise.update_all_models()

    workflow.onstart(onstart_slurmise_update)

    original_onsuccess = workflow._onsuccess

    def onsuccess_slurmise_update(log):
        original_onsuccess(log)
        if not record_benchmarks:
            logger.info("SLURMISE: Skipping recording completed jobs")
            return
        logger.info("SLURMISE: Recording completed jobs")
        # TODO: make adapter function
        md5_parser = FileMD5()
        for file in benchmark_dir.rglob("*.jsonl"):
            benchmark_data = json.loads(file.read_text())
            slurmise_data = json.loads(benchmark_data["params"]["slurmise_data"])
            try:
                runtime = (float(benchmark_data["s"]) / 60,)
            except ValueError:
                runtime = 0
            try:
                memory = (float(benchmark_data["max_rss"]),)
            except ValueError:
                memory = 0

            # if a value is a thread, update it to true value
            slurmise_data = _correct_threads(slurmise_data, benchmark_data)

            try:
                runtime = (float(benchmark_data["s"]) / 60,)
            except ValueError:
                runtime = 0
            try:
                memory = (float(benchmark_data["max_rss"]),)
            except ValueError:
                memory = 0

            job_data = JobData(
                job_name=benchmark_data["rule_name"],
                slurm_id=md5_parser.parse_file(file),
                categories=slurmise_data["categories"],
                numerics=slurmise_data["numerics"],
                runtime=runtime,
                memory=memory,
            )

            slurmise.raw_record(job_data, processed_data=True)
        if not extras.get('keep_benchmarks', False):
            shutil.rmtree(benchmark_dir)

    workflow.onsuccess(onsuccess_slurmise_update)

    if record_benchmarks:
        # force extended benchmark recording
        adapter.extend_benchmark(workflow)

    # TODO: make adapter function or accept a file to write to?
    def make_predictor(variables, rule, resource):
        def slurmise_predict(wildcards, input, attempt=1):
            vars = {
                name: func(rule, wildcards, input)
                for name, func in variables.items()
                if not name.startswith("SLURMISE")
            }
            job_data = slurmise.job_data_from_dict(vars, rule.name)
            if resource == "logging":
                # if we are recording threads need to mark in benchmark file
                for name, func in variables.items():
                    if name.startswith("SLURMISE"):
                        continue
                    if func.__name__ == "get_threads":
                        # update name to flag as thread
                        job_data = _mark_threads(job_data, name)

                job_data_variables = {
                    "categories": job_data.categories,
                    "numerics": job_data.numerics,
                }
                return json.dumps(job_data_variables)

            job_data = slurmise.raw_predict(job_data)[0]

            # TODO: add to rule configuration
            exp = variables.get("SLURMISE_attempt_exp", 1)

            return getattr(job_data, resource) * attempt**exp

        return slurmise_predict

    if rules is None:
        rules = slurmise.configuration.jobs.keys()
        # TODO: handle extra rules in slurmise

    for rule_name in rules:
        rule = workflow.get_rule(rule_name)
        variables = adapter.build_variables(
            slurmise.configuration.get_sources(rule_name),
        )

        if record_benchmarks:
            # set benchmark to record stats
            adapter.record_benchmark(rule, workflow, benchmark_dir, make_predictor(variables, rule, "logging"))

        rule.resources["mem_mb"] = make_predictor(variables, rule, "memory")
        rule.resources["runtime"] = make_predictor(variables, rule, "runtime")


def _mark_threads(job_data, variable_name):
    if variable_name in job_data.categories:
        job_data.categories[f"SLURMISETHREAD_{variable_name}"] = job_data.categories[variable_name]
        job_data.categories.pop(variable_name)
    if variable_name in job_data.numerics:
        job_data.numerics[f"SLURMISETHREAD_{variable_name}"] = job_data.numerics[variable_name]
        job_data.numerics.pop(variable_name)
    return job_data


def _correct_threads(slurmise_data, benchmark_data):
    result = {}
    for key, values in slurmise_data.items():
        result[key] = {}
        for name, value in values.items():
            if name.startswith("SLURMISETHREAD"):
                name = name.removeprefix("SLURMISETHREAD_")
                value = benchmark_data["threads"]
            result[key][name] = value

    return result

def _make_patch(adapter: snake_parsers.SnakemakeAdapter):
    def patch(self, workflow: Workflow):
        return patch_snakemake_workflow(self, workflow, adapter)

    return patch

patching_fncs = {
    7: _make_patch(adapter=snake_parsers.SnakemakeV7()),
    8: _make_patch(adapter=snake_parsers.SnakemakeV8()),
    9: _make_patch(adapter=snake_parsers.SnakemakeV9()),
}
snakemake_version = version.parse(snakemake.__version__)
if snakemake_version.major < 7:
    raise ValueError("Slurmise only supports snakemake>=7.0")

elif snakemake_version.major not in patching_fncs:
    raise ValueError(f"Slurmise does not support snakemake version {snakemake_version}")

else:
    logger.info(f"SLURMISE: detected snakemake v{snakemake_version}")
    Slurmise.register_patch(patching_fncs[snakemake_version.major])
