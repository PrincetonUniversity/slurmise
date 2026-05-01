import json
import shutil
from pathlib import Path

from snakemake.logging import logger
from snakemake.path_modifier import PathModifier
from snakemake.workflow import Workflow

from slurmise.api import Slurmise
from slurmise.extras import snake_parsers
from slurmise.job_data import JobData
from slurmise.job_parse.file_parsers import FileMD5


def patch_snakemake_workflow(
    slurmise: Slurmise,
    workflow: Workflow,
    rules: dict[str, dict],
    benchmark_dir: str | Path = "slurmise/benchmarks",
    keep_benchmarks: bool = False,
    record_benchmarks: bool = True,
):
    benchmark_dir = Path(benchmark_dir)

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

            job_data = JobData(
                job_name=benchmark_data["rule_name"],
                slurm_id=md5_parser.parse_file(file),
                categories=slurmise_data["categories"],
                numerics=slurmise_data["numerics"],
                runtime=runtime,
                memory=memory,
            )

            slurmise.raw_record(job_data, processed_data=True)
        if not keep_benchmarks:
            shutil.rmtree(benchmark_dir)

    workflow.onsuccess(onsuccess_slurmise_update)

    if record_benchmarks:
        # force extended benchmark recording
        workflow.output_settings.benchmark_extended = True

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

            exp = variables.get("SLURMISE_attempt_exp", SLURMISE_DEFAULTS["attempt_exp"])
            scale = variables.get(
                f"SLURMISE_{resource}_scale",
                SLURMISE_DEFAULTS[f"{resource}_scale"],
            )

            return scale * getattr(job_data, resource) * attempt**exp

        return slurmise_predict

    for rule_name, variables in rules.items():
        rule = workflow.get_rule(rule_name)

        thread_scaling = variables.get("SLURMISE_thread_scaling", None)
        if isinstance(thread_scaling, (int, float)):
            thread_scaling = snake_parsers.ThreadScaler(memory_per_thread=thread_scaling)
        variables["SLURMISE_thread_scaling"] = thread_scaling

        if record_benchmarks:
            # set benchmark to record stats
            if rule.benchmark is not None:
                raise ValueError(f"Slurmise needs to set benchmark locations, remove benchmark for rule {rule.name}.")

            old_modifier = rule.benchmark_modifier
            if old_modifier is None:
                rule.benchmark_modifier = PathModifier(
                    prefix=None,
                    replace_prefix=None,
                    workflow=workflow,
                )

            # wc1:val1~wc2:val2.jsonl
            if len(rule.wildcard_names) == 0:
                benchmark_name = f"{rule.name}.jsonl"
            else:
                benchmark_name = "~".join(f"{wc}:{{{wc}}}" for wc in sorted(rule.wildcard_names)) + ".jsonl"

            rule.benchmark = benchmark_dir / rule.name / benchmark_name

            rule.benchmark_modifier = old_modifier
            # get the slurmise parsed data for recroding in the benchmark file
            rule.params.update({"slurmise_data": make_predictor(variables, rule, "logging")})

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
