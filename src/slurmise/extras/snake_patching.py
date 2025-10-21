from slurmise.api import Slurmise
from slurmise.job_data import JobData
from slurmise.job_parse.file_parsers import FileMD5
from slurmise.extras import snake_parsers

from snakemake.path_modifier import PathModifier
from snakemake.logging import logger

import shutil
from pathlib import Path
import json


def patch_snakemake_workflow(
    slurmise: Slurmise,
    workflow,
    rules: list[str],
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

            job_data = JobData(
                job_name=benchmark_data["rule_name"],
                slurm_id=md5_parser.parse_file(file),
                categorical=slurmise_data["categorical"],
                numerical=slurmise_data["numerical"],
                runtime=float(benchmark_data["s"]) / 60,
                memory=float(benchmark_data["max_rss"]),
            )

            slurmise.raw_record(job_data, processed_data=True)
        if not keep_benchmarks:
            shutil.rmtree(benchmark_dir)

    workflow.onsuccess(onsuccess_slurmise_update)

    if record_benchmarks:
        # force extended benchmark recording
        workflow.output_settings.benchmark_extended = True

    def make_predictor(variables, rule, resource):
        # TODO: chaching?
        def slurmise_predict(wildcards, input, attempt=1):
            vars = {
                name: func(rule, wildcards, input)
                for name, func in variables.items()
                if not name.startswith("SLURMISE")
            }
            job_data = slurmise.job_data_from_dict(vars, rule.name)
            if resource == "logging":
                return job_data.to_json()
            job_data = slurmise.raw_predict(job_data)[0]

            exp = variables.get("SLURMISE_attempt_exp", 1)
            scale = variables.get(f"SLURMISE_{resource}_scale", 1)

            # thread_scaling = variables['SLURMISE_thread_scaling']
            # if thread_scaling is not None:
            # TODO: need to decide how to handle threads in recording and
            # here in prediction...
            # threads = snake_parsers.threads()(rule, wildcards, input)

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
                raise ValueError(
                    "Slurmise needs to set benchmark locations, " f"remove benchmark for rule {rule.name}."
                )

            old_modifier = rule.benchmark_modifier
            if old_modifier is None:
                rule.benchmark_modifier = PathModifier(
                    prefix=None,
                    replace_prefix=None,
                    workflow=workflow,
                )

            # wc1:val1~wc2:val2.jsonl
            benchmark_name = "~".join(f"{wc}:{{{wc}}}" for wc in sorted(rule.wildcard_names)) + ".jsonl"

            rule.benchmark = benchmark_dir / rule.name / benchmark_name

            rule.benchmark_modifier = old_modifier
            # get the slurmise parsed data for recroding in the benchmark file
            rule.params.update({"slurmise_data": make_predictor(variables, rule, "logging")})

        rule.resources["mem_mb"] = make_predictor(variables, rule, "memory")
        rule.resources["runtime"] = make_predictor(variables, rule, "runtime")
