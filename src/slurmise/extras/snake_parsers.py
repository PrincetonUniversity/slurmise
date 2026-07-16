from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from slurmise.api import Slurmise
from slurmise.job_data import JobData
from slurmise.job_parse.file_parsers import FileMD5

from snakemake.logging import logger
from snakemake.path_modifier import PathModifier

if TYPE_CHECKING:
    from snakemake.workflow import Workflow


class ResourceFunction(Protocol):
    def __call__(self, rule: Any, wildcards: Any, input: Any) -> Any: ...


class SnakemakeAdapter(ABC):
    '''Abstract Adapter class for different version of snakemake.

    Most implemented methods are for version 9.'''

    def patch_snakemake_workflow(
        self,
        slurmise: Slurmise,
        workflow: Workflow,
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
            md5_parser = FileMD5()
            for file, benchmark_data in self.iter_benchmark_data(benchmark_dir):
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
            if not extras.get('keep_benchmarks', False):
                shutil.rmtree(benchmark_dir)

        workflow.onsuccess(onsuccess_slurmise_update)

        if record_benchmarks:
            # force extended benchmark recording
            self.extend_benchmark(workflow)

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
            variables = self.build_variables(
                slurmise.configuration.get_sources(rule_name),
            )

            if record_benchmarks:
                # set benchmark to record stats
                self.record_benchmark(rule, workflow, benchmark_dir, make_predictor(variables, rule, "logging"))

            rule.resources["mem_mb"] = make_predictor(variables, rule, "memory")
            rule.resources["runtime"] = make_predictor(variables, rule, "runtime")

    def iter_benchmark_data(self, benchmark_dir: Path):
        """Yield (file_path, benchmark_data_dict) for each benchmark. Default for V8/V9."""
        for file in benchmark_dir.rglob("*.jsonl"):
            yield file, json.loads(file.read_text())

    def build_variables(self, sources):
        result = {}
        for variable_name, source in sources.items():
            if not isinstance(source, tuple):
                source, key = source, None
            else:
                source, key = source

            if source == "input":
                result[variable_name] = self.input(key)
            elif source == "wildcards":
                if key is None:
                    msg = f"The wildcards source for {variable_name} requires a key entry"
                    raise ValueError(msg)
                result[variable_name] = self.wildcards(key)
            elif source == "threads":
                result[variable_name] = self.threads()
            elif source == "params":
                if key is None:
                    msg = f"The params source for {variable_name} requires a key entry"
                    raise ValueError(msg)
                result[variable_name] = self.params(key)

        return result

    def params(self, name: str) -> ResourceFunction:
        def get_params(rule, wildcards, input):
            param = rule.params[name]
            # is a value, return it directly
            if not callable(param):
                return param
            else:  # is a function, need to invoke it
                call_params = inspect.signature(param).parameters
                arg_list = [wildcards]
                # we support fewer options than snakemake, prevent circular dependencies
                if any(input_type in call_params for input_type in ("output", "threads", "resources")):
                    message = (
                        f"Cannot use param {name!r} in slurmise.  Input functions may only depend on wildcards or input."
                    )
                    raise ValueError(message)
                # if the param function also takes snakemake input add it to the call
                if "input" in call_params:
                    arg_list.append(input)
                return param(*arg_list)
        return get_params

    def input(self, index: str | int | None = None) -> ResourceFunction:
        def get_input(rule, wildcards, input):
            if index is None:
                return input[0]
            return input[index]

        return get_input

    def wildcards(self, name: str) -> ResourceFunction:
        def get_wildcard(rule, wildcards, input):
            return wildcards[name]

        return get_wildcard

    def threads(self) -> ResourceFunction:
        def get_threads(rule, wildcards, input):
            threads = rule.resources["_cores"]
            # is a value, return it directly
            if not threads.is_evaluable():
                return threads.value
            else:  # is a function, need to invoke it
                # get the names of parameters to the threads function
                call_params = inspect.signature(threads._evaluator).parameters
                arg_list = [wildcards]  # wildcards are always a parameter
                # if the threads function also takes snakemake input add it to the call
                if "input" in call_params:
                    arg_list.append(input)
                # invoke the threads function
                return threads.evaluate(*arg_list).value

        return get_threads

    def extend_benchmark(self, workflow) -> None:
        workflow.output_settings.benchmark_extended = True

    def record_benchmark(self, rule, workflow, benchmark_dir, logging_predictor):
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
        rule.params.update({"slurmise_data": logging_predictor})


class SnakemakeV7(SnakemakeAdapter):
    def threads(self) -> ResourceFunction:
        def get_threads(rule, wildcards, input):
            threads = rule.resources["_cores"]
            # is a value, return it directly
            if not callable(threads):
                return threads
            else:  # is a function, need to invoke it
                # get the names of parameters to the threads function
                call_params = inspect.signature(threads).parameters
                arg_list = [wildcards]  # wildcards are always a parameter
                # if the threads function also takes snakemake input add it to the call
                if "input" in call_params:
                    arg_list.append(input)
                # invoke the threads function
                return threads(*arg_list)

        return get_threads

    def extend_benchmark(self, workflow) -> None:
        # not supported in V7
        pass

    def record_benchmark(self, rule, workflow, benchmark_dir, logging_predictor):
        if rule.benchmark is not None:
            raise ValueError(f"Slurmise needs to set benchmark locations, remove benchmark for rule {rule.name}.")

        old_modifier = rule.benchmark_modifier
        if old_modifier is None:
            rule.benchmark_modifier = PathModifier(
                prefix=None,
                replace_prefix=None,
                workflow=workflow,
            )

        # V7 writes TSV benchmarks; we pair them with .slurmise.json companion files
        if len(rule.wildcard_names) == 0:
            benchmark_stem = rule.name
        else:
            benchmark_stem = "~".join(f"{wc}:{{{wc}}}" for wc in sorted(rule.wildcard_names))

        rule.benchmark = benchmark_dir / rule.name / f"{benchmark_stem}.tsv"
        rule.benchmark_modifier = old_modifier

        wildcard_names = sorted(rule.wildcard_names)
        rule_name = rule.name
        companion_dir = benchmark_dir / rule.name

        def v7_slurmise_data(wildcards, input, threads=None):
            # Snakemake v7 passes job.resources._cores (the post-cap actual thread count)
            # to any resource callable that declares a `threads` parameter.
            # Resources are always evaluated (unlike params), so this fires for every job.
            slurmise_json = logging_predictor(wildcards, input)

            if len(wildcard_names) == 0:
                companion_stem = rule_name
            else:
                companion_stem = "~".join(f"{wc}:{wildcards[wc]}" for wc in wildcard_names)

            companion_path = companion_dir / f"{companion_stem}.slurmise.json"
            companion_path.parent.mkdir(parents=True, exist_ok=True)

            companion_data = {
                "slurmise_data": slurmise_json,
                "threads": threads,
            }
            companion_path.write_text(json.dumps(companion_data))

            return 0  # must return int/str for a resource

        # Register as a resource so snakemake always evaluates it (params are lazy)
        rule.resources["_slurmise_log"] = v7_slurmise_data

    def iter_benchmark_data(self, benchmark_dir: Path):
        """V7 benchmark data: TSV files paired with .slurmise.json companions.

        Yields (jsonl_path, benchmark_dict) where jsonl_path is a .jsonl file
        written alongside the companion so callers can read JSON at a stable path.
        """
        for companion_file in benchmark_dir.rglob("*.slurmise.json"):
            stem = companion_file.name.removesuffix(".slurmise.json")
            tsv_file = companion_file.with_name(stem + ".tsv")
            if not tsv_file.exists():
                continue

            companion_data = json.loads(companion_file.read_text())

            with open(tsv_file) as f:
                header = f.readline().strip().split('\t')
                values = f.readline().strip().split('\t')
            tsv_data = dict(zip(header, values))

            # V7 TSV uses "-" for missing values; normalize to "NA"
            def to_na(val):
                return "NA" if val == "-" else val

            jsonl_content = {
                "s": to_na(tsv_data.get("s", "-")),
                "max_rss": to_na(tsv_data.get("max_rss", "-")),
            }
            jsonl_file = companion_file.with_name(stem + ".jsonl")
            jsonl_file.write_text(json.dumps(jsonl_content))

            benchmark_dict = {
                **jsonl_content,
                "rule_name": companion_file.parent.name,
                "threads": companion_data.get("threads"),
                "params": {"slurmise_data": companion_data["slurmise_data"]},
            }
            yield jsonl_file, benchmark_dict


class SnakemakeV8(SnakemakeAdapter):
    def threads(self) -> ResourceFunction:
        def get_threads(rule, wildcards, input):
            threads = rule.resources["_cores"]
            # is a value, return it directly
            if not callable(threads):
                return threads
            else:  # is a function, need to invoke it
                # get the names of parameters to the threads function
                call_params = inspect.signature(threads).parameters
                arg_list = [wildcards]  # wildcards are always a parameter
                # if the threads function also takes snakemake input add it to the call
                if "input" in call_params:
                    arg_list.append(input)
                # invoke the threads function
                return threads(*arg_list)

        return get_threads


class SnakemakeV9(SnakemakeAdapter):
    pass


def _mark_threads(job_data, variable_name):
    if variable_name in job_data.categories:
        job_data.categories[f"SLURMISETHREAD_{variable_name}"] = job_data.categories[variable_name]
        job_data.categories.pop(variable_name)
    if variable_name in job_data.numerics:
        job_data.numerics[f"SLURMISETHREAD_{variable_name}"] = job_data.numerics[variable_name]
        job_data.numerics.pop(variable_name)
    return job_data


def _correct_threads(slurmise_data, benchmark_data):
    actual_threads = benchmark_data.get("threads")
    result = {}
    for key, values in slurmise_data.items():
        result[key] = {}
        for name, value in values.items():
            if name.startswith("SLURMISETHREAD"):
                name = name.removeprefix("SLURMISETHREAD_")
                if actual_threads is not None:
                    value = actual_threads
            result[key][name] = value

    return result


@dataclass()
class ThreadScaler:
    memory_per_thread: float
    runtime_overhead: float = 1
    memory_overhead: float = 1
    thread_range: tuple[int, int] = (1, 20)

    def __post_init__(self):
        if self.runtime_overhead < 1:
            self.runtime_overhead = 1
        if self.memory_overhead < 1:
            self.memory_overhead = 1

    def update_job_data(self, job_data: JobData, current_threads: int) -> tuple[JobData, int]:
        """Update the provided job data to reflect scaling threads.

        :arguments:
            :job_data: The job to update.
            :current_threads: The current request for threads for this job.

        :returns:
            The job data memory and time will be updated to reflect any change
            in the number of threads. If overheads are equal to 1, this is a
            simple linear scaling based on the memory_per_thread.  Otherwise,
            the overhead is factored in as well.  The returned thread value is
            clipped to the range of the scaler object.
        """
        # get single thread estimates
        memory = job_data.memory
        runtime = job_data.runtime * current_threads

        threads = np.ceil(np.clip(memory / self.memory_per_thread, *self.thread_range))

        if self.runtime_overhead >= 2:  # take as an offset
            runtime = int(runtime / threads + (threads - 1) * self.runtime_overhead)
        else:  # a fractional scale, e.g. 1.2 is 20% more per thread
            runtime = int(runtime / threads * self.runtime_overhead ** (threads - 1))

        if self.runtime_overhead >= 2:  # take as an offset
            memory = int(memory + (threads - 1) * self.memory_overhead)
        else:  # a fractional scale
            memory = int(memory * self.memory_overhead ** (threads - 1))

        return replace(job_data, runtime=runtime, memory=memory), int(threads)
