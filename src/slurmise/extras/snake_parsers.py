from abc import ABC, abstractmethod
import inspect
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from slurmise.job_data import JobData

import numpy as np

from slurmise.job_data import JobData
from snakemake.path_modifier import PathModifier


class ResourceFunction(Protocol):
    def __call__(self, rule: Any, wildcards: Any, input: Any) -> Any: ...


class SnakemakeAdapter(ABC):
    '''Abstract Adapter class for different version of snakemake.

    Most implemented methods are for version 9.'''
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
        pass


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
