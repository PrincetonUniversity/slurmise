import inspect
import numpy as np
from slurmise.job_data import JobData
from dataclasses import dataclass, replace


def input(index: str | int | None=None):
    def get_input(rule, wildcards, input):
        if index is None:
            return input[0]
        return input[index]

    return get_input

def wildcards(name: str):
    def get_wildcard(rule, wildcards, input):
        return wildcards[name]

    return get_wildcard

def threads():
    def get_threads(rule, wildcards, input):
        threads = rule.resources['_cores']
        # not a function
        if not callable(threads):
            return threads
        call_params = inspect.signature(threads).parameters
        arg_list = [wildcards]
        if 'input' in call_params:
            arg_list.append(input)
        return threads(*arg_list)

    return get_threads

def params(name: str):
    def get_params(rule, wildcards, input):
        param = rule.params[name]
        # not a function
        if not callable(param):
            return param
        call_params = inspect.signature(param).parameters
        arg_list = [wildcards]
        if any(input_type in call_params for input_type in (
            'output', 'threads', 'resources')):
            message = (
                f'Cannot use param {name!r} in slurmise.  '
                'Input functions may only depend on wildcards or input.'
            )
            raise ValueError(message)
        if 'input' in call_params:
            arg_list.append(input)
        return param(*arg_list)

    return get_params

@dataclass()
class ThreadScaler:
    memory_per_thread: float
    runtime_overhead: float = 1
    memory_overhead: float = 1
    thread_range: tuple[int, int] = (1,20)

    def __post_init__(self):
        if self.runtime_overhead < 1:
            self.runtime_overhead = 1
        if self.memory_overhead < 1:
            self.memory_overhead = 1

    def update_job_data(self, job_data: JobData, current_threads: int) -> tuple[JobData, int]:
        '''Update the provided job data to reflect scaling threads.

        :arguments:
            :job_data: The job to update.
            :current_threads: The current request for threads for this job.

        :returns:
            The job data memory and time will be updated to reflect any change
            in the number of threads. If overheads are equal to 1, this is a
            simple linear scaling based on the memory_per_thread.  Otherwise,
            the overhead is factored in as well.  The returned thread value is
            clipped to the range of the scaler object.
        '''
        # get single thread estimates
        memory = job_data.memory
        runtime = job_data.runtime * current_threads 

        threads = np.ceil(np.clip(memory / self.memory_per_thread, *self.thread_range))

        if self.runtime_overhead >= 2:  # take as an offset
            runtime = int(runtime / threads + (threads - 1) * self.runtime_overhead)
        else:  # a fractional scale, e.g. 1.2 is 20% more per thread
            runtime = int(runtime / threads * self.runtime_overhead ** (threads-1))

        if self.runtime_overhead >= 2:  # take as an offset
            memory = int(memory + (threads - 1)* self.memory_overhead)
        else:  # a fractional scale
            memory = int(memory * self.memory_overhead ** (threads-1))

        return replace(job_data, runtime=runtime, memory=memory), int(threads)
