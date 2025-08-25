from __future__ import annotations

import ast
import pathlib
import warnings
from dataclasses import astuple, dataclass, field

import h5py
import numpy as np


def array_safe_eq(a, b) -> bool:
    """
    Check if a and b are equal, even if they are numpy arrays.
    When a and be are dictionaries call recursively for all key, value pairs.
    """

    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(array_safe_eq(a[key], b[key]) for key in a.keys())
    try:
        return a == b
    except TypeError:  # pragma: no cover
        return NotImplemented


def dc_eq(dc1, dc2) -> bool:
    """
    Checks if two dataclasses which hold numpy arrays are equal
    """

    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:  # pragma: no cover
        return NotImplemented  # better than False
    t1 = astuple(dc1)
    t2 = astuple(dc2)
    return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2, strict=False))


@dataclass(eq=False)
class JobData:
    """
    Jobdata class holds the information of a unique slurm job.

    :arguments:

        :job_name: The unique command name to execute under slurm.
        :slurm_id: The slurm job id assigned by the sceduler for a job run.
        :categorical: the CLI parameters of this job. This has parameter that affect the performance of the job and are fit seperately.
        :numerical: These are parameters that are used as the free variables for fits, such input size, number of iterations etc.
        :memory: The maximum amount of memory in MBs this job used.
        :runtime: The time this job needed to complete in minutes.
    """

    job_name: str
    slurm_id: str | None = None
    categorical: dict = field(default_factory=dict)
    numerical: dict = field(default_factory=dict)
    memory: int | None = None  # in MBs
    runtime: int | None = None  # in minutes
    cmd: str | None = None  # TODO: NOT STORED OR RETURNED

    @staticmethod
    def from_dataset(job_name: str, slurm_id: str, dataset: h5py.Dataset, categorical: dict) -> JobData:
        """
        This method creates a JobData object from a HDF5 dataset that describes a job.
        :arguments:

            :job_name: The unique command name to execute under slurm.
            :slurm_id: The slurm job id assigned by the sceduler for a job run.
            :dataset: The HDF5 dataset used to populate numerical, memory and runtime information of the job.
        """

        runtime = dataset.get("runtime", None)
        if runtime is not None:
            runtime = runtime[()]
        memory = dataset.get("memory", None)
        if memory is not None:
            memory = memory[()]
        numerical = {key: value[()] for key, value in dataset.items() if key not in ("runtime", "memory")}
        categorical = dict(**categorical)

        return JobData(
            job_name=job_name,
            slurm_id=slurm_id,
            numerical=numerical,
            categorical=categorical,
            memory=memory,
            runtime=runtime,
        )

    @staticmethod
    def from_snakemake_benchmark_file(benchmark_path: pathlib.Path) -> JobData:
        """
        This method creates a JobData object from a benchmark file created by Snakemake benchmark: directive.
        :arguments:

            :benchmark_path: The path to the Snakemake benchmark file.
        """
        with benchmark_path.open() as f:
            lines = f.readlines()

        # Parse the header
        header = lines[0].strip().split("\t")
        data = lines[1].strip().split("\t")

        # Map header to data
        job_data = {key: value for key, value in zip(header, data)}

        # Convert the wildcards and params into categorical and numerical
        wildcards = ast.literal_eval(job_data.get("wildcards", "{}"))
        params = ast.literal_eval(job_data.get("params", "{}"))
        shared_keys = set(wildcards.keys()) & set(params.keys())

        if shared_keys:
            warnings.warn(f"Shared keys found in Snakemake wildcards and params: {shared_keys}")

        categorical = {}
        numerical = {}
        for k,v in (wildcards | params).items():
            try:
                numerical[k] = float(v)
            except ValueError:
                categorical[k] = v

        job_data["categorical"] = categorical
        job_data["numerical"] = numerical

        # Convert max_rss (in MBs) and cpu_time (seconds) to integer MBs and minutes
        job_data["max_rss"] = int(float(job_data.get("max_rss", 0)))
        job_data["cpu_time"] = int(float(job_data.get("cpu_time", 0))) // 60

        return JobData(
            job_name=job_data.get("rule_name"),
            slurm_id=job_data.get("jobid"),
            categorical=job_data.get("categorical"),
            numerical=job_data.get("numerical"),
            memory=job_data.get("max_rss"),
            runtime=job_data.get("cpu_time"),
        )

    def __eq__(self, other):
        return dc_eq(self, other)
