import contextlib
import dataclasses
import os
from typing import Any
import time

import h5py
import numpy as np

from slurmise import slurm
from slurmise.job_data import JobData


class JobDatabase:
    """
    This class creates the database to store job information.
    It saves the database in HDF5 file.
    """

    def __init__(self, db_file: str, max_retries: int = 5):
        """
        The DB file is and HDF5 file.
        Use **get_database** and a context manager to have the file automatically
        closed.
        """

        attempt = 0
        while True:
            attempt += 1
            try:
                self.db = h5py.File(db_file, "a")
                break
            except BlockingIOError:
                if attempt > max_retries:
                    raise
                time.sleep(attempt)

    def _close(self):
        self.db.close()

    @staticmethod
    @contextlib.contextmanager
    def get_database(db_file: str, max_retries: int = 5) -> "JobDatabase":
        """
        Use in context manager to automatically open and close db file.

        :arguments:

            :db_file: HDF5 file to use as database

        :yields:

            JobDatabase with opened db file

        :finally:

            Closes h5py database
        """

        db = JobDatabase(db_file, max_retries)
        try:
            yield db
        finally:
            db._close()

    def record(self, job_data: JobData, ignore_existing_job: bool = False) -> None:
        """
        It records JobData information in the database. A tree is created based
        on the job name, categorical values and slurm id. The leaves of the tree
        are the memory, runtime and numericals of the JobData.
        """
        table_name = JobDatabase.get_table_name(job_data)
        if ignore_existing_job and self.job_exists(job_data):
            return

        table = self.db.require_group(name=table_name)

        if job_data.memory is not None:
            val = np.asarray(job_data.memory)
            _ = table.create_dataset(name="memory", shape=val.shape, data=val)

        if job_data.runtime is not None:
            val = np.asarray(job_data.runtime)
            _ = table.create_dataset(name="runtime", shape=val.shape, data=val)

        for var, value in job_data.numerical.items():
            val = np.asarray(value)
            _ = table.create_dataset(name=var, shape=val.shape, data=val)

    def job_exists(self, job_data: JobData) -> bool:
        table_name = JobDatabase.get_table_name(job_data)
        return table_name in self.db

    def update(self, **kargs):
        raise NotImplementedError("Later feature")

    def query(self, job_data: JobData, update_missing: bool = False) -> list[JobData]:
        """
        Query returns a list of JobData objects based on the requested JobData.
        The returned jobs match the query JobData's job name and categoricals.
        `update_missing` will try to get maxRSS and elapsed from sacct if not found in the DB.

        Note: It does not decent into all child categories, only the highest matching leaves
        """
        group_name = JobDatabase.get_group_name(job_data)

        job_group = self.db.get(group_name, default={})
        result = []
        for slurm_id, slurm_data in job_group.items():
            if JobDatabase.is_slurm_job(slurm_data):
                result.append(
                    JobData.from_dataset(
                        job_name=job_data.job_name,
                        slurm_id=slurm_id,
                        categorical=job_data.categorical,
                        dataset=slurm_data,
                    )
                )

        if update_missing:
            result = self.update_missing_data(result)

        return result

    def delete(self, job_data: JobData, delete_all_children: bool = False) -> None:
        """
        Delete jobs with matching job name and categoricals.

        :arguments:

            :job_data: JobData object with name and categorical which should be removed.
            :delete_all_children: When true, will delete recursively any matching jobs
        """
        group_name = JobDatabase.get_group_name(job_data)

        if group_name in self.db:
            if delete_all_children:
                del self.db[group_name]
            else:
                # Traverse and delete only Datasets
                job_group = self.db.get(group_name, default={})
                for slurm_id, slurm_data in job_group.items():
                    if JobDatabase.is_slurm_job(slurm_data):
                        del job_group[slurm_id]

    def clear(self):
        raise NotImplementedError("Empting the DB is not yet supported")

    def record_fit(self, fit):
        raise NotImplementedError("Storing fits is not supported yet")

    def query_fit(self, fit):
        raise NotImplementedError("Storing fits is not supported yet")

    def update_missing_data(self, jobs: list[JobData]) -> list[JobData]:
        """
        Update missing mem and runtime for jobs with incomplete data in the db.
        Takes a list of JobData which was queried from the db, updates the db, and returns the updated job list.

        TODO: gather slurm_ids of jobs that need updating and do it in one call
        """
        updated_jobs = []
        for job in jobs:
            if job.memory is None or job.runtime is None:
                if "." in job.slurm_id:
                    slurm_id, step_id = job.slurm_id.split(".")
                else:
                    slurm_id, step_id = job.slurm_id, None
                job_info = slurm.parse_slurm_job_metadata(slurm_id = slurm_id, step_id = step_id)

                # job dataclass is immutable, so this creates a new object with the updated values
                # ternary's are to avoid updating if the value is already present which causes a "dataset already exists" error
                self.record(
                    dataclasses.replace(
                        job,
                        memory=job_info["max_rss"] if job.memory is None else None,
                        runtime=(
                            job_info["elapsed_seconds"] if job.runtime is None else None
                        ),
                        numerical={},
                    )
                )

                job = dataclasses.replace(
                    job,
                    memory=job_info["max_rss"] if job.memory is None else job.memory,
                    runtime=(
                        job_info["elapsed_seconds"]
                        if job.runtime is None
                        else job.runtime
                    ),
                )

            updated_jobs.append(job)

        return updated_jobs

    @staticmethod
    def get_table_name(job_data: JobData) -> str:
        table_name = JobDatabase.get_group_name(job_data)
        table_name += f"/{job_data.slurm_id}"
        return table_name

    @staticmethod
    def get_group_name(job_data: JobData) -> str:
        group_name = f"/{job_data.job_name}"
        for key in sorted(job_data.categorical.keys()):
            group_name += f"/{key}={job_data.categorical[key]}"
        return group_name

    @staticmethod
    def is_dataset(f: Any) -> bool:
        """
        Test if object is an h5py Dataset
        """
        return type(f) == h5py._hl.dataset.Dataset

    @staticmethod
    def is_slurm_job(f: Any) -> bool:
        """
        Test if object is non-empty or its first element is a Dataset.
        This is consistent with a slurm job
        """

        if len(f) == 0:  # group contains no values
            return True
        first_element = f[list(f.keys())[0]]
        # group contains a dataset
        return JobDatabase.is_dataset(first_element)

    def print(self):
        JobDatabase.print_hdf5(self.db)

    def iterate_database(self, update_missing: bool = False):
        """
        Yield key (query job) value (list of jobs) pairs of entire database.
        """
        for job_name in self.db.keys():
            entry = self.db[job_name]
            for categoricals, jobs in JobDatabase.iterate_jobs(entry):
                categoricals = dict(cat.split("=") for cat in categoricals)
                query = JobData(job_name=job_name, categorical=categoricals)
                jobs = [
                    JobData.from_dataset(
                        job_name=query.job_name,
                        slurm_id=slurm_id,
                        categorical=query.categorical,
                        dataset=slurm_data,
                    )
                    for slurm_id, slurm_data in jobs.items()
                ]
                if update_missing:
                    jobs = self.update_missing_data(jobs)

                yield query, jobs

    @staticmethod
    def iterate_jobs(h5py_obj, categoricals=None):
        if categoricals is None:
            categoricals = tuple()

        jobs = {}
        for key, entry in h5py_obj.items():
            if JobDatabase.is_slurm_job(entry):
                jobs[key] = entry
            else:
                yield from JobDatabase.iterate_jobs(entry, categoricals + (key,))
        yield categoricals, jobs

    @staticmethod
    def print_hdf5(
        h5py_obj, level=-1, print_full_name: bool = False, print_attrs: bool = True
    ) -> None:
        """Prints the name and shape of datasets in a H5py HDF5 file.

        Parameters
        ----------
        h5py_obj: [h5py.File, h5py.Group]
            the h5py.File or h5py.Group object
        level: int
            What level of the file tree you are in
        print_full_name
            If True, the full tree will be printed as the name, e.g. /group0/group1/group2/dataset: ...
            If False, only the current node will be printed, e.g. dataset:
        print_attrs
            If True: print all attributes in the file
        Returns
        -------
        None

        """

        def is_group(f):
            return type(f) == h5py._hl.group.Group

        def is_dataset(f):
            return type(f) == h5py._hl.dataset.Dataset

        def print_level(level, n_spaces=5) -> str:
            if level == -1:
                return ""
            prepend = "|" + " " * (n_spaces - 1)
            prepend *= level
            tree = "|" + "-" * (n_spaces - 2) + " "
            return prepend + tree

        for key in h5py_obj.keys():
            entry = h5py_obj[key]
            name = entry.name if print_full_name else os.path.basename(entry.name)
            if is_group(entry):
                print(f"{print_level(level)}{name}")
                JobDatabase.print_hdf5(
                    entry, level + 1, print_full_name=print_full_name
                )
            elif is_dataset(entry):
                shape = entry.shape
                dtype = entry.dtype
                print(f"{print_level(level)}{name}: {shape} {dtype} {entry[()]}")
        if level == -1:
            if print_attrs:
                print("attrs: ")
                for key, value in h5py_obj.attrs.items():
                    print(f" {key}: {value}")
