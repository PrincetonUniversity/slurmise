from typing import Any, Optional
import h5py
import contextlib
import numpy as np
from slurmise.job_data import JobData


class JobDatabase():
    """
    This class creates the database to store job information.
    It saves the database in HDF5 file.
    """
    def __init__(self, db_file):
        """
        The DB file is and HDF5 file.
        Use **get_database** and a context manager to have the file automatically
        closed.
        """
        self.db = h5py.File(db_file, "a")

    def _close(self):
        self.db.close()

    @staticmethod
    @contextlib.contextmanager
    def get_database(db_file):
        db = JobDatabase(db_file)
        try:
            yield db
        finally:
            db._close()

    def record(self, job_data: JobData) -> None:
        """
        It records JobData information in the database. A tree is created based
        on the job name, categorical values and slurm id. The leaves of the tree
        are the memory, runtime and numericals of the JobData.
        """
        table_name = f"/{job_data.job_name}"
        for key in sorted(job_data.categorical.keys()):
            table_name += f"/{key}={job_data.categorical[key]}"
        table_name += f"/{job_data.slurm_id}"
        table = self.db.require_group(name=table_name)

        if job_data.memory is not None:
            val = np.asarray(job_data.memory)
            _ = table.create_dataset(name='memory', shape=val.shape, data=val)

        if job_data.runtime is not None:
            val = np.asarray(job_data.runtime)
            _ = table.create_dataset(name='runtime', shape=val.shape, data=val)

        for var, value in job_data.numerical.items():
            val = np.asarray(value)
            _ = table.create_dataset(name=var, shape=val.shape, data=val)

        return

    def query(self, job_data: JobData) -> list[JobData]:
        """
        Query returns a list of JobData objects based on the requested JobData.
        The returned jobs match the query JobData's job name and categoricals.

        Note: It does not decent into all child categories, only the highest matching leaves

        """
        group_name = f"/{job_data.job_name}"
        for key in sorted(job_data.categorical.keys()):
            group_name += f"/{key}={job_data.categorical[key]}"

        job_group = self.db.get(group_name, default={})
        result = []
        for slurm_id, slurm_data in job_group.items():
            if JobDatabase.is_slurm_job(slurm_data):
                result.append(JobData.from_dataset(
                    job_name=job_data.job_name,
                    slurm_id=slurm_id,
                    categorical=job_data.categorical,
                    dataset=slurm_data,
                ))

        return result

    def update(self, **kargs):
        raise NotImplementedError("Later feature")

    def delete(self, job_data: JobData, delete_all: bool = False) -> None:

        group_name = f"/{job_data.job_name}"
        for key in sorted(job_data.categorical.keys()):
            group_name += f"/{key}={job_data.categorical[key]}"

        if group_name in self.db:
            if delete_all:
                del self.db[group_name]
            else:
                # Traverse and delete only Datasets
                job_group = self.db.get(group_name, default={})
                for slurm_id, slurm_data in job_group.items():
                    if JobDatabase.is_slurm_job(slurm_data):
                        del job_group[slurm_id]

    @staticmethod
    def is_dataset(f):
        return type(f) == h5py._hl.dataset.Dataset

    @staticmethod
    def is_slurm_job(f):
        if len(f) == 0:  # group contains no values
            return True
        first_element = f[list(f.keys())[0]]
        # group contains a dataset
        return JobDatabase.is_dataset(first_element)

    #
    # def clear(self):
    # raise NotImplementedError("Empting the DB is not yet supported")
    #     pass
    #
    # def cache_fit(ResourceFit):
    #     pass
    #
