from typing import Any, Optional
import h5py
import contextlib
import numpy as np
from slurmise.job_data import JobData


class JobDatabase():
    def __init__(self, db_file):
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
        variables: {"runtime":number of minutes,
                    "memory": number of MBs}
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
        # TODO add categorical hierarchy
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

    def delete(self, job_data: JobData) -> None:
        del self.db[job_data.job_name]

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
