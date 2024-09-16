from typing import Any, Optional
import h5py
import contextlib
import numpy as np


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

    def record(
        self,
        job_name: str,
        variables: dict[str, Any],
        params: Optional[dict[str, Any]] = None,
        slurm_id: Optional[str] = None,
    ) -> None:
        """
        variables: {"runtime":number of minutes,
                    "memory": number of MBs}
        """
        table_name = "/".join([job_name, slurm_id])
        table = self.db.require_group(name=table_name)
        for var, value in variables.items():
            val = np.asarray(value)
            _ = table.create_dataset(name=var, shape=val.shape, data=val)

        return

    def query(
        self, job_name: str, params: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        job_group = self.db[job_name]
        result = []
        for _, slurm_data in job_group.items():
            result.append({key: value[()] for key, value in slurm_data.items()})

        return result

    # def delete(self, **kwargs):
    #     pass
    #
    # def clear(**kwargs):
    #     pass
    #
    # def cache_fit(ResourceFit):
    #     pass
    #
