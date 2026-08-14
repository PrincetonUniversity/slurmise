from __future__ import annotations

import numpy as np

from slurmise import job_database, slurm
from slurmise.config import SlurmiseConfiguration


class Slurmise:
    """
    API class for interacting with slurmise.
    """

    def __init__(self, toml_path=None):
        self.toml_path = toml_path
        self.configuration = SlurmiseConfiguration(toml_path)

    def record(
        self,
        cmd: str,
        job_name: str | None = None,
        slurm_id: str | None = None,
        step_id: str | None = None,
    ):
        parsed_jd = self.configuration.parse_job_cmd(
            cmd=cmd,
            job_name=job_name,
            slurm_id=slurm_id,
            step_id=step_id,
        )
        self.raw_record(parsed_jd)

    def dry_parse(
        self,
        cmd: str,
        job_name: str | None = None,
    ):
        return self.configuration.dry_parse(
            cmd=cmd,
            job_name=job_name,
        )

    def raw_record(self, job_data, processed_data=False):
        if not processed_data:
            job_data.slurm_id = slurm.resolve_job_id(job_data.slurm_id)
            slurm_id, step_id = slurm.split_job_id(job_data.slurm_id)

            metadata_json = slurm.parse_slurm_job_metadata(slurm_id=slurm_id, step_id=step_id)

            job_data.memory = metadata_json["max_rss"]
            job_data.runtime = metadata_json["elapsed_seconds"]

        with job_database.JobDatabase.get_database(self.configuration.db_filename) as database:
            database.record(job_data)

    def print(self):
        self.print_database(self.configuration.db_filename)

    @classmethod
    def print_database(cls, h5_path):
        """Print the contents of a slurmise HDF5 database.

        This does not require a toml configuration; only the path to the
        ``.h5`` database is needed.
        """
        with job_database.JobDatabase.get_database(h5_path) as database:
            database.print()

    def predict(self, cmd, job_name):
        query_jd = self.configuration.parse_job_cmd(cmd=cmd, job_name=job_name)

        return self.raw_predict(query_jd)

    def raw_predict(self, query_jd):
        query_jd = self.configuration.add_defaults(query_jd)
        model = self.configuration.get_model_class(query_jd.job_name)
        query_model = model.load(query=query_jd, path=self.configuration.slurmise_base_dir)
        query_jd, query_warns = query_model.predict(query_jd)
        query_jd = self.configuration.correct_minimum(query_jd)
        return query_jd, query_warns

    def update_model(self, cmd, job_name):
        query_jd = self.configuration.parse_job_cmd(cmd=cmd, job_name=job_name)
        with job_database.JobDatabase.get_database(self.configuration.db_filename) as database:
            jobs = database.query(query_jd)

        self._update_model(query_jd, jobs)

    def _update_model(self, query_jd, jobs):
        model_path = self.configuration.slurmise_base_dir
        model = self.configuration.get_model_class(query_jd.job_name)

        try:
            query_model = model.load(query=query_jd, path=model_path)
        except FileNotFoundError:
            query_model = model(query=query_jd, path=model_path)

        random_state = np.random.RandomState(42)
        query_model.fit(jobs, random_state=random_state)

        query_model.save()

    def update_all_models(self):
        with job_database.JobDatabase.get_database(self.configuration.db_filename) as database:
            for query_jd, jobs in database.iterate_database():
                self._update_model(query_jd, jobs)

    def job_data_from_dict(
        self,
        variables: dict,
        job_name: str,
        slurm_id: str | None = None,
        step_id: str | None = None,
    ):
        return self.configuration.parse_job_from_dict(
            variables,
            job_name,
            slurm_id,
            step_id,
        )
