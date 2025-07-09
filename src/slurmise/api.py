from slurmise import job_database, slurm
from slurmise.config import SlurmiseConfiguration
from slurmise.fit.poly_fit import PolynomialFit

import numpy as np


class Slurmise:
    """
    API class for interacting with slurmise.
    """

    def __init__(self, toml_path=None):
        self.configuration = SlurmiseConfiguration(toml_path)

    def record(
        self,
        cmd: str,
        job_name: str | None = None,
        slurm_id: str | None = None,
        step_id: str | None = None,
    ):
        # Pull just the slurm ID
        metadata_json = slurm.parse_slurm_job_metadata(
            slurm_id=slurm_id, step_id=step_id
        )

        parsed_jd = self.configuration.parse_job_cmd(
            cmd=cmd,
            job_name=job_name,
            slurm_id=metadata_json["slurm_id"],
            step_id=metadata_json["step_id"],
        )

        parsed_jd.memory = metadata_json["max_rss"]
        parsed_jd.runtime = metadata_json["elapsed_seconds"]

        with job_database.JobDatabase.get_database(
            self.configuration.db_filename
        ) as database:
            database.record(parsed_jd)

    def dry_parse(
        self,
        cmd: str,
        job_name: str | None = None,
    ):
        parsed_output = self.configuration.dry_parse(
            cmd=cmd,
            job_name=job_name,
        )
        return parsed_output

    def raw_record(self, job_data):
        if "." in job_data.slurm_id:
            # If the slurm_id is in the format "1234.0", split it to get the step_id
            slurm_id, step_name = job_data.slurm_id.split(".")
        else:
            # If no step_id is provided, use the slurm_id as is
            slurm_id, step_name = job_data.slurm_id, None
        metadata_json = slurm.parse_slurm_job_metadata(
            slurm_id=slurm_id, step_name=step_name
        )

        job_data.memory = metadata_json["max_rss"]
        job_data.runtime = metadata_json["elapsed_seconds"]

        with job_database.JobDatabase.get_database(
            self.configuration.db_filename
        ) as database:
            database.record(job_data)

    def print(self):
        with job_database.JobDatabase.get_database(
            self.configuration.db_filename
        ) as database:
            database.print()

    def predict(self, cmd, job_name):
        query_jd = self.configuration.parse_job_cmd(cmd=cmd, job_name=job_name)
        query_jd = self.configuration.add_defaults(query_jd)
        query_model = PolynomialFit.load(
            query=query_jd, path=self.configuration.slurmise_base_dir
        )
        query_jd, query_warns = query_model.predict(query_jd)
        return query_jd, query_warns

    def raw_predict(self, query_jd):
        # breakpoint()
        query_jd = self.configuration.add_defaults(query_jd)
        query_model = PolynomialFit.load(
            query=query_jd, path=self.configuration.slurmise_base_dir
        )
        query_jd, query_warns = query_model.predict(query_jd)
        return query_jd, query_warns

    def update_model(self, cmd, job_name):
        query_jd = self.configuration.parse_job_cmd(cmd=cmd, job_name=job_name)
        with job_database.JobDatabase.get_database(
            self.configuration.db_filename
        ) as database:
            jobs = database.query(query_jd)

        self._update_model(query_jd, jobs)

    def _update_model(self, query_jd, jobs):
        model_path = self.configuration.slurmise_base_dir

        try:
            query_model = PolynomialFit.load(query=query_jd, path=model_path)
        except FileNotFoundError:
            query_model = PolynomialFit(query=query_jd, degree=2, path=model_path)

        random_state = np.random.RandomState(42)
        query_model.fit(jobs, random_state=random_state)

        query_model.save()

    def update_all_models(self):
        with job_database.JobDatabase.get_database(
            self.configuration.db_filename
        ) as database:
            for query_jd, jobs in database.iterate_database():
                self._update_model(query_jd, jobs)
