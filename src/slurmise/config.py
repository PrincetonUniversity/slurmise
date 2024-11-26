import tomllib
from pathlib import Path
from slurmise import job_data
from slurmise.job_parse import file_parsers
from slurmise.job_parse.job_specification import JobSpec


class SlurmiseConfiguration:
    """SlurmiseConfiguration class parses and stores TOML configuration files for slurmise."""
    def __init__(self, toml_file: Path):
        """Parse a configuration TOML file which """
        self.file_parsers = {
            'file_size': file_parsers.FileSizeParser(),
            'file_lines': file_parsers.FileLinesParser(),
        }
        with open(toml_file, 'rb') as f:
            toml_data = tomllib.load(f)

            self.slurmise_base_dir = toml_data['slurmise']['base_dir']
            self.jobs = toml_data['slurmise']['job']

            for job_name,job in self.jobs.items():
                self.jobs[job_name]['job_spec_obj'] = JobSpec(job['job_spec'])

    def parse_job_cmd(self, job_name: str, slurm_id: str, cmd: str) -> job_data.JobData:
        """Parse a job data dataset into a JobData object."""
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name} not found in configuration.")

        jd = job_data.JobData(
            job_name=job_name,
            slurm_id=slurm_id,
            cmd=cmd
        )

        job_spec = self.jobs[job_name]["job_spec_obj"]
        return job_spec.parse_job_cmd(jd)
