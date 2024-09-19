import tomllib
from typing import Any
from pathlib import Path
from slurmise import job_data
import re

JOB_SPEC_REGEX = re.compile(r"{(?P<name>[^:]+):(?P<kind>[^}]+)}")

class SlurmiseConfiguration:
    """SlurmiseConfiguration class parses and stores TOML configuration files for slurmise."""
    def __init__(self, toml_file: Path):
        """Parse a configuration TOML file which """
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


class JobSpec:
    def __init__(self, job_spec: str):
        """Parse a job spec string into a regex with named capture groups."""
        self.job_spec_str = job_spec

        kind_to_regex = {
            'file': '.+?',
            'numeric': '[-0-9.]+',
            'category': '.+'
        }

        self.token_kinds = {}

        while match := JOB_SPEC_REGEX.search(job_spec):
            name = match.group('name')
            kind = match.group('kind')
            self.token_kinds[name] = kind
            job_spec = job_spec.replace(match.group(0), f'(?P<{name}>{kind_to_regex[kind]})')

        self.job_regex = job_spec

    def parse_job_cmd(self, job: job_data.JobData) -> job_data.JobData:
        m = re.match(self.job_regex, job.cmd)
        if m is None:
            raise ValueError(f"Job spec {self.job_spec_str} does not match command {job.cmd}.")

        for name,kind in self.token_kinds.items():
            if kind == 'numeric':
                job.numerical[name] = float(m.group(name))
            elif kind == 'category':
                job.categorical[name] = m.group(name)
                #TODO HANDLE FILE KINDs
                #TODO open file and read size
                #TODO if file is a file of filenames, read the files and get their sizes etc
                #TODO open file and read specific data from it
                #TODO deal with gzip files(?)

                #TODO ADD "IGNORE" kind which will ignore the parameter
                #TODO ADD "numeric_list" kind(?) and "category_list" kind(?)
            else:
                raise ValueError(f"Unknown kind {kind}.")
            
        return job

