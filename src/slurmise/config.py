import tomllib
from pathlib import Path
from slurmise import job_data
from slurmise.job_parse import file_parsers
from slurmise.job_parse.job_specification import JobSpec


class SlurmiseConfiguration:
    """SlurmiseConfiguration class parses and stores TOML configuration files for slurmise."""
    def __init__(self, toml_file: Path):
        """Parse a configuration TOML file"""
        self.file_parsers = {
            'file_size': file_parsers.FileSizeParser(),
            'file_lines': file_parsers.FileLinesParser(),
        }
        with open(toml_file, 'rb') as f:
            toml_data = tomllib.load(f)

            self.slurmise_base_dir = toml_data['slurmise']['base_dir']
            parsers = toml_data['slurmise'].get('file_parsers', {})

            for parser_name, config in parsers.items():
                return_type = config.get('return_type', 'categorical')
                if 'awk_script' in config:
                    script_is_file = config.get('script_is_file', False)
                    self.file_parsers[parser_name] = file_parsers.AwkParser(
                        parser_name, return_type, config['awk_script'], script_is_file,
                    )

            self.jobs = toml_data['slurmise'].get('job', {})
            self.job_prefixes: dict[str, str] = {}

            for job_name, job in self.jobs.items():
                self.jobs[job_name]['job_spec_obj'] = JobSpec(
                    job['job_spec'],
                    file_parsers=job.get('file_parsers', {}),
                    available_parsers=self.file_parsers,
                )
                if 'job_prefix' in job:
                    self.job_prefixes[job['job_prefix']] = job_name

    def parse_job_cmd(self, cmd: str, job_name: str | None = None, slurm_id: str | None = None) -> job_data.JobData:
        """Parse a job data dataset into a JobData object."""
        if job_name is None:  # try to infer
            for prefix, name in self.job_prefixes.items():
                if cmd.startswith(prefix):
                    job_name = name
                    cmd = cmd.removeprefix(prefix).lstrip()
                    break

            else:  # not a prefix
                for name in self.jobs.keys():
                    if cmd.startswith(name):
                        job_name = name
                        cmd = cmd.removeprefix(name).lstrip()
                        break

                else:
                    raise ValueError(f'Unable to match job name to {cmd!r}')

        # TODO decide if prefix is removed from command?
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name} not found in configuration.")

        jd = job_data.JobData(
            job_name=job_name,
            slurm_id=slurm_id,
            cmd=cmd
        )

        job_spec = self.jobs[job_name]["job_spec_obj"]
        return job_spec.parse_job_cmd(jd)

    def add_defaults(self, job_data: job_data.JobData) -> job_data.JobData:
        """Add default values to a job data object."""
        job_data.memory = 1000
        job_data.runtime = 60
        return job_data