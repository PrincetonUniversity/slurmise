import tomllib
from pathlib import Path
from collections import defaultdict
from slurmise import job_data
from slurmise.job_parse import file_parsers
from slurmise.job_parse.job_specification import JobSpec


class SlurmiseConfiguration:
    """SlurmiseConfiguration class parses and stores TOML configuration files for slurmise."""

    def __init__(self, toml_file: Path):
        """Parse a configuration TOML file"""
        self.file_parsers = {
            "file_size": file_parsers.FileSizeParser(),
            "file_lines": file_parsers.FileLinesParser(),
            "file_basename": file_parsers.FileBasename(),
            "file_md5": file_parsers.FileMD5(),
        }
        with open(toml_file, "rb") as f:
            toml_data = tomllib.load(f)

            self.slurmise_base_dir = toml_data["slurmise"]["base_dir"]
            Path(self.slurmise_base_dir).mkdir(parents=True, exist_ok=True)
            self.db_filename = Path(self.slurmise_base_dir) / toml_data["slurmise"].get(
                "db_filename", "slurmise.h5"
            )
            parsers = toml_data["slurmise"].get("file_parsers", {})

            for parser_name, config in parsers.items():
                return_type = config.get("return_type", "categorical")
                if "awk_script" in config:
                    script_is_file = config.get("script_is_file", False)
                    self.file_parsers[parser_name] = file_parsers.AwkParser(
                        parser_name,
                        return_type,
                        config["awk_script"],
                        script_is_file,
                    )

            self.jobs = toml_data["slurmise"].get("job", {})
            self.job_prefixes: dict[str, str] = {}
            self.default_runtime = defaultdict(lambda: int(toml_data["slurmise"].get("default_time", 60)))
            self.default_memory = defaultdict(lambda: int(toml_data["slurmise"].get("default_mem", 1000)))

            for job_name, job in self.jobs.items():
                self.jobs[job_name]["job_spec_obj"] = JobSpec(
                    job["job_spec"],
                    file_parsers=job.get("file_parsers", {}),
                    available_parsers=self.file_parsers,
                )
                if "job_prefix" in job:
                    self.job_prefixes[job_name] = job["job_prefix"]
                if "default_time" in job:
                    self.default_runtime[job_name] = int(job['default_time'])
                if "default_mem" in job:
                    self.default_memory[job_name] = int(job['default_mem'])

    def parse_job_cmd(
            self, cmd: str, job_name: str | None = None, slurm_id: str | None = None,
            step_id: str | None = None
            ) -> job_data.JobData:
        """Parse a job data dataset into a JobData object."""

        jd = self._fill_job_name(cmd, job_name, slurm_id, step_id)
        job_spec = self.jobs[jd.job_name]["job_spec_obj"]

        return job_spec.parse_job_cmd(jd)

    def dry_parse(
            self,
            cmd: str,
            job_name: str | None = None,
        ):

        jd = self._fill_job_name(cmd, job_name)
        job_spec = self.jobs[jd.job_name]["job_spec_obj"]
        return job_spec.align_and_indicate_differences(jd.cmd, try_exact_match=True)

    def _fill_job_name(
            self,
            cmd: str,
            job_name: str | None = None,
            slurm_id: str | None = None,
            step_id: str | None = None,
        ) -> job_data.JobData:
        """From the user supplied input, create a job data object."""
        if job_name is None:  # try to infer
            for name, prefix in self.job_prefixes.items():
                if cmd.startswith(prefix):
                    job_name = name
                    cmd = cmd.removeprefix(prefix).lstrip()
                    break

            else:  # not a prefix. Runs when it does not hit the break.
                for name in self.jobs.keys():
                    if cmd.startswith(name):
                        job_name = name
                        cmd = cmd.removeprefix(name).lstrip()
                        break

                else:
                    raise ValueError(f"Unable to match job name to {cmd!r}")
        else:
            job_prefix = self.job_prefixes.get(job_name, None)
            if job_prefix is not None:
                cmd = cmd.removeprefix(job_prefix).lstrip()

        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name} not found in configuration.")

        if step_id is not None:
            slurm_id = '.'.join([str(slurm_id), str(step_id)])
        return job_data.JobData(
            job_name=job_name,
            slurm_id=slurm_id,
            cmd=cmd
        )


    def add_defaults(self, job_data: job_data.JobData) -> job_data.JobData:
        """Add default values to a job data object."""
        job_data.memory = self.default_memory[job_data.job_name]
        job_data.runtime = self.default_runtime[job_data.job_name]
        return job_data
