from dataclasses import dataclass
import tomllib
from typing import Any
import numbers
from pathlib import Path
from slurmise import job_data
import re
import subprocess


# matches tokens like {threads:numeric}
JOB_SPEC_REGEX = re.compile(r"{(?:(?P<name>[^:]+):)?(?P<kind>[^}]+)}")
KIND_TO_REGEX = {
    'file': '.+?',
    'numeric': '[-0-9.]+',
    'category': '.+',
    'ignore': '.+',
}
CATEGORICAL = "CATEGORICAL"
NUMERICAL = "NUMERICAL"


class SlurmiseConfiguration:
    """SlurmiseConfiguration class parses and stores TOML configuration files for slurmise."""
    def __init__(self, toml_file: Path):
        """Parse a configuration TOML file which """
        self.file_parsers = {
            'file_size': FileSizeParser(),
            'file_lines': FileLinesParser(),
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


@dataclass()
class FileParser:
    name: str = 'UNK'
    return_type: str = NUMERICAL

    def parse_file(self, path: Path):
        raise NotImplementedError()


class JobSpec:
    def __init__(
            self,
            job_spec: str,
            file_parsers: dict[str, str] | None = None,
            available_parsers: dict[str, FileParser] | None = None,
        ):
        """Parse a job spec string into a regex with named capture groups.

        job_spec: The specification of parsing the supplied command.  Can contain
        placeholders for variables to parse as numerics, strings, or files.
        file_parsers: A dict of file variable names to parser names.  Can be a
        comma separate list or single string
        available_parsers: A dict of parser names to parser objects
        """
        self.job_spec_str = job_spec

        self.token_kinds = {}
        self.file_parsers: dict[str, list[FileParser]] = {}

        while match := JOB_SPEC_REGEX.search(job_spec):
            kind = match.group('kind')
            name = match.group('name')

            if kind not in KIND_TO_REGEX:
                raise ValueError(f"Token kind {kind} is unknown.")

            if kind == 'ignore':
                job_spec = job_spec.replace(match.group(0), f'{KIND_TO_REGEX[kind]}')

            else:
                if name is None:
                    raise ValueError(f"Token {match.group(0)} has no name.")
                self.token_kinds[name] = kind
                job_spec = job_spec.replace(match.group(0), f'(?P<{name}>{KIND_TO_REGEX[kind]})')

                if kind == 'file':
                    self.file_parsers[name] = [
                        available_parsers[parser_type]
                        for parser_type in file_parsers[name].split(',')
                    ]

        self.job_regex = f'^{job_spec}$'

    def parse_job_cmd(self, job: job_data.JobData) -> job_data.JobData:
        m = re.match(self.job_regex, job.cmd)
        if m is None:
            raise ValueError(f"Job spec {self.job_spec_str} does not match command {job.cmd}.")

        for name, kind in self.token_kinds.items():
            if kind == 'numeric':
                job.numerical[name] = float(m.group(name))
            elif kind == 'category':
                job.categorical[name] = m.group(name)
            elif kind == 'file':
                for parser in self.file_parsers[name]:
                    if parser.return_type == NUMERICAL:
                        job.numerical[f"{name}_{parser.name}"] = parser.parse_file(Path(m.group(name)))
                    else:
                        job.categorical[f"{name}_{parser.name}"] = parser.parse_file(Path(m.group(name)))
                #TODO HANDLE FILE KINDs
                #TODO open file and read size
                #TODO if file is a file of filenames, read the files and get their sizes etc
                #TODO open file and read specific data from it
                #TODO deal with gzip files(?)

                #TODO ADD "numeric_list" kind(?) and "category_list" kind(?)
            else:
                raise ValueError(f"Unknown kind {kind}.")
            
        return job

class FileSizeParser(FileParser):
    def __init__(self):
        super().__init__(name='file_size', return_type=NUMERICAL)

    def parse_file(self, path: Path):
        return path.stat().st_size  # in bytes

class FileLinesParser(FileParser):
    def __init__(self):
        super().__init__(name='file_lines', return_type=NUMERICAL)

    def parse_file(self, path: Path):
        with open(path, 'rb') as infile:
            lines = 1  # will count the last line as well.  Off by one for empty files
            buf_size = 1024 * 1024
            read_f = infile.raw.read

            while buf := read_f(buf_size):
                lines += buf.count(b'\n')

        return lines

class FileRegexParser(FileParser):
    def __init__(self, parser_spec):
        super().__init__(self, parser_spec)
        self.regex = re.compile(parser_spec['regex'])
        self.return_mapper = float if parser_spec['return_type'] == 'numerical' else str

    def parse_file(self, path: Path):
        with open(path) as infile:
            for line in infile:
                if match := self.regex.search(line):
                    return self.return_mapper(match.group(1))

        raise ValueError(f'Unable to parse {self.parser_spec['regex']:r} from {path}')


class AwkCommandParser(FileParser):
    def __init__(self, name, return_type, script):
        return_type = return_type.upper()
        super().__init__(name=name, return_type=return_type)
        self.script = script
        self.return_mapper = float if return_type == NUMERICAL else str

    def parse_file(self, path: Path):
        result = subprocess.run([
            'awk',
            self.script,
            path,
        ], capture_output=True, check=True, text=True)

        return self.return_mapper(result.stdout.strip())

# TODO awk command with file script
# multiple returns from one awk?  Only numeric?
