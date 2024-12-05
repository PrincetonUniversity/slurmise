from pathlib import Path
from slurmise import job_data
from slurmise.job_parse.file_parsers import FileParser
import re


# matches tokens like {threads:numeric}
JOB_SPEC_REGEX = re.compile(r"{(?:(?P<name>[^:]+):)?(?P<kind>[^}]+)}")
KIND_TO_REGEX = {
    'file': '.+?',
    'gzip_file': '.+?',
    'numeric': '[-0-9.]+',
    'category': '.+?',
    'ignore': '.+?',
}
CATEGORICAL = "CATEGORICAL"
NUMERICAL = "NUMERICAL"

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

                if kind in ('file', 'gzip_file', 'file_list'):
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
            elif kind in ('file', 'gzip_file', 'file_list'):
                for parser in self.file_parsers[name]:
                    if kind == 'file':
                        file_value = parser.parse_file(Path(m.group(name)))
                    elif kind == 'gzip_file':
                        file_value = parser.parse_file(Path(m.group(name)), gzip_file=True)
                    if parser.return_type == NUMERICAL:
                        job.numerical[f"{name}_{parser.name}"] = file_value
                    else:
                        job.categorical[f"{name}_{parser.name}"] = file_value
                # TODO if file is a file of filenames, read the files and get their sizes etc
                # TODO deal with gzip files(?)

            else:
                raise ValueError(f"Unknown kind {kind}.")
            
        return job
