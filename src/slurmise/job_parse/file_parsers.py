import gzip
from dataclasses import dataclass, field
from pathlib import Path
import subprocess


NUMERICAL = "NUMERICAL"


@dataclass()
class FileParser:
    name: str = 'UNK'
    return_type: str = NUMERICAL

    def parse_file(self, path: Path, gzip_file: bool=False):
        raise NotImplementedError()


@dataclass()
class FileSizeParser(FileParser):
    def __init__(self):
        super().__init__(name='file_size', return_type=NUMERICAL)

    def parse_file(self, path: Path, gzip_file: bool=False):
        return path.stat().st_size  # in bytes

@dataclass()
class FileLinesParser(FileParser):
    def __init__(self):
        super().__init__(name='file_lines', return_type=NUMERICAL)

    def parse_file(self, path: Path, gzip_file: bool=False):
        if gzip_file:
            with gzip.open(path, 'rb') as infile:
                # gzip files have no raw read, use slower loop
                for lines, _ in enumerate(infile):
                    pass
                return lines + 1
        else:
            with open(path, 'rb') as infile:
                lines = 1  # will count the last line as well.  Off by one for empty files
                buf_size = 1024 * 1024
                read_f = infile.raw.read

                while buf := read_f(buf_size):
                    lines += buf.count(b'\n')

            return lines


@dataclass()
class AwkParser(FileParser):
    args: list[str] = field(default_factory=list)

    def __init__(self, name, return_type, script, script_is_file=False):
        return_type = return_type.upper()
        super().__init__(name=name, return_type=return_type)
        self.args = ['awk', script]
        if script_is_file:
            # add file argument to awk
            self.args.insert(1, '-f')

    def parse_file(self, path: Path, gzip_file: bool=False):
        if gzip_file:
            # zcat = subprocess.run(('zcat', path), check=True, capture_output=True)
            # result = subprocess.run(self.args, input=zcat.stdout,
            #                         capture_output=True, check=True, text=True)
            zcat = subprocess.Popen(('zcat', path), stdout=subprocess.PIPE)
            result = subprocess.check_output(self.args, stdin=zcat.stdout, text=True)
            zcat.wait()
        else:
            result = subprocess.check_output(self.args + [path], text=True)

        if self.return_type == NUMERICAL:
            return [float(token) for token in result.split()]
        return result.strip()
