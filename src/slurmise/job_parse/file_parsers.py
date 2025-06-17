import gzip
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import hashlib


NUMERICAL = "NUMERICAL"
CATEGORICAL = "CATEGORICAL"


@dataclass()
class FileParser:
    name: str = 'UNK'
    return_type: str = NUMERICAL

    def parse_file(self, path: Path, gzip_file: bool=False):   # pragma: no cover
        raise NotImplementedError()


@dataclass()
class FileBasename(FileParser):
    def __init__(self):
        super().__init__(name='file_basename', return_type=CATEGORICAL)

    def parse_file(self, path: Path, gzip_file: bool=False):
        return path.name


@dataclass()
class FileMD5(FileParser):
    def __init__(self):
        super().__init__(name='file_md5', return_type=CATEGORICAL)

    def parse_file(self, path: Path, gzip_file: bool=False):
        md5_hash = hashlib.md5()
        md5_hash.update(path.read_bytes())
        return md5_hash.hexdigest()


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
            zcat = subprocess.Popen(('zcat', path), stdout=subprocess.PIPE)
            result = subprocess.check_output(self.args, stdin=zcat.stdout, text=True)
            zcat.wait()
        else:
            result = subprocess.check_output(self.args + [path], text=True)

        if self.return_type == NUMERICAL:
            return [float(token) for token in result.split()]
        return result.strip()
