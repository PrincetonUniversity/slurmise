from dataclasses import dataclass
from pathlib import Path
import subprocess


NUMERICAL = "NUMERICAL"


@dataclass()
class FileParser:
    name: str = 'UNK'
    return_type: str = NUMERICAL

    def parse_file(self, path: Path):
        raise NotImplementedError()


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


class AwkCommandParser(FileParser):
    def __init__(self, name, return_type, script):
        return_type = return_type.upper()
        super().__init__(name=name, return_type=return_type)
        self.script = script

    def parse_file(self, path: Path):
        result = subprocess.run([
            'awk',
            self.script,
            path,
        ], capture_output=True, check=True, text=True)

        if self.return_type == NUMERICAL:
            return [float(token) for token in result.stdout.split()]
        return result.stdout.strip()


class AwkFileParser(FileParser):
    def __init__(self, name, return_type, script_file):
        return_type = return_type.upper()
        super().__init__(name=name, return_type=return_type)
        self.script_file = script_file

    def parse_file(self, path: Path):
        result = subprocess.run([
            'awk',
            '-f',
            self.script_file,
            path,
        ], capture_output=True, check=True, text=True)

        if self.return_type == NUMERICAL:
            return [float(token) for token in result.stdout.split()]
        return result.stdout.strip()
