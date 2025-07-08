from pathlib import Path
from slurmise import job_data
from slurmise.job_parse.file_parsers import FileParser
import re


# matches tokens like {threads:numeric}
JOB_SPEC_REGEX = re.compile(r"{(?:(?P<name>[^:}]+):)?(?P<kind>[^}]+)}")
KIND_TO_REGEX = {
    'file': '.+?',
    'gzip_file': '.+?',
    'file_list': '.+?',
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

        self.job_regex = self.build_regex(available_parsers, file_parsers)

    def build_regex(self, available_parsers=None, file_parsers=None, named_ignore=False):
        job_spec = self.job_spec_str
        ignore_ind = 0
        while match := JOB_SPEC_REGEX.search(job_spec):
            kind = match.group('kind')
            name = match.group('name')

            if kind not in KIND_TO_REGEX:
                raise ValueError(f"Token kind {kind} is unknown.")

            if kind == 'ignore':
                if named_ignore:
                    if name is None:
                        name = f'ignore_{ignore_ind}'
                        ignore_ind += 1
                    job_spec = job_spec.replace(match.group(0), f'(?P<{name}>{KIND_TO_REGEX[kind]})', 1)
                else:
                    job_spec = job_spec.replace(match.group(0), f'{KIND_TO_REGEX[kind]}', 1)

            else:
                if name is None:
                    raise ValueError(f"Token {match.group(0)} has no name.")
                self.token_kinds[name] = kind
                job_spec = job_spec.replace(match.group(0), f'(?P<{name}>{KIND_TO_REGEX[kind]})', 1)

                if (kind in ('file', 'gzip_file', 'file_list')
                        and available_parsers and file_parsers):
                    self.file_parsers[name] = [
                        available_parsers[parser_type]
                        for parser_type in file_parsers[name].split(',')
                    ]

        return f'^{job_spec}$'

    def parse_job_cmd(self, job: job_data.JobData) -> job_data.JobData:
        m = re.match(self.job_regex, job.cmd)
        if m is None:
            result = self.align_and_indicate_differences(job.cmd)
            raise ValueError(f"Job spec for {job.job_name} does not match command:\n{result}")

        for name, kind in self.token_kinds.items():
            if kind == 'numeric':
                job.numerical[name] = float(m.group(name))
            elif kind == 'category':
                job.categorical[name] = m.group(name)
            elif kind in ('file', 'gzip_file', 'file_list'):
                for parser in self.file_parsers[name]:
                    match kind:
                        case 'file':
                            file_value = parser.parse_file(Path(m.group(name)))
                        case 'gzip_file':
                            file_value = parser.parse_file(Path(m.group(name)), gzip_file=True)
                        case 'file_list':
                            file_value = [
                            parser.parse_file(Path(file.strip()))
                            for file in open(Path(m.group(name)), 'r')
                        ]

                    if parser.return_type == NUMERICAL:
                        job.numerical[f"{name}_{parser.name}"] = file_value
                    else:
                        job.categorical[f"{name}_{parser.name}"] = file_value

            else:
                raise ValueError(f"Unknown kind {kind}.")
            
        return job


    def align_and_indicate_differences(self, cmd: str, try_exact_match: bool=False) -> str:
        """
        Compares two strings and aligns with indicators for differences.

        Args:
            cmd: The user supplied string.
            try_exact_match: Attempt to match regex exactly, fall back to fuzzy.

        Returns:
            multi-line, aligned string of differences
        """
        from difflib import SequenceMatcher
        import regex

        # need to handle ignore tokens differently as they have no names in the regex
        # rebuild regex to keep names if given and give default if not
        raw_regex = self.job_regex
        job_spec_str = self.job_spec_str
        if 'ignore}' in self.job_spec_str:
            raw_regex = self.build_regex(named_ignore=True)
            # add names to job spec str as well
            ignore_index = 0
            while '{ignore}' in job_spec_str:
                job_spec_str = job_spec_str.replace('{ignore}',
                                      f'{{ignore_{ignore_index}:ignore}}',
                                      1)
                ignore_index += 1

        match = None
        parsable = False

        if try_exact_match:
            parsable = True
            match = re.match(raw_regex, cmd)

        # unable or unwilling to exact match
        if not match:
            parsable = False
            # ?b is for best match
            # {e} indicates to allow errors
            match = regex.fullmatch(f"(?b)(?:{raw_regex})" + r"{e}", cmd)

        # still no matches
        if not match:
            raise ValueError('TODO: handle no matches')

        simple_spec = re.sub(r'{([^:}]+)(:[^}]+)?}', r'{\1}', job_spec_str)
        spec_with_matches = simple_spec.format(**match.groupdict())
        display_spec = re.sub(r'{([^}]+)}', r'{{\1⇒{\1}}}', simple_spec)
        display_spec = display_spec.format(**match.groupdict())

        # this holds indicies for mapping a position in the match string
        # to the display spec
        matches_to_display = []
        offset = 0
        for wc in re.finditer(r'{([^⇒]+)⇒([^}]*)}', display_spec):
            matches_to_display.append(
                (
                    wc.start() + offset,
                    wc.start() + offset + len(wc.group(2)),
                    wc.start(),  # start of match in display_spec
                    wc.end(),  # end of match in display_spec
                    wc.group(1),  # wc name
                )
            )
            offset += len(wc.group(2)) - wc.end() + wc.start()

        # convert to list for slicing
        display_spec = list(display_spec)

        s = SequenceMatcher(None, spec_with_matches, cmd)
        opcodes = s.get_opcodes()

        aligned_spec = []
        aligned_cmd = []
        indicator_line = []

        for tag, spec_start, spec_end, cmd_start, cmd_end in opcodes:
            if tag == 'equal':
                # Matching parts, convert to lists for slicing
                spec_line = list(spec_with_matches[spec_start:spec_end])
                cmd_line = list(cmd[cmd_start:cmd_end])
                ind_line = [' '] * (spec_end - spec_start)

                # replace spec with display portion
                offset = 0
                while (matches_to_display and
                       matches_to_display[0][0] >= spec_start and
                       matches_to_display[0][1] <= spec_end):
                    # get first set of indices, remove spec_start offset
                    match_start, match_end, display_start, display_end, wc_name = (
                        matches_to_display.pop(0))
                    match_start += offset - spec_start
                    match_end += offset - spec_start

                    # insert display spec into spec_line
                    spec_line = (spec_line[:match_start] +
                        display_spec[display_start:display_end] +
                        spec_line[match_end:])

                    # insert markers to ind_line
                    ind_with_arrows = [' '] * (display_end-display_start)
                    # start bracket on first and last position
                    ind_with_arrows[0] = '│'
                    ind_with_arrows[-1] = '│'

                    # if the regex fails because a numeric can't parse, add an indicator
                    if self.token_kinds.get(wc_name, '') == 'numeric':
                        try:
                            float(''.join(cmd_line[match_start:match_end]))
                        except ValueError:  # cannot parse
                            ind_with_arrows[len(ind_with_arrows) // 2] = '⚠'


                    ind_line = (ind_line[:match_start] +
                        ind_with_arrows +
                        ind_line[match_end:])

                    # this many characters were added to spec_line
                    added_chars = display_end - display_start - (
                        match_end - match_start)
                    # extract match in cmd_line and add spaces to stay aligned
                    # surround with corner and indicator
                    cmd_line = (cmd_line[:match_start] +  # unchanged
                        ['└'] +  # start callout
                        # half of added chars, %2 to deal with odd added chars,
                        # remove two for corner and callout
                        ['─'] * (added_chars // 2 + added_chars % 2 - 2) +
                        ['┤'] +  # start of matching chars
                        cmd_line[match_start:match_end] +  # matching chars
                        ['├'] +  # end of matching chars
                        # half of added chars, remove two for corner and callout
                        ['─'] * (added_chars // 2 - 2) +
                        ['┘'] +  # end callout
                        cmd_line[match_end:]  # rest of cmd line
                    )

                    # offset accounts for the difference between what was added
                    # and the original positions
                    offset += added_chars

                aligned_spec.append(''.join(spec_line))
                aligned_cmd.append(''.join(cmd_line))
                indicator_line.append(''.join(ind_line))
            elif tag == 'replace':
                # Replacement
                spec_len = spec_end - spec_start
                cmd_len = cmd_end - cmd_start
                max_len = max(spec_len, cmd_len)
                aligned_spec.append(spec_with_matches[spec_start:spec_end] + ' ' * (max_len - spec_len))
                aligned_cmd.append(cmd[cmd_start:cmd_end] + ' ' * (max_len - cmd_len))
                indicator_line.append('╳' * max_len)
            elif tag == 'delete':
                # Deletion from spec_with_matches
                spec_len = spec_end - spec_start
                aligned_spec.append(spec_with_matches[spec_start:spec_end])
                aligned_cmd.append(' ' * spec_len)
                indicator_line.append('∧' * spec_len)
            elif tag == 'insert':
                # Insertion into cmd
                cmd_len = cmd_end - cmd_start
                aligned_spec.append(' ' * cmd_len)
                aligned_cmd.append(cmd[cmd_start:cmd_end])
                indicator_line.append('∨' * cmd_len)

        result = []

        if try_exact_match:
            if parsable:
                result += ["Able to parse"]
            else:
                result += ["Failed to parse"]

        result += [
                "".join(aligned_spec),
                "".join(indicator_line),
                "".join(aligned_cmd),
            ]

        return '\n'.join(result)
