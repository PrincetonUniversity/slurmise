import pytest

from slurmise import parse_args


# @pytest.mark.parametrize(
#     "args",
#     [
#         "record cmd subcmd -o -k 2 -j -i 3 -m fast -q=5",
#     ],
# )
# def test_record_subcommand(args, monkeypatch):
#     """Use click test runner to ensure record subcommand exits with errorcode 0."""

#     def mock_get_slurm_job_sacct():
#         with open("tests/sacct_output.json") as f:
#             return json.load(f)

#     def mock_get_slurm_job_sstat():
#         return {}

#     monkeypatch.setattr(slurm, "get_slurm_job_sacct", mock_get_slurm_job_sacct)
#     monkeypatch.setattr(slurm, "get_slurm_job_sstat", mock_get_slurm_job_sstat)

#     runner = CliRunner()
#     result = runner.invoke(main, args.split())

#     assert result.exit_code == 0


@pytest.mark.parametrize(
    "args,exp",
    [
        (
            ["sleep", "2"],
            {
                "cmd": ["sleep", "2"],
                "positional": ["sleep", "2"],
                "options": {},
                "flags": {},
            },
        ),
        (
            ["sort", "-k", "2"],
            {
                "cmd": ["sort", "-k", "2"],
                "positional": ["sort"],
                "options": {"-k": "2"},
                "flags": {},
            },
        ),
        (
            ["time", "-l", "-p", "2", "--dog"],
            {
                "cmd": ["time", "-l", "-p", "2", "--dog"],
                "positional": ["time"],
                "options": {"-p": "2"},
                "flags": {"-l": True, "--dog": True},
            },
        ),
        (
            ["samtools", "view", "chr:1-100", "-b", "-o", "output.bam"],
            {
                "cmd": ["samtools", "view", "chr:1-100", "-b", "-o", "output.bam"],
                "positional": ["samtools", "view", "chr:1-100"],
                "options": {"-o": "output.bam"},
                "flags": {"-b": True},
            },
        ),
        (
            ["grep", "--color=auto", "-i", "hello world"],
            {
                "cmd": ["grep", "--color=auto", "-i", "hello world"],
                "positional": ["grep"],
                "options": {"--color": "auto", "-i": "hello world"},
                "flags": {},
            },
        ),
        (
            ["jq", "-l", ".[]", "file.json"],
            {
                "cmd": ["jq", "-l", ".[]", "file.json"],
                "positional": ["jq", "file.json"],
                "options": {"-l": ".[]"},
                "flags": {},
            },
        ),
    ],
)
def test_parse_record_args(args, exp):
    assert parse_args.parse_slurmise_record_args(args) == exp


@pytest.mark.parametrize(
    "parsed_args,exp",
    [
        (
            {
                "cmd": ["sort", "-k", "2" "tests/sacct_output.json"],
                "positional": ["sort", "tests/sacct_output.json"],
                "options": {"-k": "2"},
                "flags": {},
            },
            [
                {
                    "name": "sort",
                    "arg_type": "positional",
                    "type": "string",
                    "value": "sort",
                },
                {
                    "name": "tests/sacct_output.json",
                    "arg_type": "positional",
                    "type": "file",
                    "size": 13398,
                },
                {"name": "-k", "arg_type": "option", "type": "string", "value": "-k"},
            ],
        ),
    ],
)
def test_process_record_args(parsed_args, exp):
    processed_args = parse_args.process_slurmise_record_args(parsed_args)
    print(processed_args)
    assert processed_args == exp
