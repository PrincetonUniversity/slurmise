from slurmise.__main__ import main
from slurmise import utils

from click.testing import CliRunner
import pytest
import json

@pytest.mark.parametrize("args", [
    "record cmd subcmd -o -k 2 -j -i 3 -m fast -q=5", 
])
def test_record_subcommand(args):
    """Use click test runner to ensure record subcommand exits with errorcode 0."""
    runner = CliRunner()
    result = runner.invoke(main, args.split())

    assert result.exit_code == 0

@pytest.mark.parametrize("args,exp", [
    (["sleep", "2"], {"cmd": ["sleep", "2"], "positional": ["sleep", "2"], "options": {}, "flags": {}}),
    (["sort", "-k", "2"], {"cmd": ["sort", "-k", "2"], "positional": ["sort"], "options": {"-k": "2"}, "flags": {}}),
])
def test_parse_record_args(args, exp):
    assert utils.parse_slurmise_record_args(args) == exp