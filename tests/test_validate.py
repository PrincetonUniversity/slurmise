import json
import shutil

import pytest
from click.testing import CliRunner

from slurmise.__main__ import main
from slurmise.validate import validate_config


def write_config(tmp_path, body):
    toml = tmp_path / "slurmise.toml"
    toml.write_text(f'[slurmise]\nbase_dir = "{tmp_path / "slurmise_dir"}"\n{body}')
    return toml


NUPACK = """
[slurmise.job.nupack]
job_spec = "monomer -T {threads} -C {complexity}"
[slurmise.job.nupack.variables]
threads = {type = "numeric"}
complexity = {type = "category"}
"""


def test_valid_config(tmp_path):
    report = validate_config(write_config(tmp_path, NUPACK))

    assert report.ok
    assert report.errors == []
    assert report.warnings == []


def test_validate_does_not_create_base_dir(tmp_path):
    validate_config(write_config(tmp_path, NUPACK))

    assert not (tmp_path / "slurmise_dir").exists()


def test_example_command_parsed(tmp_path):
    report = validate_config(write_config(tmp_path, NUPACK), ["nupack monomer -T 4 -C high"])

    assert report.ok
    assert report.parsed == [
        {
            "cmd": "nupack monomer -T 4 -C high",
            "job_name": "nupack",
            "numerics": {"threads": 4.0},
            "categories": {"complexity": "high"},
        }
    ]


def test_example_command_with_explicit_job_name(tmp_path):
    report = validate_config(write_config(tmp_path, NUPACK), ["monomer -T 4 -C high"], job_name="nupack")

    assert report.ok
    assert report.parsed[0]["job_name"] == "nupack"


def test_example_command_mismatch(tmp_path):
    report = validate_config(write_config(tmp_path, NUPACK), ["nupack dimer -T 4 -C high", "nupack monomer -T 1 -C a"])

    assert not report.ok
    assert len(report.errors) == 1
    assert "'nupack dimer -T 4 -C high' failed to parse" in report.errors[0]
    # the good command is still reported
    assert [p["cmd"] for p in report.parsed] == ["nupack monomer -T 1 -C a"]


def test_example_command_missing_file(tmp_path):
    body = """
    [slurmise.job.align]
    job_spec = "{reads} -t {threads}"
    [slurmise.job.align.variables]
    reads = {type = "file", file_parsers = "file_size"}
    threads = {type = "numeric"}
    """
    report = validate_config(write_config(tmp_path, body), ["align missing.fq -t 2"])

    assert not report.ok
    assert "missing.fq" in report.errors[0]


def test_example_command_with_existing_file(tmp_path):
    reads = tmp_path / "reads.fq"
    reads.write_text("ACGT\n")
    body = """
    [slurmise.job.align]
    job_spec = "{reads} -t {threads}"
    [slurmise.job.align.variables]
    reads = {type = "file", file_parsers = "file_size"}
    threads = {type = "numeric"}
    """
    report = validate_config(write_config(tmp_path, body), [f"align {reads} -t 2"])

    assert report.ok
    assert report.parsed[0]["numerics"] == {"threads": 2.0, "reads_file_size": 5}


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('[slurmise.job.nupack]\njob_spec = "monomer {x}"\n', "no variable types"),
        (NUPACK.replace('complexity = {type = "category"}', ""), "Unknown variable type for variable complexity"),
        (
            NUPACK + '[slurmise.job.nupack.model]\nmodel = "forest"',
            "Job nupack: Unknown model: 'forest'",
        ),
        ("[slurmise.runtime]\nminimum = -5\n" + NUPACK, "Invalid runtime configuration for global"),
        ("this is = not valid toml [", "Invalid TOML"),
    ],
)
def test_invalid_config(tmp_path, body, match):
    report = validate_config(write_config(tmp_path, body))

    assert not report.ok
    assert match in report.errors[0]


def test_missing_base_dir(tmp_path):
    toml = tmp_path / "slurmise.toml"
    toml.write_text(NUPACK)

    report = validate_config(toml)

    assert not report.ok
    assert "Missing required configuration key: 'base_dir'" in report.errors[0]


def test_missing_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    report = validate_config()

    assert not report.ok
    assert "No slurmise.toml was found" in report.errors[0]


def test_shadowed_job_prefix_warns(tmp_path):
    body = """
    [slurmise.job.git]
    job_spec = "{n}"
    [slurmise.job.git.variables]
    n = {type = "numeric"}

    [slurmise.job.git_checkout]
    job_prefix = "git checkout"
    job_spec = "{n}"
    [slurmise.job.git_checkout.variables]
    n = {type = "numeric"}
    """
    report = validate_config(write_config(tmp_path, body))

    assert report.ok
    assert len(report.warnings) == 1
    assert "Job git_checkout (prefix 'git checkout') is shadowed by job git (prefix 'git')" in report.warnings[0]


def test_specific_prefix_listed_first_does_not_warn(tmp_path):
    body = """
    [slurmise.job.git_checkout]
    job_prefix = "git checkout"
    job_spec = "{n}"
    [slurmise.job.git_checkout.variables]
    n = {type = "numeric"}

    [slurmise.job.git]
    job_spec = "{n}"
    [slurmise.job.git.variables]
    n = {type = "numeric"}
    """

    assert validate_config(write_config(tmp_path, body)).warnings == []


def test_file_parser_warnings(tmp_path):
    body = (
        NUPACK
        + """
    [slurmise.file_parsers.no_script]
    type = "numeric"

    [slurmise.file_parsers.file_size]
    awk_script = "{print 1}"
    """
    )
    report = validate_config(write_config(tmp_path, body))

    assert report.ok
    assert report.warnings == [
        "File parser 'no_script' has no awk_script and is ignored.",
        "File parser 'file_size' replaces the built-in parser of the same name.",
    ]


def test_missing_awk_warns(tmp_path, monkeypatch):
    body = """
    [slurmise.file_parsers.first]
    type = "numeric"
    awk_script = "{print $1}"

    [slurmise.job.score]
    job_spec = "{infile}"
    [slurmise.job.score.variables]
    infile = {type = "file", file_parsers = "first"}
    """
    monkeypatch.setattr(shutil, "which", lambda name: None)

    report = validate_config(write_config(tmp_path, body))

    assert report.ok
    assert report.warnings == ["A job uses an awk file parser but `awk` was not found on PATH."]


def test_cli_valid(tmp_path):
    toml = write_config(tmp_path, NUPACK)

    result = CliRunner().invoke(main, ["--toml", str(toml), "validate", "nupack monomer -T 2 -C a"])

    assert result.exit_code == 0
    assert "Parsed 'nupack monomer -T 2 -C a' as job nupack" in result.output
    assert "Configuration is valid." in result.output


def test_cli_invalid_config_is_reported_not_raised(tmp_path):
    toml = write_config(tmp_path, '[slurmise.job.nupack]\njob_spec = "monomer {x}"\n')

    result = CliRunner().invoke(main, ["--toml", str(toml), "validate"])

    assert result.exit_code == 1
    assert "Error: Job nupack has no variable types" in result.output
    assert "Configuration is NOT valid." in result.output


def test_cli_json(tmp_path):
    toml = write_config(tmp_path, NUPACK)

    result = CliRunner().invoke(main, ["--toml", str(toml), "validate", "--json", "nupack dimer -T 2 -C a"])

    assert result.exit_code == 1
    output = json.loads(result.output)
    assert output["ok"] is False
    assert output["toml_path"] == str(toml)
    assert "failed to parse" in output["errors"][0]
    assert output["warnings"] == []
    assert output["parsed"] == []


def test_cli_discovers_config(tmp_path, monkeypatch):
    write_config(tmp_path, NUPACK)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["validate"])

    assert result.exit_code == 0
    assert str(tmp_path / "slurmise.toml") in result.output
