import math
from pathlib import Path

import pytest

from slurmise.config import SlurmiseConfiguration, find_config_file
from slurmise.job_parse import file_parsers
from slurmise.resource_corrector import ResourceCorrector


def write_toml(tmp_path, toml_str):
    d = tmp_path.mkdir("slurmise_dir")
    f = d.join("basic.toml")
    f.write(toml_str)
    return f


def test_missing_variables_section(tmpdir):
    """Test the default can be set at the slurmise level for all jobs without additional defaults."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    """
    toml = write_toml(tmpdir, toml_str)

    with pytest.raises(ValueError, match="Job nupack has no variable types. A `variables` entry is required."):
        SlurmiseConfiguration(toml)


def test_job_without_numeric_variable(tmpdir):
    """A job needs something to regress on, so an all category job is rejected (issue #78)."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -M {mode} -C {complexity}"
    [slurmise.job.nupack.variables]
    mode = {type = "category"}
    complexity = {type = "category"}
    """
    toml = write_toml(tmpdir, toml_str)

    with pytest.raises(ValueError, match="at least one numeric variable"):
        SlurmiseConfiguration(toml)


def test_job_with_numeric_variable_is_accepted(tmpdir):
    """One numeric alongside the categories is enough."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complexity = {type = "category"}
    """
    toml = write_toml(tmpdir, toml_str)

    config = SlurmiseConfiguration(toml)

    assert config.jobs["nupack"]["job_spec_obj"].token_kinds == {
        "threads": "numeric",
        "complexity": "category",
    }


def test_missing_variable_type(tmpdir):
    """Test the default can be set at the slurmise level for all jobs without additional defaults."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    """
    toml = write_toml(tmpdir, toml_str)

    with pytest.raises(ValueError, match="Unknown variable type for variable complexity"):
        SlurmiseConfiguration(toml)


def test_no_placeholders(tmpdir):
    """Test the default can be set at the slurmise level for all jobs without additional defaults."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T some_fixed_value -C other_fixed_value"
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complexity = {type = "category"}
    """
    toml = write_toml(tmpdir, toml_str)

    with pytest.raises(ValueError, match="Job specification contains no variables"):
        SlurmiseConfiguration(toml)


def test_default_resources_no_setting(tmpdir):
    """When no runtime/memory section exists, built-in defaults are used."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complexity = {type = "category"}
    """
    toml = write_toml(tmpdir, toml_str)

    config = SlurmiseConfiguration(toml)
    job_data = config.parse_job_cmd("nupack monomer -T 3 -C high")
    config.add_defaults(job_data)

    assert job_data.memory == 1000
    assert job_data.runtime == 60


def test_init_SlurmiseConfiguration_missing_file(tmpdir):
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complexity = {type = "file"}
    """
    toml = write_toml(tmpdir, toml_str)
    with pytest.raises(ValueError, match="File 'complexity' has no assigned file parser"):
        SlurmiseConfiguration(toml)


def test_init_SlurmiseConfiguration_wrong_name(tmpdir):
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complxity = {type = "category"}
    """
    toml = write_toml(tmpdir, toml_str)
    with pytest.raises(ValueError, match="Unknown variable type for variable complexity"):
        SlurmiseConfiguration(toml)


def test_init_SlurmiseConfiguration_unknown_variable_type(tmpdir):
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complexity = {type = "asdf"}
    """
    toml = write_toml(tmpdir, toml_str)
    with pytest.raises(ValueError, match="Unknown variable type asdf for variable complexity"):
        SlurmiseConfiguration(toml)


@pytest.fixture
def basic_toml(tmpdir):
    return write_toml(
        tmpdir,
        """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.runtime]
    default = 70

    [slurmise.memory]
    default = 2000

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads} -C {complexity}"
    [slurmise.job.nupack.runtime]
    default = 80
    [slurmise.job.nupack.memory]
    default = 3000
    [slurmise.job.nupack.variables]
    threads = {type = "numeric"}
    complexity = {type = "category"}

    [slurmise.job.with_ignore]
    job_prefix = "nothing"
    job_spec = "-T {threads} -C {complexity} -i {ignore}"
    [slurmise.job.with_ignore.variables]
    threads = {type = "numeric"}
    complexity = {type = "category"}

    [slurmise.job.dict_spec]
    [slurmise.job.dict_spec.variables]
    threads = {type = "numeric"}
    runtype = {type = "category"}
    infile = {type = "file", file_parsers = "file_basename"}

    [slurmise.job.both_specs]
    job_spec = "-T {threads} -C {runtype} -i {infile}"
    file_parsers.infile = "file_basename"
    [slurmise.job.both_specs.variables]
    threads = {type = "numeric"}
    runtype = {type = "category"}
    infile = {type = "file", file_parsers = "file_basename"}

    # builtins will include file_size and file_lines
    # specify custom options here
    [slurmise.file_parsers.get_epochs]
    type = "numeric"
    awk_script = "'/^epochs:/ {print $2}'"

    [slurmise.file_parsers.fasta_lengths]
    type = "numeric"
    awk_script = "/a/path/to/file"
    script_is_file = true

    # category default return type
    [slurmise.file_parsers.script_string]
    awk_script = "/^>/"
    script_is_file = false

    # this is ignored in parsing as the argument doesn't match an awk parser
    [slurmise.file_parsers.unknown_type]
    no_awk_script = "/^>/"
    script_is_file = false
    """,
    )


def test_init_SlurmiseConfiguration(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    assert config.slurmise_base_dir == "slurmise_dir"
    assert len(config.jobs) == 4
    assert config.jobs["with_ignore"]["job_prefix"] == "nothing"


def test_parse_job_cmd(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("monomer -T 1 -C simple", "nupack", "1234")

    assert job_data.job_name == "nupack"
    assert job_data.slurm_id == "1234"
    assert job_data.categories == {"complexity": "simple"}
    assert job_data.numerics == {"threads": 1}


def test_parse_job_from_variables(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_from_dict({"threads": 3, "runtype": "something", "infile": "test.txt"}, "dict_spec")

    assert job_data.job_name == "dict_spec"
    assert job_data.categories == {"runtype": "something", "infile_file_basename": "test.txt"}
    assert job_data.numerics == {"threads": 3}


def test_parse_job_cmd_with_ignore(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("-T 1 -C simple -i ignored", "with_ignore", "1234")

    assert job_data.job_name == "with_ignore"
    assert job_data.slurm_id == "1234"
    assert job_data.categories == {"complexity": "simple"}
    assert job_data.numerics == {"threads": 1}


def test_parse_job_cmd_invalid(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec for nupack does not match command:") as ve:
        config.parse_job_cmd("dimer -T 1 -C simple", "nupack", "1234")
    print(f"\n{ve.value}")


def test_parse_job_cmd_name_mismatch(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job oldpack not found in configuration."):
        config.parse_job_cmd("monomer -T 1 -C simple", "oldpack", "1234")


def test_parse_job_cmd_invalid_numeric(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec for nupack does not match command:") as ve:
        config.parse_job_cmd("monomer -T 1A -C simple", "nupack", "1234")
    print(f"\n{ve.value}")


def test_parse_job_cmd_no_job_spec(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job dict_spec has no job spec entry for parsing commands"):
        config.parse_job_cmd("monomer -T 1A -C simple", "dict_spec", "1234")


def test_awk_parsers(basic_toml):
    config = SlurmiseConfiguration(basic_toml)

    assert config.file_parsers == {
        "file_size": file_parsers.FileSizeParser(),
        "file_lines": file_parsers.FileLinesParser(),
        "file_basename": file_parsers.FileBasename(),
        "file_md5": file_parsers.FileMD5(),
        "get_epochs": file_parsers.AwkParser("get_epochs", "numeric", "'/^epochs:/ {print $2}'", False),
        "fasta_lengths": file_parsers.AwkParser("fasta_lengths", "numeric", "/a/path/to/file", True),
        "script_string": file_parsers.AwkParser("script_string", "category", "/^>/", False),
    }


def test_parse_job_cmd_inference(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Unable to match job name to 'sort infile'"):
        config.parse_job_cmd("sort infile")

    match_prefix = config.parse_job_cmd("nothing -T 3 -C high -i something")
    assert match_prefix.job_name == "with_ignore"

    match_name = config.parse_job_cmd("nupack monomer -T 3 -C high")
    assert match_name.job_name == "nupack"


def test_default_resources_global(basic_toml):
    """Global [slurmise.runtime] / [slurmise.memory] defaults flow to jobs without overrides."""
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("nothing -T 3 -C high -i something")
    config.add_defaults(job_data)

    assert job_data.memory == 2000
    assert job_data.runtime == 70


def test_default_resources_per_job(basic_toml):
    """Per-job runtime/memory sections override the global defaults."""
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("nupack monomer -T 3 -C high")
    config.add_defaults(job_data)

    assert job_data.memory == 3000
    assert job_data.runtime == 80


def test_get_runtime_corrector_no_override(tmpdir):
    """Jobs without a runtime section use global config."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.runtime]
    default = 120
    minimum = 10
    maximum = 2880

    [slurmise.job.myjob]
    [slurmise.job.myjob.variables]
    n = {type = "numeric"}
    """
    toml = write_toml(tmpdir, toml_str)
    config = SlurmiseConfiguration(toml)

    corrector = config.get_runtime_corrector("myjob")
    assert corrector.default == 120
    assert corrector.minimum == 10
    assert corrector.maximum == 2880
    assert isinstance(corrector, ResourceCorrector)


def test_get_runtime_corrector_per_job_override(tmpdir):
    """Per-job runtime section overrides specific fields; others inherit global."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.runtime]
    default = 60
    minimum = 5
    maximum = 1440

    [slurmise.job.myjob]
    [slurmise.job.myjob.runtime]
    default = 240
    maximum = 480
    [slurmise.job.myjob.variables]
    n = {type = "numeric"}
    """
    toml = write_toml(tmpdir, toml_str)
    config = SlurmiseConfiguration(toml)

    corrector = config.get_runtime_corrector("myjob")
    assert corrector.default == 240
    assert corrector.minimum == 5  # inherited from global
    assert corrector.maximum == 480  # overridden


def test_get_memory_corrector_no_override(tmpdir):
    """Jobs without a memory section use global config."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.memory]
    default = 4000
    minimum = 500
    maximum = 64000

    [slurmise.job.myjob]
    [slurmise.job.myjob.variables]
    n = {type = "numeric"}
    """
    toml = write_toml(tmpdir, toml_str)
    config = SlurmiseConfiguration(toml)

    corrector = config.get_memory_corrector("myjob")
    assert corrector.default == 4000
    assert corrector.minimum == 500
    assert corrector.maximum == 64000


def test_get_memory_corrector_all_corrector_fields(tmpdir):
    """All ResourceCorrector fields can be set via toml."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.memory]
    default = 2000
    minimum = 100
    maximum = 128000
    multiply_prediction_by = 1.2
    retry_exponent = 2.0
    on_high_uncertainty_return = "max"

    [slurmise.job.myjob]
    [slurmise.job.myjob.variables]
    n = {type = "numeric"}
    """
    toml = write_toml(tmpdir, toml_str)
    config = SlurmiseConfiguration(toml)

    corrector = config.get_memory_corrector("myjob")
    assert corrector.default == 2000
    assert corrector.minimum == 100
    assert corrector.maximum == 128000
    assert corrector.multiply_prediction_by == pytest.approx(1.2)
    assert corrector.retry_exponent == pytest.approx(2.0)
    assert corrector.on_high_uncertainty_return == "max"


def test_no_runtime_or_memory_section_uses_builtins(tmpdir):
    """When neither [slurmise.runtime] nor [slurmise.memory] exist, use built-in defaults."""
    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.myjob]
    [slurmise.job.myjob.variables]
    n = {type = "numeric"}
    """
    toml = write_toml(tmpdir, toml_str)
    config = SlurmiseConfiguration(toml)

    rt = config.get_runtime_corrector("myjob")
    mem = config.get_memory_corrector("myjob")

    assert rt.default == 60
    assert rt.minimum == 0
    assert rt.maximum == math.inf

    assert mem.default == 1000
    assert mem.minimum == 0
    assert mem.maximum == math.inf


@pytest.mark.parametrize(
    ("in_cwd", "in_home", "expected"),
    [
        (True, False, "cwd/slurmise.toml"),
        (False, True, "home/.slurmise/slurmise.toml"),
        (True, True, "cwd/slurmise.toml"),  # the working directory takes precedence
    ],
)
def test_find_config_file(tmp_path, monkeypatch, in_cwd, in_home, expected):
    cwd = tmp_path / "cwd"
    home = tmp_path / "home"
    cwd.mkdir()
    (home / ".slurmise").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(Path, "home", lambda: home)
    if in_cwd:
        (cwd / "slurmise.toml").touch()
    if in_home:
        (home / ".slurmise" / "slurmise.toml").touch()

    assert find_config_file() == tmp_path / expected


def test_find_config_file_missing(tmp_path, monkeypatch):
    """With no config in either location, a RuntimeError is raised."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="No slurmise.toml was found"):
        find_config_file()


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("", 0.2),
        ("retrain_warning_threshold = 0.5", 0.5),
        # disabling wins over any threshold, by making it unreachable
        ("retrain_warning_enable = false", float("inf")),
        ("retrain_warning_enable = false\n    retrain_warning_threshold = 0.5", float("inf")),
    ],
)
def test_retrain_warning_threshold(tmpdir, setting, expected):
    """A fifth of the fitted records may accumulate before warning, unless configured."""
    toml = write_toml(
        tmpdir,
        f"""
    [slurmise]
    base_dir = "slurmise_dir"
    {setting}
    """,
    )

    assert SlurmiseConfiguration(toml).retrain_warning_threshold == expected
