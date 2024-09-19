import pytest
from io import StringIO
from slurmise.config import SlurmiseConfiguration, JobSpec
from slurmise.job_data import JobData
from pprint import pprint

@pytest.fixture
def basic_toml(tmpdir):
    d = tmpdir.mkdir("slurmise_dir")
    f = d.join("basic.toml")

    toml_str = """
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_prefix = "nupack"
    job_spec = "monomer -T {threads:numeric} -C {complexity:category}"
    default_mem = 1000
    default_time = 60

    [slurmise.job.with_ignore]
    job_prefix = "nothing"
    job_spec = "-T {threads:numeric} -C {complexity:category} -i {ignore}"
    default_mem = 1000
    default_time = 60
    """

    f.write(toml_str)
    return f

def test_init_SlurmiseConfiguration(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    assert config.slurmise_base_dir == "slurmise_dir"
    assert len(config.jobs) == 2
    assert config.jobs['nupack']['job_prefix'] == "nupack"


def test_parse_job_cmd(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("nupack", "1234", "monomer -T 1 -C simple")

    assert job_data.job_name == "nupack"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}


def test_parse_job_cmd_with_ignore(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    job_data = config.parse_job_cmd("with_ignore", "1234", "-T 1 -C simple -i can't see me")

    assert job_data.job_name == "with_ignore"
    assert job_data.slurm_id == "1234"
    assert job_data.categorical == {"complexity": "simple"}
    assert job_data.numerical == {"threads": 1}

def test_parse_job_cmd_invalid(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command dimer -T 1 -C simple."):
        job_data = config.parse_job_cmd("nupack", "1234", "dimer -T 1 -C simple")

def test_parse_job_cmd_name_mismatch(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job oldpack not found in configuration."):
        job_data = config.parse_job_cmd("oldpack", "1234", "monomer -T 1 -C simple")

def test_parse_job_cmd_invalid_numeric(basic_toml):
    config = SlurmiseConfiguration(basic_toml)
    with pytest.raises(ValueError, match="Job spec monomer -T {threads:numeric} -C {complexity:category} does not match command monomer -T 1A -C simple."):
        job_data = config.parse_job_cmd("nupack", "1234", "monomer -T 1A -C simple")


# Tests for JobSpec
def test_job_spec_named_ignore():
    spec = JobSpec('cmd -T {threads:numeric} -i {named:ignore}')
    jd = spec.parse_job_cmd(
        JobData(
            job_name='test',
            cmd='cmd -T 10 -i asdf',
            ))
    assert jd == JobData(
        job_name='test',
        cmd='cmd -T 10 -i asdf',
        numerical={'threads': 10},
    )


def test_job_spec_unknown_kind():
    with pytest.raises(ValueError, match="Token kind double is unknown"):
        JobSpec('cmd -T {threads:double}')

def test_job_spec_token_with_no_name():
    with pytest.raises(ValueError, match="Token {numeric} has no name."):
        JobSpec('cmd -T {numeric}')
